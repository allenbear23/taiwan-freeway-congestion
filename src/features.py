# -*- coding: utf-8 -*-
"""把走廊面板 + 外部資料組成 24 個假設的配對觀測。

每筆觀測：x, y, year, unit（分組單位，用於跨年穩定性檢驗）。
若必要來源缺席，回傳空表與原因，絕不補假資料。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CORRIDORS, FREE_FLOW_SPEED, HOLIDAY_EVENTS, SLOT_MINUTES
from src.utils import LOGGER, daterange, slot_of


@dataclass
class Observation:
    hid: str
    dimension: str
    title: str
    x_name: str
    y_name: str
    kind: str                 # continuous | event
    required: tuple[str, ...]
    df: pd.DataFrame
    note: str = ""

    @property
    def available(self) -> bool:
        return len(self.df) >= 8 and self.df[["x", "y"]].dropna().shape[0] >= 8


def _obs(hid, dim, title, xn, yn, kind, required, rows, note="") -> Observation:
    df = pd.DataFrame(rows, columns=["x", "y", "year", "unit"])
    return Observation(hid, dim, title, xn, yn, kind, required, df, note)


def _num(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, corridor_panel: pd.DataFrame, section_panel: pd.DataFrame,
                 five_panel: pd.DataFrame, external: dict[str, tuple[pd.DataFrame, str]],
                 years: list[int], m06a: pd.DataFrame | None = None,
                 m06a_status: str = "DISABLED"):
        self.panel = corridor_panel.copy()
        self.sec = section_panel.copy()
        self.five = five_panel.copy() if five_panel is not None else pd.DataFrame()
        self.ext = external
        self.years = years
        self.m06a = m06a if m06a is not None else pd.DataFrame()
        self.m06a_status = m06a_status
        self.events = self._events()
        self.daily = self._daily_metrics()
        self.weather_hourly = self._weather_hourly()
        self.trends = self.ext.get("trends", (pd.DataFrame(), ""))[0]
        self.policy = self.ext.get("policy", (pd.DataFrame(), ""))[0]
        self.calendar = self.ext.get("calendar", (pd.DataFrame(), ""))[0]
        self._cache: dict[str, Observation] = {}

    # ------------------------------------------------------------------
    def _events(self) -> pd.DataFrame:
        rows = []
        for e in HOLIDAY_EVENTS:
            if e["year"] not in self.years:
                continue
            s, en = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
            ws, we = s - pd.Timedelta(days=1), en + pd.Timedelta(days=1)
            cal = self.ext.get("calendar", (pd.DataFrame(), ""))[0]
            if not cal.empty:
                sel = cal[(cal["date"] >= s) & (cal["date"] <= en)]
                rest = int((sel["day_type"].astype(str).str.contains("休|放假")).sum()) if "day_type" in sel else 0
            else:
                rest = sum(1 for d in daterange(s, en) if d.weekday() >= 5)
            total_days = (en - s).days + 1
            rows.append({
                "event_id": f"{e['year']}_{e['event']}", "year": e["year"], "event": e["event"],
                "start": s, "end": en, "window_start": ws, "window_end": we,
                "total_days": total_days, "rest_days": rest,
                "leave_days": max(0, total_days - rest), "tentative": e.get("tentative", False),
            })
        return pd.DataFrame(rows)

    def _daily_metrics(self) -> pd.DataFrame:
        if self.panel.empty:
            return pd.DataFrame()
        d = self.panel.copy()
        def _agg(g: pd.DataFrame) -> pd.Series:
            lvl = g["congestion_level"].fillna(-1).astype(int)
            return pd.Series({
                "total_volume": g["volume_15m"].sum(),
                "mean_speed": g["speed_kmh"].mean(),
                "congestion_hours": (lvl >= 2).sum() * 0.25,
                "severe_hours": (lvl >= 3).sum() * 0.25,
                "mean_travel_min": g["travel_time_min"].mean(),
                "p90_travel_min": g["travel_time_min"].quantile(0.9),
                "mean_heavy_share": g["heavy_share"].mean(),
            })
        daily = d.groupby(["date", "corridor_id"], as_index=False).apply(_agg, include_groups=False)
        return daily.reset_index(drop=True)

    def _weather_hourly(self) -> pd.DataFrame:
        w, _ = self.ext.get("weather", (pd.DataFrame(), ""))
        if w.empty:
            return pd.DataFrame()
        w = w.copy()
        w["datetime"] = pd.to_datetime(w["datetime"], errors="coerce")
        w = w.dropna(subset=["datetime"])
        w["date"] = w["datetime"].dt.normalize()
        w["slot"] = w["datetime"].dt.hour * 4 + w["datetime"].dt.minute // 15
        return w

    def _event_id_of(self, date: pd.Timestamp) -> str | None:
        if self.events.empty:
            return None
        m = self.events[(self.events["window_start"] <= date) & (self.events["window_end"] >= date)]
        return m.iloc[0]["event_id"] if len(m) else None

    # ------------------------------------------------------------------
    def _trend_mean(self, keyword: str, start: pd.Timestamp, end: pd.Timestamp, geo: str = "TW") -> float:
        if self.trends.empty or keyword not in self.trends.columns or "date" not in self.trends.columns:
            return np.nan
        d = self.trends[(self.trends["date"] >= start) & (self.trends["date"] <= end)]
        if "geo" in d.columns:
            d = d[d["geo"] == geo]
        return _num(d[keyword].mean())

    def _weather_at(self, station: str, start: pd.Timestamp, end: pd.Timestamp,
                    slot_min: int | None = None, slot_max: int | None = None) -> float:
        if self.weather_hourly.empty:
            return np.nan
        d = self.weather_hourly[(self.weather_hourly["date"] >= start) &
                                (self.weather_hourly["date"] <= end) &
                                (self.weather_hourly["station"] == station)]
        if slot_min is not None:
            d = d[(d["slot"] >= slot_min) & (d["slot"] <= slot_max)]
        col = "precip_mm" if "precip_mm" in d else None
        return _num(d[col].mean()) if col else np.nan

    def _policy_of(self, event_id: str) -> dict:
        if self.policy.empty:
            return {}
        e = self.events[self.events["event_id"] == event_id]
        if e.empty:
            return {}
        row = self.policy[self.policy["date"] == e.iloc[0]["start"]]
        if row.empty and "event" in self.policy.columns:
            row = self.policy[self.policy["event"] == e.iloc[0]["event"]]
        return row.iloc[0].to_dict() if len(row) else {}

    def _daily_metric(self, date, cid, col):
        if self.daily.empty:
            return np.nan
        m = self.daily[(self.daily["date"] == pd.Timestamp(date)) & (self.daily["corridor_id"] == cid)]
        return _num(m.iloc[0][col]) if len(m) else np.nan

    def _event_metric(self, event_id, cid, col, date_field="start"):
        e = self.events[self.events["event_id"] == event_id]
        if e.empty:
            return np.nan
        return self._daily_metric(e.iloc[0][date_field], cid, col)

    # ==================================================================
    # 1. 行為心理與預報反身性
    # ==================================================================
    def h1a(self) -> Observation:
        rows = []
        for e in self.events.itertuples():
            base = self._trend_mean("國5塞車", e.start - pd.Timedelta(days=28), e.start - pd.Timedelta(days=4))
            pre = self._trend_mean("國5塞車", e.start - pd.Timedelta(days=3), e.start - pd.Timedelta(days=1))
            x = pre / base if np.isfinite(pre) and np.isfinite(base) and base > 0 else np.nan
            p = self.panel[(self.panel["date"] == e.start) & (self.panel["corridor_id"] == "N5_SB")]
            early = p[(p["time_slot_15m"] >= 16) & (p["time_slot_15m"] < 26)]["volume_15m"].sum()
            total = p[(p["time_slot_15m"] >= 16) & (p["time_slot_15m"] < 48)]["volume_15m"].sum()
            y = early / total if total > 0 else np.nan
            if np.isfinite(x) and np.isfinite(y):
                rows.append([x, y, e.year, e.event_id])
        return _obs("1A", "行為心理/反身性", "預報反身性（尖峰左移）", "搜尋飆升倍率", "早鳥車流佔比",
                    "continuous", ("trends", "panel"), rows, "錨點比值已跨批次縫合")

    def h1b(self) -> Observation:
        prov, st = self.ext.get("provincial", (pd.DataFrame(), ""))
        if prov.empty or not {"t9_volume", "t9_travel_min"}.issubset(prov.columns):
            return _obs("1B", "行為心理/反身性", "替代道路分流實效", "台9線搜尋熱度", "國5/台9流量比",
                        "continuous", ("provincial", "trends"), [], f"需台9線流量檔；{st}")
        rows = []
        for e in self.events.itertuples():
            x = self._trend_mean("台9線", e.start - pd.Timedelta(days=7), e.start)
            d = prov[(prov["date"] >= e.start) & (prov["date"] <= e.end)]
            t9 = _num(d["t9_volume"].mean())
            n5 = self._event_metric(e.event_id, "N5_SB", "total_volume")
            y = n5 / t9 if np.isfinite(n5) and np.isfinite(t9) and t9 > 0 else np.nan
            if np.isfinite(x) and np.isfinite(y):
                rows.append([x, y, e.year, e.event_id])
        return _obs("1B", "行為心理/反身性", "替代道路分流實效", "台9線搜尋熱度", "國5/台9流量比",
                    "continuous", ("provincial", "trends"), rows)

    def h1c(self) -> Observation:
        rows = []
        for e in self.events.itertuples():
            x = self._trend_mean("宜蘭住宿", e.start - pd.Timedelta(days=14), e.start - pd.Timedelta(days=7))
            y = self._event_metric(e.event_id, "N5_SB", "total_volume")
            if np.isfinite(x) and np.isfinite(y):
                rows.append([x, y, e.year, e.event_id])
        return _obs("1C", "行為心理/反身性", "目的地搜尋領先指標", "住宿搜尋指數(D-14~D-7)", "連假總旅次量",
                    "continuous", ("trends", "panel"), rows)

    # ==================================================================
    # 2. 交管政策邊界效應
    # ==================================================================
    def h2a(self) -> Observation:
        if self.policy.empty or "hov_release_min" not in self.policy.columns:
            return _obs("2A", "交管政策", "高乘載水壩洩洪效應", "距解除分鐘數", "車速驟降斜率/流量倍數",
                        "continuous", ("policy", "sec"), [],
                        "需 holiday_controls.csv.hov_release_min（分鐘，自 00:00 起算）")
        rows = []
        for e in self.events.itertuples():
            pol = self._policy_of(e.event_id)
            if not pol or _num(pol.get("hov_active")) != 1:
                continue
            rel = _num(pol.get("hov_release_min"))
            if not np.isfinite(rel):
                continue
            for day in daterange(e.start, e.end):
                sub = self.sec[(self.sec["date"] == day) & (self.sec["corridor_id"] == "N5_SB")]
                if sub.empty:
                    continue
                # 取雪隧南向入口（km 最接近 28.7）的區間
                k = (sub["km_from"] - 28.7).abs().idxmin()
                target = sub.loc[k, "km_from"]
                s = sub[sub["km_from"] == target].sort_values("time_slot_15m")
                s["minute"] = s["time_slot_15m"] * SLOT_MINUTES + SLOT_MINUTES / 2
                s = s[(s["minute"] >= rel - 30) & (s["minute"] <= rel + 60)].copy()
                if len(s) < 4:
                    continue
                s["dv"] = s["speed_kmh"].diff()
                s["vr"] = s["volume_15m"] / s["volume_15m"].shift(1)
                for r in s.dropna(subset=["dv"]).itertuples():
                    rows.append([r.minute - rel, -_num(r.dv), e.year, f"{e.event_id}_{day:%m%d}"])
        return _obs("2A", "交管政策", "高乘載水壩洩洪效應", "距解除分鐘數", "車速驟降幅度",
                    "continuous", ("policy", "sec"), rows)

    def h2b(self) -> Observation:
        if self.policy.empty or "free_toll_end_min" not in self.policy.columns:
            return _obs("2B", "交管政策", "凌晨免收費壓線搶閘", "免收費旗標", "04–05 交通量/低速發生率",
                        "event", ("policy", "panel"), [],
                        "需 holiday_controls.csv.free_toll_end_min")
        rows = []
        for e in self.events.itertuples():
            pol = self._policy_of(e.event_id)
            flag = 1 if _num(pol.get("free_toll_active")) == 1 and np.isfinite(_num(pol.get("free_toll_end_min"))) else 0
            p = self.panel[(self.panel["date"] == e.start) & (self.panel["corridor_id"] == "N5_SB")]
            q = p[(p["time_slot_15m"] >= 16) & (p["time_slot_15m"] < 20)]
            if q.empty:
                continue
            vol = _num(q["volume_15m"].sum())
            slow = float((q["speed_kmh"] < 60).mean())
            rows.append([flag, vol, e.year, e.event_id])
            rows.append([flag, slow, e.year, e.event_id + "_slow"])
        return _obs("2B", "交管政策", "凌晨免收費壓線搶閘", "免收費旗標", "04–05 交通量/低速率",
                    "event", ("policy", "panel"), rows, "each event 產生兩筆 Y（量、率）")

    def h2c(self) -> Observation:
        if self.policy.empty or "ramp_closure_gantry" not in self.policy.columns:
            return _obs("2C", "交管政策", "匝道封閉鄰近虹吸", "匝道封閉旗標", "相鄰區間旅行時間增幅",
                        "event", ("policy", "sec"), [], "需 holiday_controls.csv.ramp_closure_gantry")
        rows = []
        for e in self.events.itertuples():
            pol = self._policy_of(e.event_id)
            g = str(pol.get("ramp_closure_gantry", ""))
            if not g or g == "nan":
                continue
            km = _num(pol.get("ramp_closure_km"))
            if not np.isfinite(km):
                continue
            s = self.sec[(self.sec["date"] == e.start) & (self.sec["corridor_id"] == "N5_SB")]
            near = s[(s["km_from"] >= km - 3) & (s["km_from"] <= km + 3)]
            if near.empty:
                continue
            before = near[near["time_slot_15m"] < 60]["travel_time_min"].mean()
            after = near[near["time_slot_15m"] >= 60]["travel_time_min"].mean()
            if np.isfinite(before) and np.isfinite(after) and before > 0:
                rows.append([1.0, after / before - 1, e.year, e.event_id])
        return _obs("2C", "交管政策", "匝道封閉鄰近虹吸", "匝道封閉旗標", "旅行時間增幅",
                    "event", ("policy", "sec"), rows)

    def h2d(self) -> Observation:
        if self.policy.empty or "n3_discount" not in self.policy.columns:
            return _obs("2D", "交管政策", "國3差別費率轉移", "國3八折旗標", "國3佔平行流量比",
                        "event", ("policy", "panel"), [], "需 holiday_controls.csv.n3_discount")
        n3 = self.panel[(self.panel["corridor_id"].astype(str).str.startswith("N3"))]
        rows = []
        for e in self.events.itertuples():
            flag = _num(self._policy_of(e.event_id).get("n3_discount"))
            d = n3[n3["date"] == e.start]
            n3v = _num(d["volume_15m"].mean())
            n1v = self._daily_metric(e.start, "N1_SB", "total_volume")
            if np.isfinite(flag) and np.isfinite(n3v) and np.isfinite(n1v) and (n3v + n1v) > 0:
                rows.append([flag, n3v / (n3v + n1v), e.year, e.event_id])
        return _obs("2D", "交管政策", "國3差別費率轉移", "國3八折旗標", "國3平行流量佔比",
                    "event", ("policy", "panel"), rows)

    # ==================================================================
    # 3. 跨區氣候推拉
    # ==================================================================
    def h3a(self) -> Observation:
        rows = []
        for e in self.events.itertuples():
            for day in daterange(e.start, e.end):
                origin = self._weather_at("TPE", day, day, 24, 72)
                dest = self._weather_at("YLP", day, day, 24, 72)
                if not (np.isfinite(origin) and np.isfinite(dest)):
                    continue
                y = self._daily_metric(day, "N5_SB", "total_volume")
                if np.isfinite(y):
                    rows.append([origin - dest, y, e.year, f"{e.event_id}_{day:%m%d}"])
        return _obs("3A", "跨區氣候", "天氣剪刀差（逃離雨天）", "台北雨量−宜蘭雨量", "南向總流量",
                    "continuous", ("weather", "panel"), rows)

    def h3b(self) -> Observation:
        rows = []
        for e in self.events.itertuples():
            for day in daterange(e.start, e.end):
                tpe = self._weather_at("TPE", day, day, 24, 72)
                ylp = self._weather_at("YLP", day, day, 24, 72)
                if not (np.isfinite(tpe) and np.isfinite(ylp)):
                    continue
                flag = 1.0 if tpe < 0.1 and ylp >= 1.0 else 0.0
                n5 = self._daily_metric(day, "N5_SB", "total_volume")
                n1 = self._daily_metric(day, "N1_SB", "total_volume")
                if np.isfinite(n5) and np.isfinite(n1):
                    rows.append([flag, n1 - n5, e.year, f"{e.event_id}_{day:%m%d}"])
        return _obs("3B", "跨區氣候", "棄遊轉向效應", "台北晴+宜蘭雨旗標", "國1流量−國5流量",
                    "event", ("weather", "panel"), rows)

    def h3c(self) -> Observation:
        if self.weather_hourly.empty or "date" not in self.weather_hourly.columns:
            return _obs("3C", "跨區氣候", "前序週末降雨壓抑釋放", "前14日週末累積雨量", "首日超額流量倍率",
                        "continuous", ("weather", "panel"), [],
                        self.ext.get("weather", (None, "MANUAL_NEEDED"))[1])
        rows = []
        for e in self.events.itertuples():
            s = e.start - pd.Timedelta(days=14)
            prev = self.weather_hourly[
                (self.weather_hourly["date"] >= s) & (self.weather_hourly["date"] < e.start) &
                (self.weather_hourly["date"].dt.dayofweek >= 5)
            ]
            x = _num(prev["precip_mm"].sum())
            base = np.nanmean([self._daily_metric(d, "N5_SB", "total_volume")
                               for d in daterange(e.start, e.end)])
            first = self._daily_metric(e.start, "N5_SB", "total_volume")
            y = first / base if np.isfinite(base) and base > 0 else np.nan
            if np.isfinite(x) and np.isfinite(y):
                rows.append([x, y, e.year, e.event_id])
        return _obs("3C", "跨區氣候", "前序週末降雨壓抑釋放", "前14日週末累積雨量", "首日超額流量倍率",
                    "continuous", ("weather", "panel"), rows)

    def h3d(self) -> Observation:
        if self.weather_hourly.empty or self.panel.empty:
            return _obs("3D", "跨區氣候", "強降雨容量崩跌臨界點", "雨量分級×流量", "跌破40km/h",
                        "continuous", ("weather", "panel"), [])
        w = self.weather_hourly[self.weather_hourly["station"] == "TPE"]
        if w.empty:
            return _obs("3D", "跨區氣候", "強降雨容量崩跌臨界點", "雨量分級×流量", "跌破40km/h",
                        "continuous", ("weather", "panel"), [])
        w = w[["date", "slot", "precip_mm"]].drop_duplicates(["date", "slot"])
        w["rain_class"] = pd.cut(w["precip_mm"], bins=[-np.inf, 0.1, 1, 5, 10, np.inf],
                                 labels=False, include_lowest=True).astype(float)
        p = self.panel[self.panel["corridor_id"] == "N5_SB"][
            ["date", "time_slot_15m", "volume_15m", "speed_kmh"]].rename(
                columns={"time_slot_15m": "slot"})
        j = p.merge(w, on=["date", "slot"], how="inner").dropna(subset=["rain_class", "volume_15m"])
        if j.empty:
            return _obs("3D", "跨區氣候", "強降雨容量崩跌臨界點", "雨量分級×流量", "跌破40km/h",
                        "continuous", ("weather", "panel"), [])
        j["x"] = j["rain_class"] * j["volume_15m"]
        j["y"] = (j["speed_kmh"] < 40).astype(float)
        j["year"] = j["date"].dt.year
        j = j.sample(min(4000, len(j)), random_state=42)
        out = j[["x", "y", "year"]].copy()
        out["unit"] = j["date"].dt.strftime("%Y%m%d")
        return _obs("3D", "跨區氣候", "強降雨容量崩跌臨界點", "雨量分級×流量", "跌破40km/h",
                    "continuous", ("weather", "panel"), out.values.tolist())

    # ==================================================================
    # 4. 跨運具與出國替代
    # ==================================================================
    def h4a(self) -> Observation:
        ap, st = self.ext.get("airport", (pd.DataFrame(), ""))
        if ap.empty or "departures" not in ap.columns:
            return _obs("4A", "跨運具/出國", "出國潮抽乾國旅", "出境人次倍率", "國5壅塞小時數",
                        "continuous", ("airport", "panel"), [], st)
        rows = []
        ap = ap.copy()
        ap["date"] = pd.to_datetime(ap["date"])
        mean_dep = _num(ap["departures"].mean())
        for e in self.events.itertuples():
            d = ap[(ap["date"] >= e.start) & (ap["date"] <= e.end)]
            dep = _num(d["departures"].mean())
            ratio = dep / mean_dep if np.isfinite(dep) and np.isfinite(mean_dep) and mean_dep > 0 else np.nan
            y = self._event_metric(e.event_id, "N5_SB", "congestion_hours")
            if np.isfinite(ratio) and np.isfinite(y):
                rows.append([ratio, y, e.year, e.event_id])
        return _obs("4A", "跨運具/出國", "出國潮抽乾國旅", "出境人次倍率", "國5壅塞小時數",
                    "continuous", ("airport", "panel"), rows)

    def h4b(self) -> Observation:
        ap, st = self.ext.get("airport", (pd.DataFrame(), ""))
        n2 = self.panel[self.panel["corridor_id"].astype(str).str.startswith("N2")]
        if ap.empty or "hour" not in ap.columns or n2.empty:
            return _obs("4B", "跨運具/出國", "送機潮引爆國2", "桃機整點出入境", "國2車速/旅行時間",
                        "continuous", ("airport", "N2"), [],
                        f"{st}；且 TISVCLOUD M05A 無國2 門架（需 VD 補檔）")
        rows = []
        for r in ap.itertuples():
            d = pd.Timestamp(r.date)
            slot = int(r.hour) * 4
            p = n2[(n2["date"] == d) & (n2["time_slot_15m"] == slot)]
            if p.empty:
                continue
            rows.append([_num(getattr(r, "departures", np.nan)) + _num(getattr(r, "arrivals", 0)),
                         _num(p.iloc[0]["travel_time_min"]), d.year, f"{d:%Y%m%d}{slot:02d}"])
        return _obs("4B", "跨運具/出國", "送機潮引爆國2", "桃機整點出入境", "國2旅行時間",
                    "continuous", ("airport", "N2"), rows)

    def h4c(self) -> Observation:
        hsr, st = self.ext.get("hsr", (pd.DataFrame(), ""))
        if hsr.empty or "load_factor" not in hsr.columns:
            return _obs("4C", "跨運具/出國", "高鐵滿載外溢", "高鐵載客率/加班車比例", "國1湖口瓶頸壅塞時數",
                        "continuous", ("hsr", "sec"), [], st)
        rows = []
        hsr = hsr.copy()
        hsr["date"] = pd.to_datetime(hsr["date"])
        if self.sec.empty:
            return _obs("4C", "跨運具/出國", "高鐵滿載外溢", "高鐵載客率/加班車比例", "國1湖口瓶頸壅塞時數",
                        "continuous", ("hsr", "sec"), [])
        hk = self.sec[(self.sec["corridor_id"] == "N1_SB") &
                      (self.sec["km_from"].between(80, 95))]
        for e in self.events.itertuples():
            r = hsr[hsr["date"] == e.start]
            if r.empty:
                continue
            x = _num(r.iloc[0]["load_factor"])
            d = hk[hk["date"] == e.start]
            if d.empty:
                continue
            y = float((d["speed_kmh"] < 40).mean())
            if np.isfinite(x) and np.isfinite(y):
                rows.append([x, y, e.year, e.event_id])
        return _obs("4C", "跨運具/出國", "高鐵滿載外溢", "高鐵載客率", "國1湖口低速比例",
                    "continuous", ("hsr", "sec"), rows)

    # ==================================================================
    # 5. 日曆拓撲與拼假經濟
    # ==================================================================
    def h5a(self) -> Observation:
        rows = []
        for e in self.events.itertuples():
            ratio = e.total_days / max(e.leave_days, 1)
            vols = []
            for dd in (-2, -1):
                day = e.start + pd.Timedelta(days=dd)
                p = self.panel[(self.panel["date"] == day) & (self.panel["corridor_id"] == "N5_SB")]
                evening = p[(p["time_slot_15m"] >= 68) & (p["time_slot_15m"] < 76)]["volume_15m"].sum()
                allday = p["volume_15m"].sum()
                if allday > 0:
                    vols.append(evening / allday)
            y = float(np.mean(vols)) if vols else np.nan
            if np.isfinite(ratio) and np.isfinite(y):
                rows.append([ratio, y, e.year, e.event_id])
        return _obs("5A", "日曆拓撲", "請假攻略槓桿", "總休假/特休天數", "節前傍晚分流佔比",
                    "continuous", ("calendar", "panel"), rows)

    def h5b(self) -> Observation:
        if self.calendar.empty:
            return _obs("5B", "日曆拓撲", "補班日後補償反彈", "補班週旗標", "交通量季節增幅",
                        "event", ("calendar", "panel"), [], self.ext.get("calendar", (None, ""))[1])
        rows = []
        mean_v = self.daily[self.daily["corridor_id"] == "N5_SB"]["total_volume"].mean() if not self.daily.empty else np.nan
        for e in self.events.itertuples():
            cal = self.calendar[(self.calendar["date"] >= e.start - pd.Timedelta(days=7)) &
                                (self.calendar["date"] < e.start)]
            note = " ".join(cal["note"].astype(str)) if "note" in cal else ""
            flag = 1.0 if ("補班" in note or "補行上班" in note) else 0.0
            v = self._event_metric(e.event_id, "N5_SB", "total_volume")
            y = v / mean_v if np.isfinite(v) and np.isfinite(mean_v) and mean_v > 0 else np.nan
            if np.isfinite(y):
                rows.append([flag, y, e.year, e.event_id])
        return _obs("5B", "日曆拓撲", "補班日後補償反彈", "補班週旗標", "交通量季節增幅",
                    "event", ("calendar", "panel"), rows)

    def h5c(self) -> Observation:
        rows = []
        for e in self.events.itertuples():
            day = e.end - pd.Timedelta(days=1)
            p = self.panel[(self.panel["date"] == day) & (self.panel["corridor_id"] == "N5_NB")]
            q = p[(p["time_slot_15m"] >= 48) & (p["time_slot_15m"] < 68)]
            if q.empty:
                continue
            y = float((q["speed_kmh"] < 60).mean())
            rows.append([float(e.total_days), y, e.year, e.event_id])
        return _obs("5C", "日曆拓撲", "長連假提早北返分流", "連假總天數", "收假前日下午北向低速比例",
                    "continuous", ("panel",), rows)

    # ==================================================================
    # 6. 微觀車流
    # ==================================================================
    def h6a(self) -> Observation:
        if self.m06a.empty:
            return _obs("6A", "微觀車流", "長途跨區旅次脆弱度", ">100km旅次佔比", "速度驟降率",
                        "continuous", ("M06A", "panel"), [], self.m06a_status)
        m = self.m06a[self.m06a["VehicleType"] == 31].copy()
        m["long"] = m["TripLength"] > 100
        rows = []
        for e in self.events.itertuples():
            d = m[(m["date"] >= e.start) & (m["date"] <= e.end)]
            if d.empty:
                continue
            x = float(d["long"].mean())
            p = self.panel[(self.panel["date"] == e.start) & (self.panel["corridor_id"] == "N5_SB")]
            y = float((p["speed_kmh"].diff() < -15).mean())
            rows.append([x, y, e.year, e.event_id])
        return _obs("6A", "微觀車流", "長途跨區旅次脆弱度", ">100km旅次佔比", "速度驟降率",
                    "continuous", ("M06A", "panel"), rows)

    def h6b(self) -> Observation:
        if self.sec.empty:
            return _obs("6B", "微觀車流", "重車混流干擾比", "重車流量佔比", "爬坡段車速折減",
                        "continuous", ("sec",), [])
        climb = self.sec[(self.sec["corridor_id"] == "N5_SB") &
                         (self.sec["km_from"].between(14, 30))]
        daily = self.daily[self.daily["corridor_id"] == "N5_SB"]
        rows = []
        for r in daily.itertuples():
            d = climb[climb["date"] == r.date]
            if d.empty:
                continue
            # 重車佔比來自走廊日統計（M05A 全車種），爬坡段車速折減來自區間面板
            x = _num(getattr(r, "mean_heavy_share", np.nan))
            sp = _num(d["speed_kmh"].mean())
            y = FREE_FLOW_SPEED - sp if np.isfinite(sp) else np.nan
            if np.isfinite(x) and np.isfinite(y):
                rows.append([x, y, r.date.year, f"{r.date:%Y%m%d}"])
        return _obs("6B", "微觀車流", "重車混流干擾比", "重車流量佔比", "爬坡段車速折減",
                    "continuous", ("sec",), rows)

    def h6c(self) -> Observation:
        if self.five.empty:
            return _obs("6C", "微觀車流", "臨界減慢前兆訊號", "前30分車速標準差", "未來30分跌破40",
                        "continuous", ("five",), [])
        f = self.five[self.five["corridor_id"] == "N5_SB"].copy()
        rows = []
        for (date, gfrom, gto), g in f.groupby(["date", "GantryFrom", "GantryTo"]):
            g = g.sort_values("time_slot_5m")
            prev_std = g["speed_kmh"].rolling(6, min_periods=4).std().shift(1)
            future_min = g["speed_kmh"].iloc[::-1].rolling(6, min_periods=4).min().iloc[::-1].shift(-1)
            tmp = pd.DataFrame({"x": prev_std, "y": (future_min < 40).astype(float),
                                "year": date.year, "unit": f"{date:%Y%m%d}_{gfrom}"}).dropna()
            rows.extend(tmp.values.tolist())
        if len(rows) > 6000:
            idx = np.random.default_rng(42).choice(len(rows), 6000, replace=False)
            rows = [rows[i] for i in idx]
        return _obs("6C", "微觀車流", "臨界減慢前兆訊號", "前30分車速標準差", "未來30分跌破40",
                    "continuous", ("five",), rows)

    # ==================================================================
    # 7. 在地文化與大型活動
    # ==================================================================
    def h7a(self) -> Observation:
        lunar, st = self.ext.get("lunar", (pd.DataFrame(), ""))
        if lunar.empty:
            return _obs("7A", "在地文化", "農民曆吉日效應", "宜嫁娶/宜祭祀", "10–16 時跨區流量",
                        "event", ("lunar", "panel"), [], st)
        lunar = lunar.copy()
        lunar["date"] = pd.to_datetime(lunar["date"])
        rows = []
        for r in lunar.itertuples():
            if self._event_id_of(r.date) is None:
                continue
            q = self.panel[(self.panel["date"] == r.date) &
                           (self.panel["time_slot_15m"].between(40, 64))]
            y = _num(q["volume_15m"].sum())
            x = float(bool(r.yi_marriage) or bool(r.yi_ritual))
            if np.isfinite(y):
                rows.append([x, y, r.date.year, f"{r.date:%Y%m%d}"])
        return _obs("7A", "在地文化", "農民曆吉日效應", "宜嫁娶/宜祭祀", "10–16 時跨區流量",
                    "event", ("lunar", "panel"), rows)

    def h7b(self) -> Observation:
        ev, st = self.ext.get("mega_events", (pd.DataFrame(), ""))
        if ev.empty or self.sec.empty:
            return _obs("7B", "在地文化", "萬人場館與繞境衝擊", "距散場小時數", "鄰近交流道車速驛降",
                        "event", ("mega_events", "sec"), [], st)
        ev = ev.copy()
        ev["datetime"] = pd.to_datetime(ev["datetime"])
        rows = []
        for r in ev.itertuples():
            d = r.datetime.normalize()
            km = _num(getattr(r, "km", np.nan))
            sub = self.sec[(self.sec["date"] == d) & (self.sec["corridor_id"] == "N5_SB")]
            if not np.isfinite(km) or sub.empty:
                continue
            near = sub[(sub["km_from"] >= km - 2) & (sub["km_from"] <= km + 2)]
            if near.empty:
                continue
            base = near[near["time_slot_15m"] < 40]["speed_kmh"].mean()
            post = near[near["time_slot_15m"].between(40, 52)]["speed_kmh"].mean()
            if np.isfinite(base) and np.isfinite(post):
                rows.append([_num(getattr(r, "hours_from_end", 0)), base - post, d.year, str(r.datetime)])
        return _obs("7B", "在地文化", "萬人場館與繞境衝擊", "距散場小時數", "車速驟降幅度",
                    "event", ("mega_events", "sec"), rows)

    # ==================================================================
    # 8. 空間拓撲與回堵傳遞
    # ==================================================================
    def h8a(self) -> Observation:
        if self.sec.empty:
            return _obs("8A", "空間拓撲", "瓶頸回堵衝擊波延遲", "下游跌破40時點", "上游延遲分鐘",
                        "continuous", ("sec",), [])
        rows = []
        s = self.sec[self.sec["corridor_id"] == "N5_SB"]
        for date, g in s.groupby("date"):
            cross = {}
            for km, k in g.groupby("km_from"):
                hit = k[k["speed_kmh"] < 40]["time_slot_15m"].min()
                if np.isfinite(hit):
                    cross[float(km)] = float(hit)
            if len(cross) < 3:
                continue
            ordered = sorted(cross.items(), key=lambda t: -t[0])  # 下游（大 km）優先
            for kk in (1, 2, 3):
                if kk >= len(ordered):
                    continue
                down_km, down_t = ordered[0]
                up_km, up_t = ordered[kk]
                rows.append([down_t * SLOT_MINUTES, (up_t - down_t) * SLOT_MINUTES,
                             date.year, f"{date:%Y%m%d}_k{kk}"])
        return _obs("8A", "空間拓撲", "瓶頸回堵衝擊波延遲", "下游跌破40時點(分)", "上游第k門架延遲(分)",
                    "continuous", ("sec",), rows)

    def h8b(self) -> Observation:
        prov, st = self.ext.get("provincial", (pd.DataFrame(), ""))
        if prov.empty or "t9_travel_min" not in prov.columns:
            return _obs("8B", "空間拓撲", "主線與省道同步崩潰點", "國5車速區間", "台9線30分後旅行時間增幅",
                        "continuous", ("provincial", "panel"), [], st)
        p = self.panel[self.panel["corridor_id"] == "N5_SB"][
            ["date", "time_slot_15m", "speed_kmh"]].copy()
        p["slot_lag"] = p["time_slot_15m"] + 2
        t9 = prov.copy()
        t9["date"] = pd.to_datetime(t9["date"])
        t9["slot"] = t9["datetime"].dt.hour * 4 + t9["datetime"].dt.minute // 15 if "datetime" in t9 else 0
        j = p.merge(t9[["date", "slot", "t9_travel_min"]], left_on=["date", "slot_lag"],
                    right_on=["date", "slot"], how="inner")
        j["y"] = j.groupby("date")["t9_travel_min"].diff()
        rows = j.dropna(subset=["y"])[["speed_kmh", "y", "date"]].assign(
            year=lambda d: d["date"].dt.year, unit=lambda d: d["date"].dt.strftime("%Y%m%d")
        ).values.tolist()
        return _obs("8B", "空間拓撲", "主線與省道同步崩潰點", "國5車速", "台9線旅行時間增幅",
                    "continuous", ("provincial", "panel"), rows)

    # ------------------------------------------------------------------
    def build_all(self) -> list[Observation]:
        builders = [getattr(self, f"h{hid}") for hid in [
            "1a", "1b", "1c", "2a", "2b", "2c", "2d", "3a", "3b", "3c", "3d",
            "4a", "4b", "4c", "5a", "5b", "5c", "6a", "6b", "6c", "7a", "7b", "8a", "8b",
        ]]
        out = []
        for fn in builders:
            try:
                o = fn()
            except Exception as e:  # noqa: BLE001
                LOGGER.exception("假設 %s 組裝失敗", fn.__name__)
                o = _obs(fn.__name__[1:].upper(), "未知", str(fn.__name__), "", "",
                         "continuous", (), [], f"ERROR: {e}")
            out.append(o)
            LOGGER.info("假設 %-3s %-9s n=%-5d %s", o.hid,
                        "原始資料足" if o.available else "原始不足", len(o.df), o.note[:60])
        return out
