# -*- coding: utf-8 -*-
"""階段二建模特徵工程（只讀外部資料，所有特徵在出發前 D-1 23:59 前已可取得）。

防洩漏守則
----------
* 当日交通觀測（speed_kmh / volume_15m / travel_time_min / heavy_share）**绝不**
  作為特徵，只當標的；連歷史統計也只由 train（2023–2024）擬合。
* 氣象、Google Trends 一律取 D-1（或更早的前序週末）的值；train 與 test 走
  完全相同的程式碼路徑，不使用當日同時段觀測。
* 日曆／連假／交管規則為事件發生前即公告的知識，可用於特徵；交管分鐘數
  來自 policy 檔（若不可用則直接不建立該特徵）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import CONGESTION_LEVELS, HOLIDAY_EVENTS, SLOT_MINUTES

YEARS: tuple[int, ...] = (2023, 2024, 2025)
SLOTS_PER_DAY: int = int(24 * 60 / SLOT_MINUTES)
WEATHER_STATIONS: tuple[str, ...] = ("TPE", "YLP", "TAY")


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _find_col(columns, *needles: str) -> str | None:
    """先精確比對，再含入比對（大小寫不敏感）。"""
    col_list = [str(c) for c in columns]
    for needle in needles:
        for c in col_list:
            if c.strip().lower() == needle.strip().lower():
                return c
    for needle in needles:
        for c in col_list:
            if needle.strip().lower() in c.strip().lower():
                return c
    return None


# ---------------------------------------------------------------------------
# 1. 讀取面板與標的驗證
# ---------------------------------------------------------------------------
def level_from_speed(speed: pd.Series) -> pd.Series:
    """依 config.CONGESTION_LEVELS 由車速推導 4 等級（與 speed 一致才使用）。"""
    s = _to_numeric(speed).to_numpy(dtype=float)
    out = np.full(s.shape, np.nan, dtype=float)
    for threshold, label in CONGESTION_LEVELS:  # [(80,0), (60,1), (40,2), (0,3)]
        mask = (s >= threshold) & np.isnan(out)
        out[mask] = float(label)
    return pd.Series(out, index=speed.index)


def load_panel(root: Path) -> tuple[pd.DataFrame, dict]:
    """組合所有 days/YYYYMMDD/corridor.parquet 成單一 panel。"""
    day_dir = root / "data" / "processed" / "days"
    paths = sorted(day_dir.glob("*/corridor.parquet"))
    if not paths:
        raise FileNotFoundError(f"找不到走廊面板：{day_dir}/*/corridor.parquet")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["time_slot_15m"] = _to_numeric(df["time_slot_15m"]).astype(int)
    df["corridor_id"] = df["corridor_id"].astype(str)
    df["congestion_level"] = _to_numeric(df["congestion_level"])
    df = df.sort_values(["date", "corridor_id", "time_slot_15m"]).reset_index(drop=True)

    derived = level_from_speed(df["speed_kmh"]).to_numpy(dtype=float)
    observed = df["congestion_level"].to_numpy(dtype=float)
    finite = np.isfinite(derived)
    mismatch = int(np.nansum(finite & (derived != observed)))
    meta = {
        "n_parquet_files": len(paths),
        "panel_rows": int(len(df)),
        "panel_days": int(df["date"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "congestion_level_mismatch_vs_speed": mismatch,
        "corridors": sorted(df["corridor_id"].unique().tolist()),
    }
    if mismatch:
        print(f"[警告] congestion_level 與 speed 不一致的列數：{mismatch}（標的仍使用原欄位）")
    return df, meta


# ---------------------------------------------------------------------------
# 2. 外部資料（缺檔或缺欄不補值，只標記不可用）
# ---------------------------------------------------------------------------
def load_calendar(root: Path, years=YEARS) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    problems: list[str] = []
    for year in years:
        path = root / "data" / "external" / "calendar" / f"office_calendar_{year}.csv"
        if not path.exists():
            problems.append(f"MANUAL_NEEDED:{_rel(root, path)}")
            continue
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"UNREADABLE:{_rel(root, path)}({exc})")
            continue
        dcol = _find_col(raw.columns, "西元日期", "日期", "date")
        if dcol is None:
            problems.append(f"UNUSABLE:{_rel(root, path)} 缺少日期欄")
            continue
        fcol = _find_col(raw.columns, "是否放假", "放假", "day_type")
        ncol = _find_col(raw.columns, "備註", "note")
        dates = pd.to_datetime(_to_numeric(raw[dcol]).astype("Int64").astype(str),
                               format="%Y%m%d", errors="coerce")
        out = pd.DataFrame({"date": dates})
        if fcol is not None:
            out["is_day_off"] = _to_numeric(raw[fcol]).fillna(0).gt(0)
        else:
            out["is_day_off"] = np.nan
        out["note"] = (raw[ncol].fillna("").astype(str) if ncol is not None else "")
        frames.append(out.dropna(subset=["date"]))
    if not frames:
        return pd.DataFrame(columns=["date", "is_day_off", "note"]), "; ".join(problems)
    status = "OK" if not problems else "PARTIAL: " + "; ".join(problems)
    return pd.concat(frames, ignore_index=True), status


def load_weather(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """每站回傳日統計（precip_mean / precip_sum / temp_mean），索引為 date。"""
    daily_by_station: dict[str, pd.DataFrame] = {}
    status: dict[str, str] = {}
    for st in WEATHER_STATIONS:
        path = root / "data" / "external" / "weather" / f"hourly_{st}.csv"
        if not path.exists():
            status[st] = f"MANUAL_NEEDED:{_rel(root, path)}"
            continue
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            status[st] = f"UNREADABLE:{_rel(root, path)}({exc})"
            continue
        tcol = _find_col(raw.columns, "datetime", "time", "日期", "obs_time")
        rcol = _find_col(raw.columns, "precip_mm", "precip", "rain", "降水", "雨量")
        mcol = _find_col(raw.columns, "temp_c", "temp", "氣溫")
        if tcol is None:
            status[st] = f"UNUSABLE:{_rel(root, path)} 缺少時間欄"
            continue
        hourly = pd.DataFrame({
            "datetime": pd.to_datetime(raw[tcol], errors="coerce"),
            "precip_mm": _to_numeric(raw[rcol]) if rcol else np.nan,
            "temp_c": _to_numeric(raw[mcol]) if mcol else np.nan,
        }).dropna(subset=["datetime"])
        if "station" in raw.columns and len(hourly):
            # 檔名已指定站碼；若檔內有多站，過濾成同名站，避免混站。
            obs_station = raw.loc[hourly.index, "station"].astype(str).str.upper()
            sel = obs_station.str.contains(st, na=False)
            if sel.sum() > 0:
                hourly = hourly.loc[sel].copy()
        hourly["date"] = hourly["datetime"].dt.normalize()
        daily = hourly.groupby("date", sort=True).agg(
            precip_mean=("precip_mm", "mean"),
            precip_sum=("precip_mm", "sum"),
            temp_mean=("temp_c", "mean"),
        )
        if daily.empty:
            status[st] = f"EMPTY:{_rel(root, path)}"
            continue
        daily_by_station[st] = daily
        status[st] = "OK"
    return daily_by_station, status


def load_trends(root: Path) -> tuple[pd.Series, str, list[str]]:
    """回傳 D 日的『搜尋焦慮指數』（可用關鍵字的每日平均），索引為 date。"""
    path = root / "data" / "external" / "trends" / "trends_TW.csv"
    if not path.exists():
        return pd.Series(dtype=float), f"MANUAL_NEEDED:{_rel(root, path)}", []
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return pd.Series(dtype=float), f"UNREADABLE:{_rel(root, path)}({exc})", []
    dcol = _find_col(raw.columns, "date", "日期")
    if dcol is None:
        return pd.Series(dtype=float), f"UNUSABLE:{_rel(root, path)} 缺少日期欄", []
    df = raw.copy()
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    if "geo" in df.columns and (df["geo"].astype(str) == "TW").any():
        df = df[df["geo"].astype(str) == "TW"]
    excluded = {dcol, "date", "geo", "station"}
    keywords = [c for c in df.columns if c not in excluded]
    numeric: dict[str, pd.Series] = {}
    for c in keywords:
        vals = _to_numeric(df[c])
        if vals.notna().any():
            numeric[str(c)] = vals
    if not numeric:
        return pd.Series(dtype=float), f"UNUSABLE:{_rel(root, path)} 沒有數值關鍵字", []
    index = df["date"]
    series = pd.concat(numeric.values(), axis=1).mean(axis=1, skipna=True)
    series.index = index
    series = series.groupby(level=0).mean().rename("search_anxiety")
    return series, "OK", sorted(numeric)


def load_policy(root: Path) -> tuple[pd.DataFrame, str, dict[str, bool]]:
    """交管規則；檔不存在或缺欄時回傳空表，呼叫端直接略過相關特徵。"""
    path = root / "data" / "external" / "policy" / "holiday_controls.csv"
    empty = pd.DataFrame(columns=["event_id", "event", "year", "date", "corridor_id",
                                  "hov_active", "hov_release_min",
                                  "free_toll_active", "free_toll_end_min"])
    if not path.exists():
        return empty, f"MANUAL_NEEDED:{_rel(root, path)}", {}
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return empty, f"UNREADABLE:{_rel(root, path)}({exc})", {}
    ecol = _find_col(raw.columns, "event_id", "event", "事件")
    ycol = _find_col(raw.columns, "year", "年份")
    dcol = _find_col(raw.columns, "date", "日期")
    ccol = _find_col(raw.columns, "corridor_id", "corridor", "走廊")
    cols = {
        "hov_active": _find_col(raw.columns, "hov_active", "hov", "高乘載啟用"),
        "hov_release_min": _find_col(raw.columns, "hov_release_min", "hov_release", "高乘載解除"),
        "free_toll_active": _find_col(raw.columns, "free_toll_active", "free_toll", "免收費啟用"),
        "free_toll_end_min": _find_col(raw.columns, "free_toll_end_min", "free_toll_end", "免收費結束"),
    }
    if not any(cols.values()):
        return empty, f"UNUSABLE:{_rel(root, path)} 缺少 hov_release_min/free_toll_end_min 等欄位", {}
    out = pd.DataFrame(index=raw.index)
    out["event_id"] = raw[ecol].astype(str) if ecol else np.nan
    out["event"] = raw[ecol].astype(str) if ecol else np.nan
    out["year"] = _to_numeric(raw[ycol]) if ycol else np.nan
    out["date"] = pd.to_datetime(raw[dcol], errors="coerce").dt.normalize() if dcol else pd.NaT
    out["corridor_id"] = raw[ccol].astype(str) if ccol else "ALL"
    for name, src in cols.items():
        out[name] = _to_numeric(raw[src]) if src else np.nan
    available = {name: bool(out[name].notna().any()) for name in cols}
    status = "OK" if any(available.values()) else f"UNUSABLE:{_rel(root, path)} 所有欄位皆缺值"
    return out, status, available


# ---------------------------------------------------------------------------
# 3. 連假拓撲（日曆／事件特徵；皆為事前可知資訊）
# ---------------------------------------------------------------------------
def build_events(calendar: pd.DataFrame, calendar_ok: bool, years=YEARS) -> pd.DataFrame:
    rows: list[dict] = []
    cal = calendar.dropna(subset=["date"]) if calendar_ok else pd.DataFrame()
    for ev in HOLIDAY_EVENTS:
        if ev["year"] not in years:
            continue
        start, end = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        total_days = int((end - start).days) + 1
        if not cal.empty:
            sel = cal[(cal["date"] >= start) & (cal["date"] <= end)]
            rest_days = int(sel["is_day_off"].fillna(False).sum())
        else:
            # 日曆不可用時只做確定的週末推算；leave_leverage 特徵仍會被跳過。
            rest_days = sum((start + pd.Timedelta(days=i)).weekday() >= 5
                            for i in range(total_days))
        leave_days = max(total_days - rest_days, 1)
        note = ""
        if not cal.empty:
            near = cal[(cal["date"] >= start - pd.Timedelta(days=21)) &
                       (cal["date"] <= end + pd.Timedelta(days=1))]
            note = " ".join(near["note"].astype(str).tolist())
        makeup = int(("補班" in note) or ("補行上班" in note) or ("調整放假" in note))
        rows.append({
            "event_id": f"{ev['year']}_{ev['event']}",
            "year": ev["year"],
            "event": ev["event"],
            "start": start,
            "end": end,
            # 前後各一天也視為事件窗口（與 src/features.py 一致）
            "window_start": start - pd.Timedelta(days=1),
            "window_end": end + pd.Timedelta(days=1),
            "total_days": total_days,
            "rest_days": rest_days,
            "leave_days": leave_days,
            "makeup_week_flag": makeup,
        })
    return pd.DataFrame(rows)


def _policy_values(policy: pd.DataFrame, panel: pd.DataFrame,
                   value_col: str, active_col: str) -> np.ndarray:
    """依 (日期, 走廊) 精確映射交管分鐘數；無適用規則的列維持 NaN。"""
    empty = np.full(len(panel), np.nan, dtype=float)
    if policy.empty or value_col not in policy.columns:
        return empty
    sub = policy
    if active_col in policy.columns:
        sub = sub[sub[active_col].eq(1)]
    sub = sub[sub[value_col].notna()]
    if sub.empty or "date" not in sub.columns or "corridor_id" not in sub.columns:
        return empty
    keyed = sub.groupby(["date", "corridor_id"], sort=False)[value_col].first()
    row_index = pd.MultiIndex.from_arrays([panel["date"], panel["corridor_id"]])
    return _to_numeric(pd.Series(keyed.reindex(row_index).to_numpy(),
                                 index=panel.index)).to_numpy(dtype=float)


def _shift_one_day(s: pd.Series) -> pd.Series:
    """把 D-1 的觀測對齊到 D：只搬索引，不改值。"""
    out = s.copy()
    out.index = out.index + pd.Timedelta(days=1)
    return out


def _prior_weekend_cumulative(daily: pd.DataFrame, dates: pd.DatetimeIndex,
                              column: str) -> pd.Series:
    """每個日期所屬『最近一個已結束的週末』累積雨量（嚴格早於 D）。"""
    cum: dict[pd.Timestamp, float] = {}
    if column in daily.columns:
        for d in daily.index:
            if d.weekday() not in (5, 6):
                continue
            sat = d if d.weekday() == 5 else d - pd.Timedelta(days=1)
            sun = sat + pd.Timedelta(days=1)
            vals = []
            for x in (sat, sun):
                if x in daily.index and pd.notna(daily.at[x, column]):
                    vals.append(float(daily.at[x, column]))
            value = float(np.sum(vals)) if vals else np.nan
            cum[sat] = value
            cum[sun] = value
    keys = sorted(cum)
    out = pd.Series(np.nan, index=dates, dtype=float)
    for d in dates:
        earlier = [k for k in keys if k < d]
        if earlier:
            out.loc[d] = cum[earlier[-1]]
    return out


def build_static_features(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    calendar_ok: bool,
    weather_daily: dict[str, pd.DataFrame],
    weather_status: dict[str, str],
    trends: pd.Series,
    trend_status: str,
    policy: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """建立所有與當日車流無關的特徵（D-1 前可取得）；回傳 (特徵, 來源狀態)。

    回傳的 DataFrame 與 panel 列數、順序完全一致。
    """
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    daily = pd.DataFrame(index=dates)

    # --- 連假拓撲與日曆 ----------------------------------------------------
    ctx = []
    for d in dates:
        row = {
            "date": d,
            "event_type": "NONE",
            "event_id": "",
            "event_day_index": 0,
            "is_holiday_period": 0,
            "event_total_days": 0,
            "days_to_event_end": np.nan,
            "leave_leverage": np.nan,
            "makeup_week_flag": np.nan,
        }
        for ev in events.itertuples():
            if ev.window_start <= d <= ev.window_end:
                in_core = ev.start <= d <= ev.end
                row.update({
                    "event_type": ev.event,
                    "event_id": ev.event_id,
                    "event_day_index": int((d - ev.start).days) + 1 if in_core else 0,
                    "is_holiday_period": int(in_core),
                    "event_total_days": int(ev.total_days),
                    "days_to_event_end": int((ev.end - d).days) if in_core else np.nan,
                    "leave_leverage": float(ev.total_days) / max(ev.leave_days, 1) if calendar_ok else np.nan,
                    "makeup_week_flag": float(ev.makeup_week_flag) if calendar_ok else np.nan,
                })
                break
        ctx.append(row)
    ctx_df = pd.DataFrame(ctx).set_index("date")
    for col in ctx_df.columns:
        daily[col] = ctx_df[col]

    # --- 氣象：只取 D-1 / 前序週末 ----------------------------------------
    ylp = weather_daily.get("YLP")
    tpe = weather_daily.get("TPE")
    if ylp is not None:
        daily["ylp_precip_d1"] = _shift_one_day(ylp["precip_mean"]).reindex(dates)
        daily["ylp_precip_prev_weekend"] = _prior_weekend_cumulative(ylp, dates, "precip_sum")
    if tpe is not None:
        daily["tpe_precip_d1"] = _shift_one_day(tpe["precip_mean"]).reindex(dates)
    if tpe is not None and ylp is not None:
        daily["precip_scissors_d1"] = (daily["tpe_precip_d1"] - daily["ylp_precip_d1"])
        daily["temp_diff_d1"] = (_shift_one_day(tpe["temp_mean"]).reindex(dates) -
                                 _shift_one_day(ylp["temp_mean"]).reindex(dates))

    # --- Google Trends：D-1 搜尋焦慮指數 ----------------------------------
    if not trends.empty:
        daily["search_anxiety_d1"] = _shift_one_day(trends).reindex(dates)

    # 對齊到 panel 的每一列（日期重複也沒問題）
    aligned = daily.reindex(pd.DatetimeIndex(panel["date"])).reset_index(drop=True)

    # --- 列級時間週期特徵 --------------------------------------------------
    slot = panel["time_slot_15m"].to_numpy(dtype=float)
    minute = slot * SLOT_MINUTES + SLOT_MINUTES / 2.0
    weekday = panel["date"].dt.weekday.to_numpy(dtype=float)
    aligned["slot_sin"] = np.sin(2.0 * np.pi * slot / SLOTS_PER_DAY)
    aligned["slot_cos"] = np.cos(2.0 * np.pi * slot / SLOTS_PER_DAY)
    aligned["hour_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    aligned["hour_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    aligned["weekday"] = weekday
    aligned["is_weekend"] = (weekday >= 5).astype(int)

    # --- 列級交管倒數（依表定規則的日期+走廊，非當日觀測） ------------------
    hov_release = _policy_values(policy, panel, "hov_release_min", "hov_active")
    toll_end = _policy_values(policy, panel, "free_toll_end_min", "free_toll_active")
    aligned["min_to_hov_release"] = hov_release - minute
    aligned["min_to_free_toll_end"] = toll_end - minute

    source_status = {
        "ylp_precip_d1": "OK" if ylp is not None else weather_status.get("YLP", "UNKNOWN"),
        "ylp_precip_prev_weekend": "OK" if ylp is not None else weather_status.get("YLP", "UNKNOWN"),
        "tpe_precip_d1": "OK" if tpe is not None else weather_status.get("TPE", "UNKNOWN"),
        "precip_scissors_d1": "OK" if tpe is not None and ylp is not None else "UNAVAILABLE: 需要 TPE 與 YLP",
        "temp_diff_d1": "OK" if tpe is not None and ylp is not None else "UNAVAILABLE: 需要 TPE 與 YLP",
        "search_anxiety_d1": trend_status,
        "leave_leverage": "OK" if calendar_ok else "UNAVAILABLE: office calendar",
        "makeup_week_flag": "OK" if calendar_ok else "UNAVAILABLE: office calendar",
        "min_to_hov_release": ("OK" if policy["hov_release_min"].notna().any()
                               else "UNAVAILABLE: holiday_controls.csv.hov_release_min"),
        "min_to_free_toll_end": ("OK" if policy["free_toll_end_min"].notna().any()
                                else "UNAVAILABLE: holiday_controls.csv.free_toll_end_min"),
    }
    return aligned, source_status


# ---------------------------------------------------------------------------
# 4. 歷史同期統計：只在 train 擬合；train 自己用 leave-one-out 避免自洩漏
# ---------------------------------------------------------------------------
HISTORY_LEVELS = (
    ("corridor_id", "time_slot_15m", "event_type"),
    ("corridor_id", "time_slot_15m"),
    ("corridor_id",),
)


def historical_features(train: pd.DataFrame, target: pd.DataFrame,
                        columns: dict[str, str], leave_one_out: bool) -> pd.DataFrame:
    """以 train 的 groupby(sum,count) 統計映射到 target。

    columns: {輸出欄名: train 中的來源欄名}。target 為 train 本人時改用
    leave-one-out（扣掉自己那一列），避免訓練特徵偷到自身標的。
    """
    out = pd.DataFrame(index=target.index)
    for name, source in columns.items():
        values = pd.Series(np.nan, index=target.index, dtype=float)
        for keys in HISTORY_LEVELS:
            grouped = train.groupby(list(keys), sort=False, dropna=False)[source].agg(["sum", "count"])
            if len(keys) == 1:
                key_index = pd.Index(target[keys[0]])
            else:
                key_index = pd.MultiIndex.from_frame(target[list(keys)])
            sums = _to_numeric(pd.Series(grouped["sum"].reindex(key_index).to_numpy(), index=target.index)).to_numpy(dtype=float)
            counts = _to_numeric(pd.Series(grouped["count"].reindex(key_index).to_numpy(), index=target.index)).to_numpy(dtype=float)
            if leave_one_out:
                own = _to_numeric(target[source]).to_numpy(dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    candidate = (sums - own) / np.where(counts > 1, counts - 1, np.nan)
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    candidate = sums / np.where(counts > 0, counts, np.nan)
            values = values.fillna(pd.Series(candidate, index=target.index))
        out[name] = values
    return out
