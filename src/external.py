# -*- coding: utf-8 -*-
"""外部跨領域資料載入器。

每一個 loader 回傳 (DataFrame, 狀態字串)。狀態：
  OK          真實資料已取得
  MANUAL_NEEDED 需手動下載放到指定路徑（程式不生成假資料）
  API_ERROR   遠端 API 失敗
  DISABLED    未啟用（例如 pytrends 需 --fetch-trends）
"""
from __future__ import annotations

import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CACHE, EXTERNAL, EXTERNAL_FILES, TRENDS_ANCHOR, TRENDS_GEO, TRENDS_GEOS,
    TRENDS_HL, TRENDS_MAX_SPAN_DAYS, TRENDS_OVERLAP_DAYS, TRENDS_SLEEP_RANGE,
    TRENDS_TZ, WEATHER_STATIONS,
)
from src.utils import LOGGER


def _read_csv(path: Path, **kw) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("讀取 %s 失敗：%s", path, e)
        return pd.DataFrame()


def _manual(key: str) -> str:
    return f"MANUAL_NEEDED: data/external/{EXTERNAL_FILES[key]}"


# ---------------------------------------------------------------------------
# 氣象：優先讀取手動下載的 CODiS 逐時觀測；若有 CWA_API_KEY 則嘗試 API
# ---------------------------------------------------------------------------
def load_weather(fetch_api: bool = False) -> tuple[pd.DataFrame, str]:
    frames = []
    for st in WEATHER_STATIONS:
        p = EXTERNAL / EXTERNAL_FILES["weather"].format(station=st)
        df = _read_csv(p)
        if df.empty:
            continue
        rename = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=rename)
        tcol = next((c for c in df.columns if c in ("datetime", "time", "date", "obs_time")), None)
        rcol = next((c for c in df.columns if "rain" in c or "precip" in c or "降水" in c), None)
        tcol2 = next((c for c in df.columns if "temp" in c or "氣溫" in c), None)
        if tcol is None:
            continue
        out = pd.DataFrame({
            "datetime": pd.to_datetime(df[tcol], errors="coerce"),
            "station": st,
            "precip_mm": pd.to_numeric(df[rcol], errors="coerce") if rcol else np.nan,
            "temp_c": pd.to_numeric(df[tcol2], errors="coerce") if tcol2 else np.nan,
        }).dropna(subset=["datetime"])
        frames.append(out)
    if frames:
        return pd.concat(frames, ignore_index=True), "OK"

    if fetch_api and os.getenv("CWA_API_KEY"):
        return _fetch_cwa_api()
    return pd.DataFrame(columns=["datetime", "station", "precip_mm", "temp_c"]), _manual("weather")


def _fetch_cwa_api() -> tuple[pd.DataFrame, str]:
    """CWA 開放資料 API（需環境變數 CWA_API_KEY）；只取自動站觀測。"""
    import requests
    key = os.getenv("CWA_API_KEY")
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"
    rows = []
    for st in WEATHER_STATIONS:
        try:
            r = requests.get(url, params={
                "Authorization": key, "format": "JSON", "stationId": st,
            }, timeout=60)
            r.raise_for_status()
            for item in r.json()["records"]["Station"]:
                rows.append({
                    "datetime": pd.to_datetime(item.get("ObsTime", {}).get("DateTime")),
                    "station": st,
                    "temp_c": pd.to_numeric(item.get("AirTemperature"), errors="coerce"),
                    "precip_mm": pd.to_numeric(item.get("Precipitation"), errors="coerce"),
                })
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("CWA API %s 失敗：%s", st, e)
    if not rows:
        return pd.DataFrame(columns=["datetime", "station", "precip_mm", "temp_c"]), "API_ERROR: CWA"
    return pd.DataFrame(rows).dropna(subset=["datetime"]), "OK"


# ---------------------------------------------------------------------------
# Google Trends：≤180 天 / 30 天重疊 / 錨點「天氣」尺度縫合
# ---------------------------------------------------------------------------
def _chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    cur = start
    while cur < end:
        stop = min(cur + timedelta(days=TRENDS_MAX_SPAN_DAYS - 1), end)
        out.append((cur, stop))
        if stop >= end:
            break
        cur = stop - timedelta(days=TRENDS_OVERLAP_DAYS - 1)
    return out


def fetch_trends(keywords: tuple[str, ...], geo: str,
                 start: str, end: str) -> tuple[pd.DataFrame, str]:
    """暴力但合規的抓取；快取到 CACHE/trends 後縫合。"""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return pd.DataFrame(), "API_ERROR: pytrends 未安裝"

    pytrends = TrendReq(hl=TRENDS_HL, tz=TRENDS_TZ, retries=2, backoff_factor=1.5)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    all_parts: list[pd.DataFrame] = []
    prev: pd.DataFrame | None = None
    for kw in keywords:
        pieces = []
        prev = None  # 尺度縫合只能在「同一關鍵字」的相鄰批次間進行
        for c_start, c_end in _chunks(s, e):
            key = f"{kw}_{geo}_{c_start:%Y%m%d}_{c_end:%Y%m%d}"
            fp = CACHE / "trends" / f"{key}.csv"
            fp.parent.mkdir(parents=True, exist_ok=True)
            if fp.exists():
                sub = _read_csv(fp, parse_dates=["date"])
            else:
                try:
                    pytrends.build_payload([kw, TRENDS_ANCHOR], cat=0,
                                           timeframe=f"{c_start:%Y-%m-%d} {c_end:%Y-%m-%d}",
                                           geo=geo, gprop="")
                    raw = pytrends.interest_over_time()
                    if raw.empty:
                        raise RuntimeError("回傳空資料")
                    sub = raw.reset_index().rename(columns={"date": "date"})
                    sub = sub[["date", kw, TRENDS_ANCHOR]]
                    sub.to_csv(fp, index=False, encoding="utf-8-sig")
                    LOGGER.info("Trends 取得 %s (%d 天)", key, len(sub))
                except Exception as ex:  # noqa: BLE001
                    LOGGER.warning("Trends 失敗 %s：%s", key, ex)
                    return pd.DataFrame(), f"API_ERROR: Google Trends ({type(ex).__name__})"
                time.sleep(random.uniform(*TRENDS_SLEEP_RANGE))
            sub["ratio"] = pd.to_numeric(sub[kw], errors="coerce") / pd.to_numeric(
                sub[TRENDS_ANCHOR], errors="coerce").replace(0, np.nan)
            sub = sub[["date", "ratio"]].rename(columns={"ratio": kw})
            if prev is not None:
                overlap = pd.merge(sub, prev, on="date", suffixes=("_cur", "_prev"))
                if len(overlap) >= 7:
                    a = overlap[f"{kw}_prev"].replace(0, np.nan).median()
                    b = overlap[f"{kw}_cur"].replace(0, np.nan).median()
                    if np.isfinite(a) and np.isfinite(b) and b > 0:
                        sub[kw] = sub[kw] * (a / b)
            pieces.append(sub)
            prev = sub
        if pieces:
            merged = pd.concat(pieces).drop_duplicates("date", keep="last").sort_values("date")
            all_parts.append(merged.set_index("date"))
    if not all_parts:
        return pd.DataFrame(), "API_ERROR: Trends 無資料"
    out = pd.concat(all_parts, axis=1).reset_index()
    out["geo"] = geo
    # 縫合結果落檔，後續 --skip-download 可直接讀取真實快取
    merged_path = EXTERNAL / "trends" / f"trends_{geo}.csv"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(merged_path, index=False, encoding="utf-8-sig")
    return out, "OK"


def load_trends(fetch: bool = False, keywords: tuple[str, ...] = (),
                start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, str]:
    """讀取已縫合的 trends CSV；fetch=True 時先嘗試抓取。"""
    frames = []
    for geocode in TRENDS_GEOS:
        p = EXTERNAL / "trends" / f"trends_{geocode}.csv"
        df = _read_csv(p, parse_dates=["date"])
        if not df.empty:
            df["geo"] = geocode
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True), "OK"
    if fetch:
        kws = keywords or ("國5塞車", "高乘載", "替代道路", "台9線", "宜蘭住宿")
        return fetch_trends(kws, TRENDS_GEO, start or "2023-01-01", end or "2025-12-31")
    return pd.DataFrame(), "DISABLED: 加 --fetch-trends 才連 Google Trends"


# ---------------------------------------------------------------------------
# 人事行政總處辦公日曆
# ---------------------------------------------------------------------------
def load_calendar(years: list[int]) -> tuple[pd.DataFrame, str]:
    frames = []
    for y in years:
        csv_p = EXTERNAL / f"calendar/office_calendar_{y}.csv"
        if csv_p.exists():  # data.gov.tw 資料集 14718：西元日期,星期,是否放假,備註
            try:
                df = pd.read_csv(csv_p, encoding="utf-8-sig")
                dcol = next((c for c in df.columns if "日期" in str(c)), None)
                if dcol is not None:
                    out = pd.DataFrame({
                        "date": pd.to_datetime(df[dcol].astype(str), format="%Y%m%d", errors="coerce"),
                    })
                    fcol = next((c for c in df.columns if "放假" in str(c)), None)
                    out["day_type"] = (df[fcol].map({0: "上班", 2: "放假"}).fillna("")
                                       if fcol is not None else "")
                    ncol = next((c for c in df.columns if "備註" in str(c)), None)
                    out["note"] = df[ncol].fillna("").astype(str) if ncol is not None else ""
                    frames.append(out.dropna(subset=["date"]))
                    continue
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("讀取日曆 CSV %s 失敗：%s", csv_p, e)
        p = EXTERNAL / EXTERNAL_FILES["calendar"].format(year=y)
        if not p.exists():
            continue
        try:
            raw = pd.read_excel(p, header=None)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("讀取日曆 %s 失敗：%s", p, e)
            continue
        # 自動尋找標題列（含「日期」與「休假日別」）
        head_idx = None
        for i in range(min(15, len(raw))):
            if raw.iloc[i].astype(str).str.contains("日期").any():
                head_idx = i
                break
        if head_idx is None:
            continue
        df = pd.read_excel(p, header=head_idx)
        dcol = next((c for c in df.columns if "日期" in str(c)), None)
        if dcol is None:
            continue
        out = pd.DataFrame({"date": pd.to_datetime(df[dcol], errors="coerce")})
        for c in df.columns:
            cs = str(c)
            if "休假日別" in cs:
                out["day_type"] = df[c].astype(str)
            if "補班" in cs or ("備註" in cs):
                out["note"] = df[c].astype(str)
        out = out.dropna(subset=["date"])
        if "day_type" not in out:
            out["day_type"] = ""
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["date", "day_type"]), _manual("calendar")
    return pd.concat(frames, ignore_index=True), "OK"


# ---------------------------------------------------------------------------
# 桃機、高鐵、交管公告、省道、大型活動
# ---------------------------------------------------------------------------
def _simple_csv(key: str, parse_dates: list[str]) -> tuple[pd.DataFrame, str]:
    p = EXTERNAL / EXTERNAL_FILES[key]
    df = _read_csv(p, parse_dates=parse_dates)
    if df.empty:
        return df, _manual(key)
    return df, "OK"


def load_airport() -> tuple[pd.DataFrame, str]:
    return _simple_csv("airport", ["date"])


def load_hsr() -> tuple[pd.DataFrame, str]:
    return _simple_csv("hsr", ["date"])


def load_policy() -> tuple[pd.DataFrame, str]:
    return _simple_csv("policy", ["date"])


def load_provincial() -> tuple[pd.DataFrame, str]:
    return _simple_csv("provincial", ["datetime"])


def load_mega_events() -> tuple[pd.DataFrame, str]:
    return _simple_csv("mega_events", ["datetime"])


# ---------------------------------------------------------------------------
# 農民曆（本地生成，真實演算，非假資料）
# ---------------------------------------------------------------------------
def load_lunar(dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, str]:
    rows = []
    try:
        from lunar_python import Lunar, Solar
        for d in pd.to_datetime(pd.Series(dates)).dt.normalize().unique():
            s = Solar.fromYmd(int(d.year), int(d.month), int(d.day))
            l = Lunar.fromSolar(s)
            yi = [str(x) for x in l.getDayYi()]
            ji = [str(x) for x in l.getDayJi()]
            rows.append({
                "date": pd.Timestamp(d),
                "lunar_date": l.getDayInChinese(),
                "yi_marriage": any("嫁娶" in x for x in yi),
                "yi_ritual": any("祭祀" in x for x in yi),
                "yi_travel": any("出行" in x for x in yi),
                "lunar_yi": "、".join(yi),
            })
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("農民曆生成失敗：%s", e)
        return pd.DataFrame(), f"API_ERROR: lunar-python ({e})"
    return pd.DataFrame(rows), "OK"


def collect_all(years: list[int], dates: list[pd.Timestamp],
                fetch_trends_flag: bool = False, trend_start: str | None = None,
                trend_end: str | None = None) -> dict[str, tuple[pd.DataFrame, str]]:
    """一次載入全部外部來源，回傳 {name: (df, status)}。"""
    out: dict[str, tuple[pd.DataFrame, str]] = {}
    out["weather"] = load_weather(fetch_api=bool(os.getenv("CWA_API_KEY")))
    out["trends"] = load_trends(fetch=fetch_trends_flag, start=trend_start, end=trend_end)
    out["calendar"] = load_calendar(years)
    out["airport"] = load_airport()
    out["hsr"] = load_hsr()
    out["policy"] = load_policy()
    out["provincial"] = load_provincial()
    out["mega_events"] = load_mega_events()
    out["lunar"] = load_lunar(dates)
    for k, (df, st) in out.items():
        LOGGER.info("外部資料 %-12s %-60s rows=%d", k, st, len(df))
    return out
