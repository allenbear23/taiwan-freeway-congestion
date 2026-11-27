# -*- coding: utf-8 -*-
"""一次性補抓 1A 用的 Google Trends 關鍵字，縫合進既有 trends_TW.csv。

只新增缺欄（雪隧 / 國道5號 / 天氣；高乘載已存在於真實檔），
绝不覆蓋既有關鍵字欄位，也不製造任何數值：抓取失敗就留空並如實輸出狀態。

尺度說明：既有欄位是 kw / TRENDS_ANCHOR(=天氣) 比值；「天氣」本身即錨點，
因此該欄存原始 interest_over_time 指數（0–100），仍以重疊期之中位數比例縫合。
"""
from __future__ import annotations

import random
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import (  # noqa: E402
    CACHE, EXTERNAL, H1A_TRENDS_KEYWORDS, TRENDS_ANCHOR, TRENDS_GEO,
    TRENDS_HL, TRENDS_MAX_SPAN_DAYS, TRENDS_OVERLAP_DAYS, TRENDS_SLEEP_RANGE,
    TRENDS_TZ,
)
from src.external import _chunks, _read_csv  # noqa: E402

START, END = "2022-12-01", "2025-12-31"
TRENDS_CSV = EXTERNAL / "trends" / f"trends_{TRENDS_GEO}.csv"


def _fetch_chunk(pytrends, kw: str, c_start, c_end) -> pd.DataFrame | None:
    """回傳原始區塊（含日期與需要的欄），失敗回 None，不回假資料。"""
    key = f"{kw}_{TRENDS_GEO}_{c_start:%Y%m%d}_{c_end:%Y%m%d}"
    fp = CACHE / "trends" / f"{key}.csv"
    if fp.exists():
        sub = _read_csv(fp, parse_dates=["date"])
        if not sub.empty:
            return sub
    kws = [kw] if kw == TRENDS_ANCHOR else [kw, TRENDS_ANCHOR]
    try:
        pytrends.build_payload(kws, cat=0,
                               timeframe=f"{c_start:%Y-%m-%d} {c_end:%Y-%m-%d}",
                               geo=TRENDS_GEO, gprop="")
        raw = pytrends.interest_over_time()
        if raw.empty:
            raise RuntimeError("回傳空資料")
        sub = raw.reset_index()[["date", *kws]]
        fp.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"  GET {key} ({len(sub)} 天)")
        time.sleep(random.uniform(*TRENDS_SLEEP_RANGE))
        return sub
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {key}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _series(sub: pd.DataFrame, kw: str) -> pd.Series:
    """轉成單一數列；非錨點關鍵字用 kw/天氣比值，錨點本身用原始指數。"""
    dates = pd.to_datetime(sub["date"], errors="coerce")
    if kw != TRENDS_ANCHOR and TRENDS_ANCHOR in sub.columns:
        anchor = pd.to_numeric(sub[TRENDS_ANCHOR], errors="coerce").replace(0, np.nan)
        vals = pd.to_numeric(sub[kw], errors="coerce") / anchor
    else:
        vals = pd.to_numeric(sub[kw], errors="coerce")
    return pd.Series(vals.to_numpy(), index=dates).dropna()


def _stitch(pieces: list[pd.Series]) -> pd.Series:
    """以重疊 30 天的中位數比例縫合相鄰批次（與 fetch_trends 同邏輯）。"""
    out: list[pd.Series] = []
    prev: pd.Series | None = None
    for s in pieces:
        if prev is not None:
            ov = pd.concat([s.rename("cur"), prev.rename("prev")], axis=1, join="inner").dropna()
            if len(ov) >= TRENDS_OVERLAP_DAYS // 4:
                a, b = ov["prev"].median(), ov["cur"].median()
                if np.isfinite(a) and np.isfinite(b) and b > 0:
                    s = s * (a / b)
        out.append(s)
        prev = s
    all_s = pd.concat(out).sort_index()
    return all_s[~all_s.index.duplicated(keep="last")]


def main() -> int:
    base = _read_csv(TRENDS_CSV, parse_dates=["date"])
    if base.empty:
        print(f"找不到既有真實檔 {TRENDS_CSV}，終止（不建立假資料）。", file=sys.stderr)
        return 2
    missing = [kw for kw in H1A_TRENDS_KEYWORDS if kw not in base.columns]
    if not missing:
        print("四個 1A 關鍵字皆已存在於 trends_TW.csv，無須抓取。")
        return 0

    print(f"既有欄位：{list(base.columns)}")
    print(f"待補關鍵字：{missing}")
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("pytrends 未安裝，放棄抓取（不填假資料）。", file=sys.stderr)
        return 3
    pytrends = TrendReq(hl=TRENDS_HL, tz=TRENDS_TZ, retries=2, backoff_factor=1.5)

    idx = pd.to_datetime(base["date"], errors="coerce")
    added: dict[str, pd.Series] = {}
    failed: list[str] = []
    for kw in missing:
        print(f"[{kw}] 抓取 {START} ~ {END}")
        pieces: list[pd.Series] = []
        consec_fail = 0
        for c_start, c_end in _chunks(pd.Timestamp(START), pd.Timestamp(END)):
            sub = _fetch_chunk(pytrends, kw, c_start, c_end)
            if sub is None or kw not in sub.columns:
                consec_fail += 1
                failed.append(f"{kw} {c_start:%Y%m%d}_{c_end:%Y%m%d}")
                if consec_fail >= 3:
                    break
                continue
            consec_fail = 0
            pieces.append(_series(sub, kw))
        if pieces:
            added[kw] = _stitch(pieces).reindex(idx)

    if not added:
        print("API_ERROR：所有關鍵字皆抓取失敗，trends_TW.csv 未更動（不填假資料）。",
              file=sys.stderr)
        return 4

    out = base.copy()
    for kw, s in added.items():
        out[kw] = s.to_numpy()
    out = out[[*base.columns, *[k for k in added if k not in base.columns]]]
    TRENDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TRENDS_CSV, index=False, encoding="utf-8-sig")

    print(f"已縫合寫入 {TRENDS_CSV}；新增欄位={list(added)}")
    for kw, s in added.items():
        print(f"  {kw}: 非空={int(s.notna().sum())} / {len(s)}，"
              f"非零={int((s.fillna(0) != 0).sum())}，中位數={s.median():.6g}")
    if failed:
        print(f" PARTIAL：失敗區塊 {len(failed)} 個（{', '.join(failed[:5])}…）", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
