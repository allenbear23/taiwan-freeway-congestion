# -*- coding: utf-8 -*-
"""規則化重建連假交管索引表（data/external/policy/holiday_controls.csv）。

重要聲明：本檔**不是**交通部高公局逐日官方公告的抄錄，而是依下列固定規則
從 config.HOLIDAY_EVENTS 推導的「規則化重建」索引，供 2A/2B 的邊界效應
探索性檢定使用。不得將其當作官方施政事實引用；若要正式研究，請以高公局
當年度連假交管新聞稿/公告逐日核對後覆寫本表。

規則（依使用者指定）：
  1. 國5 南向 N5_SB：連假前兩日（D-2、D-1）06:00–12:00 高乘載
     → hov_active=1, hov_start_min=360, hov_release_min=720。
  2. 國5 北向 N5_NB：收假前一日與收假日（D_end-1、D_end）13:00–18:00 高乘載
     → hov_active=1, hov_start_min=780, hov_release_min=1080。
  3. 連假期間每日（D0..D_end）00:00–05:00 免收費
     → free_toll_active=1, free_toll_end_min=300（全線適用到現有走廊）。

另產生 date=連假首日的事件索引列（rule=event_index），把該事件在南向/北向
的規則化時點摘要放在同一列，讓以 event.start 查表的特徵（2A/2B）可取用。
event_index 列的 hov_active 表示「本事件有此規則」，不代表 D0 當日實施高乘載；
實際实施日請看 rule=*_hov 的明細列。

未指定規則的項目（匝道封閉、國3 差別費率）一律留空，绝不補值。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import EXTERNAL, HOLIDAY_EVENTS  # noqa: E402

YEARS = (2023, 2024, 2025)
OUTPUT = EXTERNAL / "policy" / "holiday_controls.csv"
FREE_TOLL_CORRIDORS = ("N5_SB", "N5_NB", "N1_SB")

COLUMNS = [
    "date", "event", "year", "corridor_id", "rule",
    "hov_active", "hov_start_min", "hov_release_min",
    "free_toll_active", "free_toll_end_min",
    "ramp_closure_gantry", "n3_discount",
]


def _row(date: pd.Timestamp, event: dict, corridor: str, rule: str, **kw) -> dict:
    """所有欄位先補空，再由規則填入；空值代表「無此規則/未指定」。"""
    rec = {
        "date": date, "event": event["event"], "year": event["year"],
        "corridor_id": corridor, "rule": rule,
        "hov_active": np.nan, "hov_start_min": np.nan, "hov_release_min": np.nan,
        "free_toll_active": np.nan, "free_toll_end_min": np.nan,
        "ramp_closure_gantry": np.nan, "n3_discount": np.nan,
    }
    rec.update(kw)
    return rec


def build() -> pd.DataFrame:
    rows: list[dict] = []
    for ev in HOLIDAY_EVENTS:
        if ev["year"] not in YEARS:
            continue
        s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])

        # --- 事件索引列：以 D0 為鍵，供 _policy_of(event_id, corridor_id) 查表 ---
        rows.append(_row(s, ev, "N5_SB", "event_index",
                         hov_active=1, hov_start_min=360, hov_release_min=720,
                         free_toll_active=1, free_toll_end_min=300))
        rows.append(_row(s, ev, "N5_NB", "event_index",
                         hov_active=1, hov_start_min=780, hov_release_min=1080,
                         free_toll_active=1, free_toll_end_min=300))

        # --- 規則 1：國5 南向，連假前兩日 06:00–12:00 高乘載 ---
        for back in (2, 1):
            rows.append(_row(s - pd.Timedelta(days=back), ev, "N5_SB",
                             "pre_holiday_hov_N5_SB",
                             hov_active=1, hov_start_min=360, hov_release_min=720))

        # --- 規則 2：國5 北向，收假前一日與收假日 13:00–18:00 高乘載 ---
        for back in (1, 0):
            rows.append(_row(e - pd.Timedelta(days=back), ev, "N5_NB",
                             "return_hov_N5_NB",
                             hov_active=1, hov_start_min=780, hov_release_min=1080))

        # --- 規則 3：連假期間每日 00:00–05:00 免收費 ---
        day = s
        while day <= e:
            for cid in FREE_TOLL_CORRIDORS:
                rows.append(_row(day, ev, cid, "holiday_free_toll",
                                 free_toll_active=1, free_toll_end_min=300))
            day += pd.Timedelta(days=1)

    return pd.DataFrame(rows, columns=COLUMNS).sort_values(
        ["date", "corridor_id", "rule"], kind="stable"
    ).reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"規則化重建（非官方逐日公告）已寫入：{OUTPUT}")
    print(f"列數={len(df)}，連假事件數={df.groupby(['year', 'event']).ngroups}，"
          f"年份={sorted(df['year'].unique().tolist())}")
    print(df.head(12).to_string(index=False))
