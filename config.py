# -*- coding: utf-8 -*-
"""全專案單一設定來源。

設計原則：走廊、連假事件、清洗閾值、外部資料路徑一律集中在這裡，
探索邏輯不得再寫死任何走廊代碼或日期。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
CACHE = DATA / "cache"
OUTPUTS = ROOT / "outputs"
EDA_FIGS = OUTPUTS / "eda_candidates"
REPORT_PATH = OUTPUTS / "feasibility_report.md"

for _p in (RAW, PROCESSED, EXTERNAL, CACHE, EDA_FIGS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 走廊定義（唯一來源）
#   gantry_id 格式：RR F NNNN D / RR H NNNN D
#   RR=道路代碼(01/03/05...)，F=主線門架，NNNN/10 ≈ 里程(km)，D=方向(N/S/E/W)
#   km 可由 ID 推得；若有官方對照表可放 data/external/gantry_km.csv 覆蓋。
# ----------------------------------------------------------------------------
CORRIDORS: dict[str, dict] = {
    "N5_SB": {
        "name": "國5 南港系統→頭城（南向）",
        "road": "05", "direction": "S", "km_min": 0.0, "km_max": 31.0,
        "gantry_kinds": ("F",), "axis": "time",
    },
    "N5_NB": {
        "name": "國5 頭城→南港系統（北向）",
        "road": "05", "direction": "N", "km_min": 0.0, "km_max": 31.0,
        "gantry_kinds": ("F",), "axis": "time",
    },
    "N1_SB": {
        "name": "國1 桃園→新竹（南向）",
        "road": "01", "direction": "S", "km_min": 45.0, "km_max": 100.0,
        "gantry_kinds": ("F",), "axis": "time",
    },
    # 注意：TDCS M05A/M06A 僅含國道 1/3/5 之 eTag 門架，國2 無 M05A 資料。
    # 此走廊保留定義，但需要另外的 VD 資料才會啟用（詳見 README 手動補檔）。
    "N2_WB": {
        "name": "國2 機場支線（西向）",
        "road": "02", "direction": "W", "km_min": 0.0, "km_max": 20.0,
        "gantry_kinds": ("F", "H"), "axis": "space",
    },
    "N2_EB": {
        "name": "國2 機場支線（東向）",
        "road": "02", "direction": "E", "km_min": 0.0, "km_max": 20.0,
        "gantry_kinds": ("F", "H"), "axis": "space",
    },
}

# 國3 差別費率對照用（2D）
N3_PARALLEL = {
    "road": "03", "direction": "S", "km_min": 0.0, "km_max": 150.0,
}

# ----------------------------------------------------------------------------
# 資料清洗鐵則
# ----------------------------------------------------------------------------
PASSENGER_TYPE = 31
HEAVY_TYPES = (42, 5)          # 僅用於混流佔比，不參與主車速
OTHER_TYPES = (32, 41)
FREE_FLOW_SPEED = 90.0         # km/h，補值用
ZERO_ZERO_AS_MISSING = True
INTERPOLATION_LIMIT = 2        # 最多補 2 個 5 分鐘時階（10 分鐘）
SLOT_MINUTES = 15
SLOTS_PER_DAY = 96
CONGESTION_LEVELS = [(80, 0), (60, 1), (40, 2), (0, 3)]  # >=80 / >=60 / >=40 / <40

TISV_BASE = "https://tisvcloud.freeway.gov.tw/history/TDCS"
RECENT_DAYS = 35               # 站台保留最近 35 天的 YYYYMMDD/HH/ 目錄，其餘為 tar.gz
REQUEST_TIMEOUT = 60
DOWNLOAD_RETRIES = 3

# Google Trends 無法避免的封鎖風險之參數（嚴格遵守）
TRENDS_HL = "zh-TW"
TRENDS_TZ = -480
TRENDS_GEO = "TW"
TRENDS_MAX_SPAN_DAYS = 180
TRENDS_OVERLAP_DAYS = 30
TRENDS_ANCHOR = "天氣"
TRENDS_GEOS = ("TW", "TW-TPE", "TW-NWT")
TRENDS_SLEEP_RANGE = (8, 20)  # 隨機延遲秒數

# ----------------------------------------------------------------------------
# 連假事件（人工整理的日期區間，含前後各 1 日對照）
#   start/end 為「連假本體」，window_start/window_end 為實際下載區間（含前後各 1 日）
#   若 data/external/calendar/ 提供人事行政總處辦公日曆，features 模組會再校正。
# ----------------------------------------------------------------------------
HOLIDAY_EVENTS: list[dict] = [
    # 2023
    dict(event="春節", year=2023, start="2023-01-20", end="2023-01-29"),
    dict(event="228", year=2023, start="2023-02-25", end="2023-02-28"),
    dict(event="清明", year=2023, start="2023-04-01", end="2023-04-05"),
    dict(event="端午", year=2023, start="2023-06-22", end="2023-06-25"),
    dict(event="中秋", year=2023, start="2023-09-29", end="2023-10-01"),
    dict(event="國慶", year=2023, start="2023-10-07", end="2023-10-10"),
    # 2024
    dict(event="春節", year=2024, start="2024-02-08", end="2024-02-14"),
    dict(event="228", year=2024, start="2024-02-28", end="2024-03-03"),
    dict(event="清明", year=2024, start="2024-04-04", end="2024-04-07"),
    dict(event="端午", year=2024, start="2024-06-08", end="2024-06-10"),
    dict(event="中秋", year=2024, start="2024-09-15", end="2024-09-17"),
    dict(event="國慶", year=2024, start="2024-10-10", end="2024-10-13"),
    # 2025
    dict(event="春節", year=2025, start="2025-01-25", end="2025-02-02"),
    dict(event="228", year=2025, start="2025-02-28", end="2025-03-02"),
    dict(event="清明", year=2025, start="2025-04-03", end="2025-04-06"),
    dict(event="端午", year=2025, start="2025-05-30", end="2025-06-01"),
    dict(event="中秋", year=2025, start="2025-10-04", end="2025-10-06"),
    dict(event="國慶", year=2025, start="2025-10-10", end="2025-10-12"),
    # 2026（保留給 Step 2 時序測試；日期為依政府公告日曆暫定，使用前請以 calendar 檔校正）
    dict(event="春節", year=2026, start="2026-02-14", end="2026-02-22", tentative=True),
    dict(event="228", year=2026, start="2026-02-27", end="2026-03-01", tentative=True),
    dict(event="清明", year=2026, start="2026-04-03", end="2026-04-06", tentative=True),
    dict(event="端午", year=2026, start="2026-06-19", end="2026-06-21", tentative=True),
    dict(event="中秋", year=2026, start="2026-09-25", end="2026-09-27", tentative=True),
    dict(event="國慶", year=2026, start="2026-10-09", end="2026-10-11", tentative=True),
]

# 一般週末對照組：每年取 5 月與 11 月各一個週末
WEEKEND_CONTROL_MONTHS = (5, 11)

# ----------------------------------------------------------------------------
# 外部資料檔名規則（手動下載時放在 data/external/ 對應子目錄）
# ----------------------------------------------------------------------------
EXTERNAL_FILES = {
    "weather": "weather/hourly_{station}.csv",
    "trends": "trends/{keyword}_{geo}_{start}_{end}.csv",
    "calendar": "calendar/office_calendar_{year}.xlsx",
    "airport": "airport/taoyuan_daily_pax.csv",
    "hsr": "hsr/hsr_holiday_capacity.csv",
    "policy": "policy/holiday_controls.csv",
    "lunar": None,  # 由 lunar-python 本地生成，不需手動檔
    "provincial": "road/t9_hourly.csv",
    "mega_events": "events/mega_events.csv",
}

WEATHER_STATIONS = ("TPE", "YLP", "TAY")  # 台北、宜蘭、桃園（CODiS 站碼需依 Actual 欄位調整）
TRENDS_KEYWORDS = (
    "國5塞車", "高乘載", "替代道路", "台9線", "宜蘭住宿", "景點住宿",
)

# ----------------------------------------------------------------------------
# 隨機種子與模型參數
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
N_MPL_BOOT = 5
PERM_REPEATS = 3
SHAP_SAMPLE_N = 200
