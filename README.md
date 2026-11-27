# 台灣國道連假塞車機率與程度預測｜Step 1 可行性探勘

本目錄只包含 **階段一** 的可執行程式。階段二（Colab 建模）等你確認 Top 3 主軸後再進行。

## 快速開始

```powershell
$env:PYTHONIOENCODING='utf-8'
python eda_screener.py --years 2023 2024 2025

# 只用本機已下載檔
python eda_screener.py --skip-download --years 2024

# 嘗試抓 Google Trends（可能遭封鎖；失敗會如實標記）
python eda_screener.py --fetch-trends --years 2024 2025

# [MOCK_TEST_ONLY] 邏輯單元測試
python eda_screener.py --selftest
```

產出：
- `outputs/feasibility_report.md`：24 假設檢定總表、排行榜、黃金三角、資料缺口
- `outputs/eda_candidates/*.png`：各可用假設的探索圖
- `data/processed/corridor_15m.parquet`、`section_15m.parquet`、`section_5m.parquet`

## 已實測的站台行為（重要）

1. TISVCLOUD 僅保留最近約 35 天的 `YYYYMMDD/HH/` 目錄；更早日期是 root 底的
   `M05A_YYYYMMDD.tar.gz` / `M06A_YYYYMMDD.tar.gz`。下載器具備兩種路徑的自動切換。
2. `TDCS_M05A_YYYYMMDD_HHMMSS.csv` **沒有表頭**，欄序為
   `TimeInterval, GantryFrom, GantryTo, VehicleType, SpaceMeanSpeed, TrafficVolume`。
3. `M05A/M06A` 只有國道 1、3、5 的 eTag 門架；**國道 2 沒有資料**。
   因此 `N2_WB/N2_EB` 走廊保留定義但在排行榜中會標為「資料不足」，4B 需另接 VD。
4. 門架 ID 的數字碼 / 10 ≈ 里程 km（例：`01F0492S` ≈ 國1 49.2 km、`05F0287S` ≈ 國5 28.7 km）。
   程式以此做區間里程加權；若有官方對照表可放 `data/external/gantry_km.csv` 覆蓋。
5. 國5 的相鄰門架區間可長達 20+ km（雪隧），因此里程差上限設 30 km；若設 5 km 會誤殺
   `05F0055S→05F0287S` 等主要區間，讓走廊時速被高估。
6. `0/0` 資料在夜間佔比很高。全日零流量的門架區間視為「未啟用」不納入加權，
   避免自由流補值灌入人造順暢；此筆數亦寫入清洗稽核。

## 清洗鐵則實作位置

`src/clean.py`：
- `TrafficVolume == 0 且 SpaceMeanSpeed == 0` → 先設 NaN，不做紫爆判定；
- 依「上游門架 + 下游門架 + 車種」分組，線性插值最多 2 個 5 分鐘時階，不足則補 90 km/h；
- 逐項記錄於 `CleaningAudit`（原始列數、0/0 筆數、插值筆數、自由流補值筆數、丟棄筆數）；
- 小客車 31 算主車速，42/5 只算重車混流佔比；
- 5 分鐘 → 15 分鐘（每日 96 時段），走廊車速以門架區間里程加權，旅行時間為各區間加總。

## 外部資料手動補檔目錄

若網路/API 不可用，請依下列檔名放置真實資料；程式**不會**生成假資料：

```
data/external/
├── weather/
│   ├── hourly_TPE.csv        # datetime, precip_mm, temp_c
│   ├── hourly_YLP.csv
│   └── hourly_TAY.csv
├── trends/
│   └── trends_TW.csv         # date, 國5塞車, 高乘載, 替代道路, 台9線, 宜蘭住宿, 景點住宿, geo
├── calendar/
│   └── office_calendar_2024.xlsx   # 人事行政總處「日期 / 休假日別 / 備註」
├── airport/taoyuan_daily_pax.csv   # date, departures, arrivals[, hour]
├── hsr/hsr_holiday_capacity.csv    # date, load_factor, extra_train_ratio
├── policy/holiday_controls.csv     # date,event,hov_active,hov_release_min,free_toll_active,
│                                   # free_toll_end_min,ramp_closure_gantry,ramp_closure_km,n3_discount
├── road/t9_hourly.csv              # date, datetime, t9_volume, t9_travel_min
├── events/mega_events.csv          # datetime, km, hours_from_end
└── gantry_km.csv                   # gantry_id, km（選填）
```

Google Trends 若改用 `--fetch-trends`：單次窗口 ≤180 天、相鄰窗口重疊 30 天、
以 `"天氣"` 作錨點取比值後按重疊區中位數縫合尺度，每次請求隨機延遲 8–20 秒，
快取在 `data/cache/trends/`。農民曆由 `lunar-python` 本地生成，不需手動下載。

## 走廊定義

統一在 `config.py` 的 `CORRIDORS` 管理：`N5_SB`、`N5_NB`（國5 南港–頭城）、
`N1_SB`（國1 桃園–新竹）、`N2_WB/N2_EB`（國2 機場支線，需 VD）。探索邏輯沒有寫死走廊。

## 免責

`[MOCK_TEST_ONLY]` 只出現在 `tests/test_clean.py`，僅驗證邏輯；排行榜所有數字皆來自真實資料或缺席標記。
