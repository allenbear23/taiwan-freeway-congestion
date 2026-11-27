# 台灣國道連假塞車程度與旅行時間區間預測

研究範圍**僅限國道**（TDCS M05A eTag，國5南/北向、國1南向三走廊），不推論省道或市區道路。
包含 **階段一**（24 假設可行性排行榜）與 **階段二**（2025 時序消融建模、五張中文敘事圖、Colab Notebook）。

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
│   └── office_calendar_2024.csv    # 人事行政總處「西元日期 / 星期 / 是否放假 / 備註」
├── airport/taoyuan_daily_pax.csv   # date, departures, arrivals[, hour]
├── hsr/hsr_holiday_capacity.csv    # date, load_factor, extra_train_ratio
├── policy/holiday_controls.csv     # date,event,hov_active,hov_release_min,free_toll_active,
│                                   # free_toll_end_min,ramp_closure_gantry,ramp_closure_km,n3_discount
├── road/t9_hourly.csv              # date, datetime, t9_volume, t9_travel_min
├── events/mega_events.csv          # datetime, km, hours_from_end
└── gantry_km.csv                   # gantry_id, km（選填）
```

氣象資料由 `fetch_weather_once.py` 抓取 [Open-Meteo Archive](https://open-meteo.com/en/docs/historical-weather-api)
（ERA5 再分析，免登入、非假資料），寫入 `data/external/weather/hourly_{TPE,YLP,TAY}.csv`。
若想改用 CODiS 觀測，直接覆蓋同名檔案即可（欄位須為 `datetime, precip_mm, temp_c`）。

Google Trends 若改用 `--fetch-trends`：單次窗口 ≤180 天、相鄰窗口重疊 30 天、
以 `"天氣"` 作錨點取比值後按重疊區中位數縫合尺度，每次請求隨機延遲 8–20 秒，
快取在 `data/cache/trends/`。農民曆由 `lunar-python` 本地生成，不需手動下載。

## 走廊定義

統一在 `config.py` 的 `CORRIDORS` 管理：`N5_SB`、`N5_NB`（國5 南港–頭城）、
`N1_SB`（國1 桃園–新竹）、`N2_WB/N2_EB`（國2 機場支線，需 VD）。探索邏輯沒有寫死走廊。

## 階段二：建模與圖表

```powershell
$env:PYTHONIOENCODING='utf-8'
python generate_policy_rules.py     # 規則化重建交管表（非官方逐日公告）
python fetch_trends_h1a.py          # 補抓 1A 關鍵字（真實抓取，失敗標缺席）
python eda_screener.py --years 2023 2024 2025 --skip-download
python phase2_model.py              # A/B/C 消融；2023–2024 train / 2025 test
python src/final_figures.py         # outputs/figures/ 五張 dpi=300 中文圖
```

`holiday_traffic_final.ipynb` 是 Colab 單一入口（Run All）：字型 → 建模 → 繪圖 → 顯示報告。
產出在 `outputs/phase2/`（指標、特徵、混淆矩陣、2025 預測）與 `outputs/figures/`。
最終結果：模型 C 的 Macro F1 = 0.5283、嚴重壅塞 Recall = 0.5429、P50 MAE = 4.25 分鐘；
相較 Baseline A 改善 30.3%–72.0%，相較傳統 B 改善 8.2%–19.5%。1A 為反假設驗證結果（不顯著）。

## 資料連結

已實測的下載連結與 `MANUAL_NEEDED` 清單見 `outputs/DATA_LINKS.md`。

## 免責

`[MOCK_TEST_ONLY]` 只出現在 `tests/test_clean.py`，僅驗證邏輯；排行榜所有數字皆來自真實資料或缺席標記。
交管規則表為「依規則重建」索引，非高公局官方逐日公告；2C/2D 仍維持資料不足。
