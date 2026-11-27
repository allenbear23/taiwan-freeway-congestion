# 階段二：2025 時序消融建模報告

產生時間：2026-11-27T15:57:21　耗時：26.0 秒

## 1. 時序切分與資料規模

| 項目 | 內容 |
|---|---|
| Train | 2023-2024（112 天 / 32256 列） |
| Test | 2025（53 天 / 15264 列） |
| 隨機切分 | 未使用；依日期年份嚴格切分，train/test 日期重疊：False |
| 面板 | 47520 列 / 165 天 / 3 走廊 |
| 標的一致性 | congestion_level 與 speed 推導不一致列數：0 |
| 分類/回歸標的 | congestion_level (0–3) / travel_time_min |

## 2. 三消融模型的 2025 test 指標

| 指標 | A Baseline | B 傳統 | C 增強 |
|---|---:|---:|---:|
| Macro F1 | 0.4054 | 0.4885 | 0.5283 |
| 嚴重壅塞 (L2+L3) Recall | 0.3157 | 0.4544 | 0.5429 |
| Multi-class Log Loss | 0.7536 | 0.6251 | 0.5397 |
| Brier Score (one-hot, 四類加總) | 0.4168 | 0.3410 | 0.2913 |
| 旅行時間 P50 MAE (分鐘) | 5.8085 | 4.7593 | 4.2453 |
| P90 覆蓋率 | 0.8898 | 0.8792 | 0.8905 |

## 3. C 相對 A / B 的改善百分比

| 指標 | C vs A | C vs B |
|---|---:|---:|
| Macro F1 | +30.31% | +8.15% |
| 嚴重壅塞 (L2+L3) Recall | +71.97% | +19.48% |
| Multi-class Log Loss | +28.39% | +13.66% |
| Brier Score (one-hot, 四類加總) | +30.11% | +14.59% |
| 旅行時間 P50 MAE (分鐘) | +26.91% | +10.80% |
| P90 覆蓋率 | +0.08% | +1.29% |

P90 覆蓋率絕對變化：C vs A +0.07 個百分點；C vs B +1.13 個百分點（理想值約 90%）。
改善定義：F1/Recall/覆蓋率越高越好，取 (C−base)/|base|；Log Loss/Brier/MAE 越低越好，取 (base−C)/|base|。

## 4. 特徵組合

| 模型 | 特徵數 | 特徵 |
|---|---:|---|
| A | 3 | `hist_mean_speed, hist_mean_level, hist_severe_rate` |
| B | 12 | `hist_mean_speed, hist_mean_level, hist_severe_rate, slot_sin, slot_cos, hour_sin, hour_cos, weekday, is_weekend, event_day_index, is_holiday_period, ylp_precip_d1` |
| C | 24 | `hist_mean_speed, hist_mean_level, hist_severe_rate, slot_sin, slot_cos, hour_sin, hour_cos, weekday, is_weekend, event_day_index, is_holiday_period, ylp_precip_d1, leave_leverage, makeup_week_flag, event_total_days, days_to_event_end, min_to_hov_release, min_to_free_toll_end, tpe_precip_d1, precip_scissors_d1, temp_diff_d1, ylp_precip_prev_weekend, hist_mean_heavy_share, search_anxiety_d1` |

## 5. 外部來源與缺失特徵

| 來源 | 狀態 |
|---|---|
| calendar | OK |
| weather | {'TPE': 'OK', 'YLP': 'OK', 'TAY': 'OK'} |
| trends | OK |
| policy | OK |
| policy_available_columns | {'hov_active': True, 'hov_release_min': True, 'free_toll_active': True, 'free_toll_end_min': True} |
| event_count | 18 |

所有候選特徵皆有可用來源。

## 6. 防洩漏設計（train 與 test 共用同一套特徵定義）

- 依日期年份切分 2023-2024 train / 2025 test，未使用任何隨機切分。
- 當日 speed_kmh、volume_15m、heavy_share、travel_time_min 只作標的，從不作為特徵。
- 歷史同期平均車速／壅塞率／重車佔比僅由 train groupby 擬合後映射；train 使用 leave-one-out 去掉自身列。
- 天氣（YLP/TPE）與 Google Trends 只取 D-1，3C 前序週末累積雨量只取嚴格早於 D 的最近週末。
- train 與 test 共用完全相同的特徵建構函式，無個別調參或個別填值。
- 交管分鐘數來自事件前公告的 holiday_controls.csv，而非當日車流觀測；缺失時不建立特徵。

## 7. 資料限制與解讀注意

- 與即時車速相比，D-1 氣象／搜尋指數的預測力天生較弱；C 若無明顯增益，代表這些事前資訊邊際貢獻有限，而非管線錯誤。
- 1A 已修復為反假設驗證結果：ρ=0.348、p=0.157（未達顯著）；真實抓取值中「國道5號」全為 0、「雪隧」僅少數天非零，故搜尋焦慮指數主要由「高乘載／天氣」構成，本次 train 標準差為 1.6479577951011408。
- 研究範圍僅限國道：不推論省道（台9線）或市區道路；國2 因無 M05A eTag 資料不納入推論。
- 2025 測試面板僅至 11 月 23 日，未覆蓋全年；指標是該 53 天、3 走廊、96 時段的加權結果。
- Brier 採多分類 one-hot 與 predict_proba 的四類平方誤差加總（未除以類別數），取值範圍 0–2；P90 覆蓋率以真實旅行時間 ≤ P90 預測的比例計算。
- LightGBM 的非確定性來源已固定 random_state=42，但不同硬體仍可能有微小數值差異。
- 缺檔的外部來源（尤以 holiday_controls.csv）需由人工補齊後重跑，本程式不會以 0 或隨機值填補。
