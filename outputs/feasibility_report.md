# 國道連假塞車預測：階段一可行性評估排行榜

> 自動產生時間：2026-11-27 15:30　|　分析年份：2023, 2024, 2025
> 所有數字皆由真實公開資料計算；缺資料者標記為 `資料不足`，未以假資料填補。

## 一、資料來源與取得狀態

| 資料源 | 狀態 | 欄位/筆數 |
| --- | --- | --- |
| M05A (ETC) | OK | 見下表 |
| M06A (ETC) | DISABLED: --m06a-days 0 | 見下表 |
| weather | OK | 見下表 |
| trends | OK | 見下表 |
| calendar | OK | 見下表 |
| airport | MANUAL_NEEDED: data/external/airport/taoyuan_daily_pax.csv | 見下表 |
| hsr | MANUAL_NEEDED: data/external/hsr/hsr_holiday_capacity.csv | 見下表 |
| policy | MANUAL_NEEDED: data/external/policy/holiday_controls.csv | 見下表 |
| provincial | MANUAL_NEEDED: data/external/road/t9_hourly.csv | 見下表 |
| mega_events | MANUAL_NEEDED: data/external/events/mega_events.csv | 見下表 |
| lunar | OK | 見下表 |

## 二、清洗前後對比（強制鐵則執行結果）

| 項目 | 值 |
| --- | --- |
| 原始 M05A 列數 | 330088320 |
| 0車量+0車速（已轉缺失） | 93921133 |
| 相鄰時段線性插值補回 | 44163840 |
| 自由流 90km/h 補值 | 49757293 |
| 非走廊/無里程區間丟棄 | 307731168 |
| 全日零流量未啟用區間丟棄 | 7575840 |
| 走廊 15 分鐘面板列數 | 47520 |
| 走廊數 | 3 |
| 日期數 | 165 |

### 各走廊 15 分鐘面板摘要

| corridor_id | 列數 | 平均車速 | 平均流量 | 壅塞比例 |
| --- | --- | --- | --- | --- |
| N1_SB | 15840 | 89.120 | 424.437 | 0.054 |
| N5_NB | 15840 | 73.110 | 234.916 | 0.260 |
| N5_SB | 15840 | 77.381 | 242.908 | 0.106 |

## 三、24 假設檢定總表

| id | 假設 | 所需資料 | 狀態 | n | pearson_r | pearson_p | spearman_r | spearman_p | welch_p | cohens_d | effect | gain | shap_mean_abs | stability | 總分 | 備註 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5A | 請假攻略槓桿 | calendar, panel | 可用 | 18 | -0.568 | 0.014 | -0.637 | 0.004 | — | — | 0.637 | 0.537 | 0.004 | 1.000 | 4.450 |  |
| 5C | 長連假提早北返分流 | panel | 可用 | 18 | 0.423 | 0.080 | 0.530 | 0.024 | — | — | 0.530 | 0.436 | 0.202 | 1.000 | 4.100 |  |
| 3A | 天氣剪刀差（逃離雨天） | weather, panel | 可用 | 81 | -0.155 | 0.168 | -0.203 | 0.069 | — | — | 0.203 | 0.403 | 1918.063 | 1.000 | 3.950 |  |
| 6C | 臨界減慢前兆訊號 | five | 可用 | 6000 | 0.060 | 0.000 | 0.023 | 0.071 | — | — | 0.023 | 0.103 | 0.023 | 1.000 | 3.700 |  |
| 6B | 重車混流干擾比 | sec | 可用 | 165 | -0.163 | 0.037 | -0.195 | 0.012 | — | — | 0.195 | 0.488 | 0.517 | 0.667 | 3.500 |  |
| 5B | 補班日後補償反彈 | calendar, panel | 可用 | 18 | — | — | — | — | 0.002 | 1.629 | 1.629 | 0.314 | 0.076 | 0.500 | 3.300 |  |
| 3C | 前序週末降雨壓抑釋放 | weather, panel | 可用 | 18 | 0.279 | 0.261 | 0.121 | 0.633 | — | — | 0.121 | 0.305 | 0.045 | 0.667 | 3.000 |  |
| 3D | 強降雨容量崩跌臨界點 | weather, panel | 可用 | 3960 | -0.003 | 0.847 | -0.019 | 0.236 | — | — | 0.019 | 0.042 | 0.007 | 0.667 | 2.300 |  |
| 7A | 農民曆吉日效應 | lunar, panel | 可用 | 117 | — | — | — | — | 0.144 | -0.294 | 0.294 | 0.032 | 376.703 | 0.500 | 2.100 |  |
| 1A | 預報反身性（尖峰左移） | trends, panel | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | 錨點比值已跨批次縫合 |
| 1B | 替代道路分流實效 | provincial, trends | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | 需台9線流量檔；MANUAL_NEEDED: data/external/road/t9_hourly.csv |
| 1C | 目的地搜尋領先指標 | trends, panel | 資料不足 | 18 | — | — | — | — | — | — | — | — | — | — | — | 變異不足 |
| 2A | 高乘載水壩洩洪效應 | policy, sec | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | 需 holiday_controls.csv.hov_release_min（分鐘，自 00:00 起算） |
| 2B | 凌晨免收費壓線搶閘 | policy, panel | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | 需 holiday_controls.csv.free_toll_end_min |
| 2C | 匝道封閉鄰近虹吸 | policy, sec | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | 需 holiday_controls.csv.ramp_closure_gantry |
| 2D | 國3差別費率轉移 | policy, panel | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | 需 holiday_controls.csv.n3_discount |
| 3B | 棄遊轉向效應 | weather, panel | 資料不足 | 81 | — | — | — | — | — | — | — | — | — | — | — | 變異不足 |
| 4A | 出國潮抽乾國旅 | airport, panel | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | MANUAL_NEEDED: data/external/airport/taoyuan_daily_pax.csv |
| 4B | 送機潮引爆國2 | airport, N2 | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | MANUAL_NEEDED: data/external/airport/taoyuan_daily_pax.csv；且 TISVCLOUD M05A 無國2 門架（需 VD 補檔） |
| 4C | 高鐵滿載外溢 | hsr, sec | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | MANUAL_NEEDED: data/external/hsr/hsr_holiday_capacity.csv |
| 6A | 長途跨區旅次脆弱度 | M06A, panel | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | DISABLED: --m06a-days 0 |
| 7B | 萬人場館與繞境衝擊 | mega_events, sec | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | MANUAL_NEEDED: data/external/events/mega_events.csv |
| 8A | 瓶頸回堵衝擊波延遲 | sec | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — |  |
| 8B | 主線與省道同步崩潰點 | provincial, panel | 資料不足 | 0 | — | — | — | — | — | — | — | — | — | — | — | MANUAL_NEEDED: data/external/road/t9_hourly.csv |

### 評分口径

- 顯著性：p<0.001 且 |effect|≥0.5→5；p<0.01 且 ≥0.3→4；p<0.05→3；p<0.10→2；其餘 1。
- 模型增益：LightGBM 置換重要度 / Var(Y)，分位切 1–5 分（样本內估计，僅供排序）。
- 穩定度：各年效果方向與全年方向一致的比例（≥0.9→5 … <0.4→1）。
- 故事性：主觀編輯分 1–5，理由已列於程式 `STORY` 常數。
- 總分 = 0.35×顯著性 + 0.25×模型增益 + 0.20×穩定度 + 0.20×故事性。

## 四、可行性排行榜 Top 10

| id | 維度 | 假設 | 狀態 | n | effect | gain | stability | 顯著性 | 模型增益 | 穩定度 | 故事性 | 總分 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5A | 日曆拓撲 | 請假攻略槓桿 | 可用 | 18 | 0.637 | 0.537 | 1.000 | 4 | 5 | 5 | 4 | 4.450 |
| 5C | 日曆拓撲 | 長連假提早北返分流 | 可用 | 18 | 0.530 | 0.436 | 1.000 | 3 | 5 | 5 | 4 | 4.100 |
| 3A | 跨區氣候 | 天氣剪刀差（逃離雨天） | 可用 | 81 | 0.203 | 0.403 | 1.000 | 2 | 5 | 5 | 5 | 3.950 |
| 6C | 微觀車流 | 臨界減慢前兆訊號 | 可用 | 6000 | 0.023 | 0.103 | 1.000 | 2 | 4 | 5 | 5 | 3.700 |
| 6B | 微觀車流 | 重車混流干擾比 | 可用 | 165 | 0.195 | 0.488 | 0.667 | 3 | 5 | 3 | 3 | 3.500 |
| 5B | 日曆拓撲 | 補班日後補償反彈 | 可用 | 18 | 1.629 | 0.314 | 0.500 | 3 | 5 | 2 | 3 | 3.300 |
| 3C | 跨區氣候 | 前序週末降雨壓抑釋放 | 可用 | 18 | 0.121 | 0.305 | 0.667 | 1 | 5 | 3 | 4 | 3.000 |
| 3D | 跨區氣候 | 強降雨容量崩跌臨界點 | 可用 | 3960 | 0.019 | 0.042 | 0.667 | 1 | 3 | 3 | 3 | 2.300 |
| 7A | 在地文化 | 農民曆吉日效應 | 可用 | 117 | 0.294 | 0.032 | 0.500 | 1 | 3 | 2 | 3 | 2.100 |

## 五、黃金三角 Top 3（一時間／一空間／一宏觀）

> 軸定義：時間＝事件內/事件間的時序動態（含 5B 補班後反彈、5C 長連假提早北返）；空間＝走廊/跨區轉移；宏觀＝旅次需求的上位驅動。1A/2A–2D 因搜尋指數變異不足或交管檔缺失，暫時無可用候選，故以真實可用的日曆時序假設代理時間軸。

| 軸 | id | 假設 | 總分 | 建議 |
| --- | --- | --- | --- | --- |
| 時間 | 5C | 長連假提早北返分流 | 4.100 | n=18，effect=0.530，p=0.0803 |
| 空間 | 3A | 天氣剪刀差（逃離雨天） | 3.950 | n=81，effect=0.203，p=0.168 |
| 宏觀 | 5A | 請假攻略槓桿 | 4.450 | n=18，effect=0.637，p=0.014 |

## 六、候選探索圖

- `outputs/eda_candidates/3A_天氣剪刀差（逃離雨天）.png`
- `outputs/eda_candidates/3C_前序週末降雨壓抑釋放.png`
- `outputs/eda_candidates/3D_強降雨容量崩跌臨界點.png`
- `outputs/eda_candidates/5A_請假攻略槓桿.png`
- `outputs/eda_candidates/5B_補班日後補償反彈.png`
- `outputs/eda_candidates/5C_長連假提早北返分流.png`
- `outputs/eda_candidates/6B_重車混流干擾比.png`
- `outputs/eda_candidates/6C_臨界減慢前兆訊號.png`
- `outputs/eda_candidates/7A_農民曆吉日效應.png`

## 七、資料缺口與手動補檔

| 資料 | 放置路徑 | 必要欄位 |
| --- | --- | --- |
| CODiS 逐時觀測 | `data/external/weather/hourly_{TPE,YLP,TAY}.csv` | datetime, precip_mm, temp_c |
| Google Trends | `data/external/trends/trends_{geo}.csv` | date, 關鍵字欄位, geo；或加 `--fetch-trends` |
| 人事行政總處日曆 | `data/external/calendar/office_calendar_{year}.csv` | 西元日期, 星期, 是否放假, 備註（已由 data.gov.tw 14718 取得） |
| 桃機人次 | `data/external/airport/taoyuan_daily_pax.csv` | date, departures, arrivals[, hour] |
| 高鐵疏運 | `data/external/hsr/hsr_holiday_capacity.csv` | date, load_factor, extra_train_ratio |
| 連假交管 | `data/external/policy/holiday_controls.csv` | date,event,hov_active,hov_release_min,free_toll_active,free_toll_end_min,ramp_closure_gantry,n3_discount |
| 台9線 VD | `data/external/road/t9_hourly.csv` | date, datetime, t9_volume, t9_travel_min |
| 大型活動 | `data/external/events/mega_events.csv` | datetime, km, hours_from_end |
| 門架里程 | `data/external/gantry_km.csv` | gantry_id, km（選填；預設由 ID 推 km=數字/10） |

> 國道 2 無 M05A/M06A eTag 門架，4B 需另接高公局 VD 資料；此狀態已如實標記。
