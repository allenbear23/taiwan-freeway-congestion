# 資料下載連結（階段一）

> .all 連結於 2026-11-27 實測；狀態 `OK` = HTTP 200 且內容可解析，`MANUAL_NEEDED` = 尚無可直接下載的機器可讀檔。
> 本專案不使用假資料；`MANUAL_NEEDED` 的項目留空並在報表標記，不以合成值填補。

## 一、已取得並已接入 pipeline（OK）

### 1. 政府行政機關辦公日曆表 → `5B` 已解鎖
- 資料集頁：https://data.gov.tw/dataset/14718
- 完整資料集清單（可用關鍵字查 ID）：https://data.gov.tw/api/front/dataset/export?format=csv
- 直接 CSV（Real URL，UTF-8 BOM）：
  - 2023（112 年）：`https://www.dgpa.gov.tw/FileConversion?filename=dgpa/files/202407/dd89fe8c-25f5-4035-a31e-7c14b44a74d7.csv&nfix=&name=112年中華民國政府行政機關辦公日曆表.csv`
  - 2024（113 年）：`https://www.dgpa.gov.tw/FileConversion?filename=dgpa/files/202407/777152e9-fdd1-4a61-876c-2733e7692538.csv&nfix=&name=113年中華民國政府行政機關辦公日曆表.csv`
  - 2025（114 年）：`https://www.dgpa.gov.tw/FileConversion?filename=dgpa/files/202407/22f9fcbc-fbb2-4387-8bcf-73b2279666c2.csv&nfix=&name=114年中華民國政府行政機關辦公日曆表.csv`
- 本地路徑：`data/external/calendar/office_calendar_{2023,2024,2025}.csv`（1096 列，補班 8 筆）

### 2. 國道 ETC 車流 M05A／M06A（主力資料源）
- 入口：https://tisvcloud.freeway.gov.tw/history/TDCS/
- M05A 範例（2024-02-08）：`https://tisvcloud.freeway.gov.tw/history/TDCS/M05A/20240208/M05A_20240208.tar.gz`
- M06A 範例（2024-02-08）：`https://tisvcloud.freeway.gov.tw/history/TDCS/M06A/20240208/M06A_20240208.tar.gz`
- 注意：近期日期改用 `YYYYMMDD/HH/` 逐小時路徑；tar 內成员為巢狀，需以 basename  filtering。M05A 無表頭。
- 本專案 M05A 165 天已全數下載；M06A 預設 `--m06a-days 0`（`6A` 因此標記 DISABLED）。

### 3. 氣象（Open-Meteo ERA5 archive，免金鑰）
- API 文件：https://open-meteo.com/en/docs/historical-weather-api
- Endpoint：`https://archive-api.open-meteo.com/v1/archive`
- 本地：`data/external/weather/hourly_{TPE,YLP,TAY}.csv`（共 81144 列）

### 4. Google Trends（pytrends，需 `urllib3<2`）
- 無穩定公開直接連結；由 `src/external.py::fetch_trends` 抓取後縫合。
- 本地：`data/external/trends/trends_TW.csv`（1127 × 7）
- 注意：搜尋指數極稀疏（宜蘭住宿僅 5 個不同值），這是 `1A/1C` 判定「變異不足」的真實原因，不是程式錯誤。

## 二、已找到官方來源但需手動編碼／尚未能自動下載（MANUAL_NEEDED）

### 5. 連假交通疏導措施 → 可解鎖 `2A–2D`
- 高公局公告總覽：https://www.freeway.gov.tw/Publish.aspx?cnid=1512
- 完整 PDF 索引（104–115 年）：https://www.freeway.gov.tw/Upload/Html/201943202/index.html
- 自動抽取全部 PDF 連結：`python extract_holiday_pdfs.py`
- 113 年春节範例（疏導措施一覽表）：
  `https://www.freeway.gov.tw/Upload/Html/201943202/inf-113/113%E5%B9%B4%E6%98%A5%E7%AF%80%E9%80%A3%E7%BA%8C%E5%81%87%E6%9C%9F%E5%9C%8B%E9%81%93%E4%BA%A4%E9%80%9A%E7%96%8F%E5%B0%8E%E6%8E%AA%E6%96%BD%E4%B8%80%E8%A6%BD%E8%A1%A8.pdf`
- 仍需把 PDF 表格人工轉成 `data/external/policy/holiday_controls.csv`，欄位：
  `date,event,hov_active,hov_release_min,free_toll_active,free_toll_end_min,ramp_closure_gantry,n3_discount`

### 6. 桃園機場旅客人次 → `4A` 所需的每日檔尚未解鎖
- 官方月統計總表（僅月維度，不適合 `4A`）：http://www.taoyuan-airport.com/uploads/govdata/PassengerVolume.csv
- **新發現：每日資料確實存在**（但僅 2022-11 單月，ID 似為日期流水號）：
  `https://odp.taoyuan-airport.com/dataset/2022110905?format=csv`
  欄位：`日期,入境人數,出境人數,每日入境過境人數,每日出境過境人數,每日入境轉機人數,每日出境轉機人數,每日總計`
- 需再找到 2023–2025 各月資料集 ID 才能補齊；目前 `data/external/airport/taoyuan_daily_pax.csv` 標記 `MANUAL_NEEDED`。

### 7. 高鐵疏運／運能 → `4C`（MANUAL_NEEDED）
- 資料.gov.tw 查「高鐵車站數」等資料集，但查無可用的「連假每日載客率／加開班次」公開檔。
- 鐵道局相關 CSV 受 Incapsula 防護，程式下載得到 HTML 挑战頁，非真實資料，故不接入。

### 8. 台 9 線每小時流量 → `1B/8B`（MANUAL_NEEDED）
- 需公路局 VD 逐時檔；公開資料集查無符合格式的每日／逐時 CSV。

### 9. 大型活動／繞境 → `7B`（MANUAL_NEEDED）
- 查無涵蓋 2023–2025、附明確 km 與時段的全國性結構化公開檔。

### 10. 國 2（N2）門架／VD → `4B`（MANUAL_NEEDED）
- TISVCLOUD M05A/M06A 只含國 1、國 3、國 5；國 2 需另接高公局 VD，查無可直接下載檔。

### 11. 門架里程對照表（選填）
- 目前由門架 ID 推 km（`NNNN/10 ≈ km`）；若有官方對照表可放 `data/external/gantry_km.csv` 覆寫。

---

## 三、目前排行榜與黃金三角（2023–2025 真實資料）

| 排名 | id | 假設 | 總分 | n |
| --- | --- | --- | --- | --- |
| 1 | 5A | 請假攻略槓桿 | 4.450 | 18 |
| 2 | 5C | 長連假提早北返分流 | 4.100 | 18 |
| 3 | 3A | 天氣剪刀差（逃離雨天） | 3.950 | 81 |
| 4 | 6C | 臨界減慢前兆訊號 | 3.700 | 6000 |
| 5 | 6B | 重車混流干擾比 | 3.500 | 165 |

**黃金三角 Top 3**
- 時間：5C 長連假提早北返分流（4.100）
- 空間：3A 天氣剪刀差（3.950）
- 宏觀：5A 請假攻略槓桿（4.450）

> 時間軸以 5B/5C 日曆時序假設代理：1A 因 Google Trends 指數變異不足、2A–2D 因交管 PDF 尚未結構化，皆無可用候選。
