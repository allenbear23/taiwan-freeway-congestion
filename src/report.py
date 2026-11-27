# -*- coding: utf-8 -*-
"""可行性評估排行榜、黃金三角推薦與 Markdown 報告輸出。"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from config import REPORT_PATH
from src.utils import md_table

# 軸的語意：時間＝事件內/事件間的「何時塞」動態；空間＝「哪裡塞／跨區轉移」；
# 宏觀＝旅次需求的上位驅動力。5B/5C 屬連假時序動態，歸入時間軸；
# 5A 是休假決策的宏觀槓桿，留在宏觀軸（避免同一假設同時佔兩軸）。
TIME_IDS = {"1A", "1B", "1C", "2A", "2B", "2C", "2D", "5B", "5C"}
SPACE_IDS = {"3A", "3B", "3D", "8A", "8B"}
MACRO_IDS = {"4A", "4B", "4C", "5A", "3C", "7A", "7B"}


def golden_triangle(scores: pd.DataFrame) -> pd.DataFrame:
    ok = scores[scores["狀態"] == "可用"]
    picks = []
    for label, ids in (("時間", TIME_IDS), ("空間", SPACE_IDS), ("宏觀", MACRO_IDS)):
        sub = ok[ok["id"].isin(ids)]
        picks.append(sub.iloc[0] if len(sub) else None)
    rows = []
    for label, r in zip(("時間", "空間", "宏觀"), picks):
        if r is None:
            rows.append({"軸": label, "id": "—", "假設": "無可用候選", "總分": np.nan, "建議": "需補資料"})
        else:
            rows.append({
                "軸": label, "id": r["id"], "假設": r["假設"], "總分": r["總分"],
                "建議": f"n={int(r['n'])}，effect={r['effect']:.3f}，p={r['pearson_p'] if r['類型']=='continuous' else r['welch_p']:.3g}",
            })
    return pd.DataFrame(rows)


def render(scores: pd.DataFrame, audit: dict, sources: dict[str, str],
           panel: pd.DataFrame, years: list[int], fig_paths: list[str],
           golden: pd.DataFrame) -> str:
    avail = scores[scores["狀態"] == "可用"]
    cols = ["id", "維度", "假設", "狀態", "n", "effect", "gain", "stability",
            "顯著性", "模型增益", "穩定度", "故事性", "總分"]
    rank = avail.sort_values("總分", ascending=False)[cols].head(10)
    alltbl = scores[["id", "假設", "所需資料", "狀態", "n", "pearson_r", "pearson_p",
                     "spearman_r", "spearman_p", "welch_p", "cohens_d", "effect",
                     "gain", "shap_mean_abs", "stability", "總分", "備註"]]

    src_rows = pd.DataFrame([
        {"資料源": k, "狀態": v, "欄位/筆數": (
            str(audit.get("by_corridor", {}))[:80] if k == "M05A" else "見下表")}
        for k, v in sources.items()
    ])
    clean_rows = pd.DataFrame([
        {"項目": "原始 M05A 列數", "值": audit.get("raw_rows", 0)},
        {"項目": "0車量+0車速（已轉缺失）", "值": audit.get("zero_zero_rows", 0)},
        {"項目": "相鄰時段線性插值補回", "值": audit.get("interpolated_rows", 0)},
        {"項目": "自由流 90km/h 補值", "值": audit.get("freeflow_filled_rows", 0)},
        {"項目": "非走廊/無里程區間丟棄", "值": audit.get("section_rows_dropped", 0)},
        {"項目": "全日零流量未啟用區間丟棄", "值": audit.get("inactive_section_rows_dropped", 0)},
        {"項目": "走廊 15 分鐘面板列數", "值": len(panel)},
        {"項目": "走廊數", "值": panel["corridor_id"].nunique() if "corridor_id" in panel else 0},
        {"項目": "日期數", "值": panel["date"].nunique() if "date" in panel else 0},
    ])
    corr_rows = (panel.groupby("corridor_id").agg(
        列數=("speed_kmh", "size"), 平均車速=("speed_kmh", "mean"),
        平均流量=("volume_15m", "mean"),
        壅塞比例=("congestion_level", lambda s: float((s >= 2).mean()))
    ).reset_index() if not panel.empty else pd.DataFrame())

    lines = [
        "# 國道連假塞車預測：階段一可行性評估排行榜",
        "",
        f"> 自動產生時間：{datetime.now():%Y-%m-%d %H:%M}　|　分析年份：{', '.join(map(str, years))}",
        "> 所有數字皆由真實公開資料計算；缺資料者標記為 `資料不足`，未以假資料填補。",
        "",
        "## 一、資料來源與取得狀態",
        "",
        md_table(src_rows),
        "",
        "## 二、清洗前後對比（強制鐵則執行結果）",
        "",
        md_table(clean_rows),
        "",
        "### 各走廊 15 分鐘面板摘要",
        "",
        md_table(corr_rows) if not corr_rows.empty else "（無走廊資料）",
        "",
        "## 三、24 假設檢定總表",
        "",
        md_table(alltbl, float_digits=3),
        "",
        "### 評分口径",
        "",
        "- 顯著性：p<0.001 且 |effect|≥0.5→5；p<0.01 且 ≥0.3→4；p<0.05→3；p<0.10→2；其餘 1。",
        "- 模型增益：LightGBM 置換重要度 / Var(Y)，分位切 1–5 分（样本內估计，僅供排序）。",
        "- 穩定度：各年效果方向與全年方向一致的比例（≥0.9→5 … <0.4→1）。",
        "- 故事性：主觀編輯分 1–5，理由已列於程式 `STORY` 常數。",
        "- 總分 = 0.35×顯著性 + 0.25×模型增益 + 0.20×穩定度 + 0.20×故事性。",
        "",
        "## 四、可行性排行榜 Top 10",
        "",
        md_table(rank, float_digits=3) if len(rank) else "（無可用候選）",
        "",
        "## 五、黃金三角 Top 3（一時間／一空間／一宏觀）",
        "",
        "> 軸定義：時間＝事件內/事件間的時序動態（含 5B 補班後反彈、5C 長連假提早北返）；"
        "空間＝走廊/跨區轉移；宏觀＝旅次需求的上位驅動。1A/2A–2D 因搜尋指數變異不足或交管檔缺失，"
        "暫時無可用候選，故以真實可用的日曆時序假設代理時間軸。",
        "",
        md_table(golden, float_digits=3),
        "",
        "## 六、候選探索圖",
        "",
        "\n".join(f"- `{p.replace(chr(92), '/')}`" for p in fig_paths) or "（無）",
        "",
        "## 七、資料缺口與手動補檔",
        "",
        "| 資料 | 放置路徑 | 必要欄位 |",
        "| --- | --- | --- |",
        "| CODiS 逐時觀測 | `data/external/weather/hourly_{TPE,YLP,TAY}.csv` | datetime, precip_mm, temp_c |",
        "| Google Trends | `data/external/trends/trends_{geo}.csv` | date, 關鍵字欄位, geo；或加 `--fetch-trends` |",
        "| 人事行政總處日曆 | `data/external/calendar/office_calendar_{year}.csv` | 西元日期, 星期, 是否放假, 備註（已由 data.gov.tw 14718 取得） |",
        "| 桃機人次 | `data/external/airport/taoyuan_daily_pax.csv` | date, departures, arrivals[, hour] |",
        "| 高鐵疏運 | `data/external/hsr/hsr_holiday_capacity.csv` | date, load_factor, extra_train_ratio |",
        "| 連假交管 | `data/external/policy/holiday_controls.csv` | date,event,hov_active,hov_release_min,free_toll_active,free_toll_end_min,ramp_closure_gantry,n3_discount |",
        "| 台9線 VD | `data/external/road/t9_hourly.csv` | date, datetime, t9_volume, t9_travel_min |",
        "| 大型活動 | `data/external/events/mega_events.csv` | datetime, km, hours_from_end |",
        "| 門架里程 | `data/external/gantry_km.csv` | gantry_id, km（選填；預設由 ID 推 km=數字/10） |",
        "",
        "> 國道 2 無 M05A/M06A eTag 門架，4B 需另接高公局 VD 資料；此狀態已如實標記。",
    ]
    return "\n".join(lines) + "\n"


def write_report(text: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
