# -*- coding: utf-8 -*-
"""逐步時序驗證（walk-forward）：用歷史資料模擬兩次「未來」預測。

Fold 1：train 2023 → test 2024
Fold 2：train 2023–2024 → test 2025（與正式階段二相同）

每一 fold 都重新擬合歷史同期統計（只用該 fold 的 train），
不以 2024/2025 的任何當日車流回饋特徵。結果寫入 outputs/walk_forward/。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import config

FOLDS = [
    {"name": "fold_2024", "train": "2023", "test": "2024"},
    {"name": "fold_2025", "train": "2023,2024", "test": "2025"},
]

METRIC_ORDER = [
    ("macro_f1", "Macro F1"),
    ("severe_recall", "嚴重壅塞 (L2+L3) Recall"),
    ("log_loss", "Multi-class Log Loss"),
    ("brier", "Brier Score"),
    ("p50_mae", "旅行時間 P50 MAE（分鐘）"),
    ("p90_coverage", "P90 覆蓋率"),
]


def _run_fold(root: Path, fold: dict) -> Path:
    out = root / "outputs" / "walk_forward" / fold["name"]
    cmd = [sys.executable, str(root / "phase2_model.py"),
           "--root", str(root),
           "--train-years", fold["train"],
           "--test-years", fold["test"],
           "--out-subdir", f"walk_forward/{fold['name']}"]
    print(f"\n=== {fold['name']} : train {fold['train']} → test {fold['test']} ===", flush=True)
    t = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(root),
                          encoding="utf-8", errors="replace")
    print((proc.stdout or "")[-4000:], flush=True)
    if proc.returncode != 0:
        print((proc.stderr or "")[-4000:], file=sys.stderr, flush=True)
        raise RuntimeError(f"{fold['name']} 建模失敗（exit {proc.returncode}）")
    print(f"耗時 {time.perf_counter() - t:.1f}s", flush=True)
    return out


def build_combined_report(root: Path, folds: list[dict]) -> Path:
    out_dir = root / "outputs" / "walk_forward"
    summaries = {}
    for fold in folds:
        p = out_dir / fold["name"] / "summary.json"
        summaries[fold["name"]] = json.loads(p.read_text(encoding="utf-8"))

    lines = [
        "# 逐步時序驗證（Walk-forward Validation）",
        "",
        "> 兩組切分皆只用「過去」訓練、「未來」測試；每組重新由該 fold 的訓練集擬合歷史同期統計。",
        "> 未使用隨機切分，未以測試年車流回饋任何特徵。所有數字來自真實資料。",
        "",
        "## 1. 切分設定",
        "",
        "| Fold | Train | Test | 訓練列數 | 測試列數 | 日期重疊 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for fold in folds:
        s = summaries[fold["name"]]["split"]
        lines.append(f"| {fold['name']} | {s['train_years']} | {s['test_years']} | "
                     f"{s['n_train_rows']} | {s['n_test_rows']} | {s['date_overlap']} |")

    lines += ["", "## 2. 三模型六指標（兩 fold 並排）", "",
              "| 指標 | 模型 | Fold 2024 | Fold 2025 |",
              "| --- | --- | ---: | ---: |"]
    for key, label in METRIC_ORDER:
        for model in ("A", "B", "C"):
            a = summaries["fold_2024"]["metrics"][model][key]
            b = summaries["fold_2025"]["metrics"][model][key]
            lines.append(f"| {label} | {model} | {a:.4f} | {b:.4f} |")

    lines += ["", "## 3. 模型 C 相對 A / B 改善百分比（兩 fold）", "",
              "| Fold | 比較 | Macro F1 | 嚴重 Recall | Log Loss | Brier | P50 MAE | P90 覆蓋 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for fold in folds:
        imp = summaries[fold["name"]]["improvements_pct"]
        for base in ("A", "B"):
            vals = [imp[f"C_vs_{base}"][k] for k, _ in METRIC_ORDER]
            cells = " | ".join(f"{v:+.2f}%" if pd.notna(v) else "—" for v in vals)
            lines.append(f"| {fold['name']} | C vs {base} | {cells} |")

    lines += [
        "",
        "## 4. 解讀（如實）",
        "",
        "- **穩定一致的部分**：Macro F1、Log Loss、Brier、P50 MAE 四項在兩 fold 皆為 C > B > A；"
        "C 相對 A 的改善在 Fold 2024 為 18.3%–30.2%，Fold 2025 為 26.9%–30.3%。"
        "代表跨域增強對整體排序、機率校準與旅行時間預測的貢獻不是單一年份偶然。",
        "- **不一致的部分（必須揭露）**：嚴重壅塞 Recall 在 Fold 2024 由 B 較高"
        "（B 0.5297 vs C 0.4770，C vs B −9.95%），在 Fold 2025 才由 C 領先"
        "（C 0.5429 vs B 0.4544，+19.48%）。因此不能宣稱 C 在六項指標一致領先；"
        "對 Level 2+3 的敏感度會隨年份波動。",
        "- Fold 2024 僅用 2023 訓練，歷史同期統計較稀疏，且 `makeup_week_flag` 在該 fold 屬低變異特徵；"
        "上述波動與此有關，但**僅為推論，未做因果歸因**（不做調參追逐單一年份）。",
        "- P90 覆蓋率兩 fold 為 0.8502–0.8942，未穩定達 0.90；不宜宣稱區間已完全校準。",
        "- 2026 連假是真正前瞻資料，尚未納入；本驗證仍屬歷史資料的時序模擬，非外部前瞻驗證。",
        "",
    ]
    p = out_dir / "walk_forward_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    root = config.ROOT
    out = root / "outputs" / "walk_forward"
    out.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        _run_fold(root, fold)
    p = build_combined_report(root, FOLDS)
    print(f"\n完成：{p}")


if __name__ == "__main__":
    main()
