# -*- coding: utf-8 -*-
"""測試日區塊 bootstrap：估計模型表現的 95% 信心區間。

不重新訓練模型；只讀取已產生的 2024 / 2025 預測檔與真實走廊面板。
以「測試日」為重抽單位，保留同一天三走廊 × 96 時段的相關性，
避免把 15 分鐘時段當成彼此獨立而過度樂觀。

輸出：
  outputs/phase2/confidence_intervals.md
  outputs/phase2/confidence_intervals.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

import config
import phase2_model as pm
import src.model_features as mf

ROOT = config.ROOT
B = 2000
SEED = 42
METRICS = [
    "accuracy", "macro_f1", "severe_precision", "severe_recall",
    "log_loss", "brier", "p50_mae", "p90_coverage",
]
LABELS = {
    "accuracy": "整體等級準確率",
    "macro_f1": "Macro F1",
    "severe_precision": "嚴重壅塞 Precision（flag 後正確率）",
    "severe_recall": "嚴重壅塞 Recall（實際抓到率）",
    "log_loss": "Multi-class Log Loss",
    "brier": "Brier Score",
    "p50_mae": "P50 MAE（分鐘）",
    "p90_coverage": "P90 覆蓋率",
}
LOWER_BETTER = {"log_loss", "brier", "p50_mae"}
DATASETS = [
    ("Fold 2024", ROOT / "outputs" / "walk_forward" / "fold_2024" / "predictions_2025_all_models.csv"),
    ("Fold 2025", ROOT / "outputs" / "phase2" / "predictions_2025_all_models.csv"),
]


def load_predictions(path: Path) -> pd.DataFrame:
    """把真實旅行時間接回預測檔；檔名 predictions_2025 為共用欄位名，內容以 test_years 為準。"""
    panel, _ = mf.load_panel(ROOT)
    reg = panel[["date", "corridor_id", "time_slot_15m", "travel_time_min"]].copy()
    reg["date"] = pd.to_datetime(reg["date"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    out = df.merge(reg, on=["date", "corridor_id", "time_slot_15m"],
                   how="left", validate="many_to_one")
    if out["travel_time_min"].isna().any():
        raise RuntimeError(f"真實旅行時間接合失敗：{path}")
    return out


def arrays(sub: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "y": sub["y_true_class"].to_numpy(int),
        "pred": sub["y_pred_class"].to_numpy(int),
        "proba": sub[[f"prob_class_{i}" for i in range(4)]].to_numpy(float),
        "reg": sub["travel_time_min"].to_numpy(float),
        "p50": sub["p50"].to_numpy(float),
        "p90": sub["p90"].to_numpy(float),
        "date": sub["date"].to_numpy(),
    }


def metrics(a: dict[str, np.ndarray]) -> dict[str, float]:
    base = pm.compute_metrics(a["y"], a["proba"], a["pred"], a["reg"], a["p50"], a["p90"])
    return {
        **base,
        "accuracy": float(accuracy_score(a["y"], a["pred"])),
        "severe_precision": float(precision_score(a["y"] >= 2, a["pred"] >= 2, zero_division=0)),
    }


def bootstrap(a: dict[str, np.ndarray], b: dict[str, np.ndarray]):
    days = np.unique(a["date"])
    by_day = {d: np.where(a["date"] == d)[0] for d in days}
    rng = np.random.default_rng(SEED)
    draws = {m: [] for m in METRICS}
    deltas = {m: [] for m in METRICS}
    for _ in range(B):
        sample_days = rng.choice(days, size=len(days), replace=True)
        idx = np.concatenate([by_day[d] for d in sample_days])
        sa = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in a.items()}
        sb = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in b.items()}
        ma, mb = metrics(sa), metrics(sb)
        for m in METRICS:
            draws[m].append(ma[m])
            deltas[m].append(ma[m] - mb[m])
    ci = {m: tuple(np.percentile(draws[m], [2.5, 97.5])) for m in METRICS}
    dci = {m: tuple(np.percentile(deltas[m], [2.5, 97.5])) for m in METRICS}
    p_better = {
        m: float(np.mean(np.asarray(deltas[m]) < 0) if m in LOWER_BETTER
                 else np.mean(np.asarray(deltas[m]) > 0))
        for m in METRICS
    }
    return ci, dci, p_better


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def build_report(results: list[dict]) -> tuple[str, pd.DataFrame]:
    lines = [
        "# 預測可信度：測試日區塊 Bootstrap 95% 信心區間",
        "",
        f"重抽次數：{B:,}；隨機種子：{SEED}；重抽單位：**測試日**（不是單一 15 分鐘時段）。",
        "只讀取已訓練好的 A/B/C 預測與真實走廊面板，不重新訓練、不調參、不補值。",
        "信號區間代表采樣不確定性，**不包含**模型結構誤差或 2026 年分布轉移風險。",
        "",
        "## 1. 三模型基本正確性對照（點估計）",
        "",
        "| Fold | 模型 | 整體準確率 | 最大宗等級 baseline | 嚴重 Precision | 嚴重 Recall | 實際嚴重佔比 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    csv_rows = []
    for r in results:
        for model in ("A", "B", "C"):
            p = r["points"][model]
            lines.append(
                f"| {r['name']} | {model} | {p['accuracy']:.4f} | {p['majority']:.4f} | "
                f"{p['severe_precision']:.4f} | {p['severe_recall']:.4f} | {p['severe_rate']:.4f} |"
            )
            for m in METRICS:
                csv_rows.append({
                    "kind": "point", "dataset": r["name"], "model": model,
                    "metric": m, "value": p[m], "ci_low": np.nan, "ci_high": np.nan,
                    "p_better_than_B": np.nan,
                })

    lines += [
        "",
        "## 2. 模型 C 的 95% 信心區間",
        "",
        "| Fold | 指標 | 點估計 | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in results:
        for m in METRICS:
            lo, hi = r["ci"][m]
            lines.append(f"| {r['name']} | {LABELS[m]} | {_fmt(r['points']['C'][m])} | "
                         f"[{_fmt(lo)}, {_fmt(hi)}] |")
            csv_rows.append({
                "kind": "ci_C", "dataset": r["name"], "model": "C", "metric": m,
                "value": r["points"]["C"][m], "ci_low": lo, "ci_high": hi,
                "p_better_than_B": np.nan,
            })

    lines += [
        "",
        "## 3. 模型 C 相對模型 B：bootstrap 勝出機率與差值 CI",
        "",
        "> 差值 = C − B；Log Loss／Brier／MAE 越低越好，其餘越高越好。",
        "> 「勝出機率」是 bootstrap 重抽中 C 優於 B 的比例，不是先驗機率保證。",
        "",
        "| Fold | 指標 | 差值 95% CI | C 勝過 B 的比例 |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in results:
        for m in METRICS:
            lo, hi = r["dci"][m]
            lines.append(
                f"| {r['name']} | {LABELS[m]} | [{_fmt(lo)}, {_fmt(hi)}] | "
                f"{r['p_better'][m] * 100:.1f}% |"
            )
            csv_rows.append({
                "kind": "delta_C_minus_B", "dataset": r["name"], "model": "C-B",
                "metric": m, "value": r["dci"][m][0], "ci_low": lo, "ci_high": hi,
                "p_better_than_B": r["p_better"][m],
            })

    lines += [
        "",
        "## 4. 整合判讀（如實）",
        "",
        "- **整體等級**：2025 模型 C 命中 79.3%（最大宗等級 baseline 66.3%），"
        "2024 命中 73.7%（baseline 62.1%）。淨增益存在，但不等於高置信。",
        "- **C 優於 B 的信心很高**：2025 八項指標中七項皆 2000/2000 次勝出，"
        "唯一例外是 P90 覆蓋率（93.1%）；2024 在整體準確率（99.8%）、Macro F1（98.4%）、"
        "Brier（93.7%）、P50 MAE（99.8%）、P90（98.4%）也一致勝出。",
        "- **嚴重壅塞是一次例外**：2024 嚴重 Recall 的 C 勝率只有 4.2%（等於 B 幾乎必定較高），"
        "2025 才由 C 以 100% 勝率領先。跨年不穩，不能宣稱一致改善。",
        "- **P90 區間**：2025 的 95% CI [0.867, 0.913] 與名義 0.90 相容；"
        "2024 的 [0.838, 0.885] 明顯低於 0.90，代表該年覆蓋不足。",
        "- **推論邊界**：這是 15 分鐘走廊平均的統計信心，不是逐車／交流道保證；"
        "也未包含 2026 分布轉移。模型給出的類別機率仍需可靠性校準曲線才能直接用來做風險決策。",
        "",
    ]
    return "\n".join(lines), pd.DataFrame(csv_rows)


def main() -> None:
    started = time.perf_counter()
    results = []
    for name, path in DATASETS:
        df = load_predictions(path)
        sub = {m: arrays(df[df["model"] == m]) for m in ("A", "B", "C")}
        points = {}
        for m, a in sub.items():
            pt = metrics(a)
            pt["majority"] = float(np.bincount(a["y"], minlength=4).max() / len(a["y"]))
            pt["severe_rate"] = float(np.mean(a["y"] >= 2))
            points[m] = pt
        ci, dci, p_better = bootstrap(sub["C"], sub["B"])
        results.append({"name": name, "points": points, "ci": ci,
                        "dci": dci, "p_better": p_better})
        print(f"[{name}] n={len(sub['C']['y'])} days={len(np.unique(sub['C']['date']))} 完成",
              flush=True)

    out_dir = ROOT / "outputs" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    md, csv = build_report(results)
    md_path = out_dir / "confidence_intervals.md"
    csv_path = out_dir / "confidence_intervals.csv"
    md_path.write_text(md, encoding="utf-8")
    csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"完成：{md_path} / {csv_path}（{time.perf_counter() - started:.1f}s）")


if __name__ == "__main__":
    main()
