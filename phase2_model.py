# -*- coding: utf-8 -*-
"""階段二完整建模管線：時序切分 → 三組消融 → 2025 test 指標 → 輸出。

執行：
    python phase2_model.py [--root /path/to/project]

設計重點（詳見 ablation_report.md）：
* 2023–2024 為 train、2025 為 test，從未使用 sklearn train_test_split。
* 歷史同期統計只由 train 擬合；train 本身用 leave-one-out 避免自洩漏。
* 天氣／Trends 只用 D-1（或前序週末），當日交通觀測完全不進特徵。
* 外部來源缺檔或缺欄時直接不建立對應特徵，並在 summary/report 標記，不補零。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, recall_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402
from src import model_features as mf  # noqa: E402

CLASS_LABELS = [0, 1, 2, 3]
QUANTILES = (0.5, 0.9)

CLF_PARAMS = {
    "objective": "multiclass",
    "num_class": 4,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}
REG_PARAMS = {
    "objective": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

A_BASE = ["hist_mean_speed", "hist_mean_level", "hist_severe_rate"]
B_TIME = ["slot_sin", "slot_cos", "hour_sin", "hour_cos", "weekday", "is_weekend",
          "event_day_index", "is_holiday_period"]
C_EXTRA = [
    "leave_leverage", "makeup_week_flag", "event_total_days", "days_to_event_end",
    "min_to_hov_release", "min_to_free_toll_end",
    "tpe_precip_d1", "precip_scissors_d1", "temp_diff_d1",
    "ylp_precip_prev_weekend", "hist_mean_heavy_share", "search_anxiety_d1",
]
METRIC_LABELS = {
    "macro_f1": "Macro F1",
    "severe_recall": "嚴重壅塞 (L2+L3) Recall",
    "log_loss": "Multi-class Log Loss",
    "brier": "Brier Score (one-hot, 四類加總)",
    "p50_mae": "旅行時間 P50 MAE (分鐘)",
    "p90_coverage": "P90 覆蓋率",
}
METRIC_DIRECTION = {
    "macro_f1": "higher", "severe_recall": "higher", "log_loss": "lower",
    "brier": "lower", "p50_mae": "lower", "p90_coverage": "higher",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 ablation modeling pipeline")
    parser.add_argument("--root", type=str, default=None,
                        help="專案根目錄；預設使用 config.ROOT（Colab 可覆蓋）")
    return parser.parse_args(argv)


def _finite_count(series: pd.Series) -> int:
    return int(np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)).sum())


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _pct_change(base: float, new: float, direction: str) -> float:
    """改善百分比；direction=lower 表示下降為改善。"""
    if not np.isfinite(base) or abs(base) < 1e-12 or not np.isfinite(new):
        return float("nan")
    return (base - new) / abs(base) * 100.0 if direction == "lower" else (new - base) / abs(base) * 100.0


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray,
                    y_reg: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict[str, float]:
    one_hot = np.eye(len(CLASS_LABELS), dtype=float)[y_true]
    return {
        "macro_f1": float(f1_score(y_true, pred, labels=CLASS_LABELS,
                                   average="macro", zero_division=0)),
        "severe_recall": float(recall_score(y_true >= 2, pred >= 2, zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=CLASS_LABELS)),
        "brier": float(np.mean(np.sum((proba - one_hot) ** 2, axis=1))),
        "p50_mae": float(np.mean(np.abs(y_reg - p50))),
        "p90_coverage": float(np.mean(y_reg <= p90)),
    }


def fit_one(X_train: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series,
            X_test: pd.DataFrame):
    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(X_train, y_cls)
    raw_proba = clf.predict_proba(X_test)
    proba = np.zeros((len(X_test), len(CLASS_LABELS)), dtype=float)
    for i, cls in enumerate(clf.classes_):
        proba[:, int(cls)] = raw_proba[:, i]
    pred = proba.argmax(axis=1)

    regressors: dict[float, lgb.LGBMRegressor] = {}
    predictions: dict[float, np.ndarray] = {}
    for alpha in QUANTILES:
        reg = lgb.LGBMRegressor(**{**REG_PARAMS, "alpha": alpha})
        reg.fit(X_train, y_reg)
        regressors[alpha] = reg
        predictions[alpha] = np.asarray(reg.predict(X_test), dtype=float)
    # 分位數單調性：P90 不得低於 P50（純後處理，不對真實值做任何補值）
    predictions[0.9] = np.maximum(predictions[0.9], predictions[0.5])
    return {"clf": clf, "regressors": regressors, "proba": proba, "pred": pred,
            "p50": predictions[0.5], "p90": predictions[0.9]}


def build_report(out_dir: Path, root: Path, summary: dict, metrics: dict,
                 improvements: dict, missing: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# 階段二：2025 時序消融建模報告\n")
    lines.append(f"產生時間：{summary['generated_at']}　耗時：{summary['elapsed_seconds']:.1f} 秒\n")
    lines.append("## 1. 時序切分與資料規模\n")
    split = summary["split"]
    lines.append("| 項目 | 內容 |\n|---|---|")
    lines.append(f"| Train | {split['train_years']}（{split['n_train_days']} 天 / {split['n_train_rows']} 列） |")
    lines.append(f"| Test | {split['test_years']}（{split['n_test_days']} 天 / {split['n_test_rows']} 列） |")
    lines.append(f"| 隨機切分 | 未使用；依日期年份嚴格切分，train/test 日期重疊：{split['date_overlap']} |")
    lines.append(f"| 面板 | {summary['panel']['panel_rows']} 列 / {summary['panel']['panel_days']} 天 / "
                 f"{len(summary['panel']['corridors'])} 走廊 |")
    lines.append(f"| 標的一致性 | congestion_level 與 speed 推導不一致列數："
                 f"{summary['panel']['congestion_level_mismatch_vs_speed']} |")
    lines.append(f"| 分類/回歸標的 | congestion_level (0–3) / travel_time_min |")

    lines.append("\n## 2. 三消融模型的 2025 test 指標\n")
    lines.append("| 指標 | A Baseline | B 傳統 | C 增強 |\n|---|---:|---:|---:|")
    for key, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {_fmt(metrics['A'][key])} | {_fmt(metrics['B'][key])} | {_fmt(metrics['C'][key])} |")

    lines.append("\n## 3. C 相對 A / B 的改善百分比\n")
    lines.append("| 指標 | C vs A | C vs B |\n|---|---:|---:|")
    for key in METRIC_LABELS:
        va = improvements[f"C_vs_A"][key]
        vb = improvements[f"C_vs_B"][key]
        lines.append(f"| {METRIC_LABELS[key]} | {va:+.2f}% | {vb:+.2f}% |")
    cov_pp_a = (metrics["C"]["p90_coverage"] - metrics["A"]["p90_coverage"]) * 100
    cov_pp_b = (metrics["C"]["p90_coverage"] - metrics["B"]["p90_coverage"]) * 100
    lines.append(f"\nP90 覆蓋率絕對變化：C vs A {cov_pp_a:+.2f} 個百分點；"
                 f"C vs B {cov_pp_b:+.2f} 個百分點（理想值約 90%）。")
    lines.append("改善定義：F1/Recall/覆蓋率越高越好，取 (C−base)/|base|；"
                 "Log Loss/Brier/MAE 越低越好，取 (base−C)/|base|。")

    lines.append("\n## 4. 特徵組合\n")
    lines.append("| 模型 | 特徵數 | 特徵 |\n|---|---:|---|")
    for model in ("A", "B", "C"):
        feats = summary["feature_lists"][model]
        lines.append(f"| {model} | {len(feats)} | `{', '.join(feats)}` |")

    lines.append("\n## 5. 外部來源與缺失特徵\n")
    lines.append("| 來源 | 狀態 |\n|---|---|")
    for name, status in summary["data_sources"].items():
        if name.endswith("_keywords"):
            continue
        lines.append(f"| {name} | {status} |")
    if missing:
        lines.append("\n下列特徵因來源不存在／欄位不足／無有限值而**未放入 C**，未補零或偽造：\n")
        for item in missing:
            lines.append(f"- `{item['feature']}`：{item['reason']}")
    else:
        lines.append("\n所有候選特徵皆有可用來源。")

    lines.append("\n## 6. 防洩漏設計（train 與 test 共用同一套特徵定義）\n")
    for note in summary["leakage_controls"]:
        lines.append(f"- {note}")

    lines.append("\n## 7. 資料限制與解讀注意\n")
    lines.append("- 與即時車速相比，D-1 氣象／搜尋指數的預測力天生較弱；"
                 "C 若無明顯增益，代表這些事前資訊邊際貢獻有限，而非管線錯誤。")
    low_var = [f for f, d in summary["feature_diagnostics"].items()
               if d["train_std"] is not None and d["train_std"] < 1e-4]
    if low_var:
        lines.append(f"- 低變異特徵（train 標準差 < 1e-4，解讀請保守）：{', '.join(low_var)}。")
    anxiety_std = summary["feature_diagnostics"].get("search_anxiety_d1", {}).get("train_std")
    lines.append("- 1A 已修復為反假設驗證結果：ρ=0.348、p=0.157（未達顯著）；真實抓取值中"
                 "「國道5號」全為 0、「雪隧」僅少數天非零，故搜尋焦慮指數主要由「高乘載／天氣」構成，"
                 f"本次 train 標準差為 {anxiety_std if anxiety_std is not None else '不可用'}。")
    lines.append("- 研究範圍僅限國道：不推論省道（台9線）或市區道路；國2 因無 M05A eTag 資料不納入推論。")
    lines.append("- 2025 面板僅至 11 月 23 日，未覆蓋全年；指標是該 53 天、3 走廊、96 時段的加權結果。")
    lines.append("- Brier 採多分類 one-hot 與 predict_proba 的四類平方誤差加總（未除以類別數），"
                 "取值範圍 0–2；P90 覆蓋率以真實旅行時間 ≤ P90 預測的比例計算。")
    lines.append("- LightGBM 的非確定性來源已固定 random_state=42，但不同硬體仍可能有微小數值差異。")
    lines.append("- 缺檔的外部來源（尤以 holiday_controls.csv）需由人工補齊後重跑，"
                 "本程式不會以 0 或隨機值填補。")
    (out_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    started = time.perf_counter()
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else config.ROOT
    root = root.resolve()
    out_dir = root / "outputs" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"root = {root}")
    print(f"output = {out_dir}")

    t = time.perf_counter()
    panel, panel_meta = mf.load_panel(root)
    print(f"[1/7] 面板 {panel_meta['panel_rows']} 列 / {panel_meta['panel_days']} 天 "
          f"({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    calendar, calendar_status = mf.load_calendar(root)
    calendar_ok = not calendar.empty
    events = mf.build_events(calendar, calendar_ok)
    weather, weather_status = mf.load_weather(root)
    trends, trend_status, trend_keywords = mf.load_trends(root)
    policy, policy_status, policy_cols = mf.load_policy(root)
    static, source_status = mf.build_static_features(
        panel, events, calendar_ok, weather, weather_status, trends, trend_status, policy)
    print(f"[2/7] 外部來源 calendars={calendar_status}; weather={weather_status}; "
          f"trends={trend_status}; policy={policy_status} ({time.perf_counter() - t:.1f}s)")

    full = panel.copy()
    for col in static.columns:
        if col in full.columns:
            raise ValueError(f"特徵欄位與原始欄位撞名：{col}")
        full[col] = static[col].to_numpy()
    level = mf.level_from_speed(full["speed_kmh"])
    full["severe_flag"] = (level >= 2).astype(int).to_numpy()

    # ---- 嚴格時序切分（絕不使用 train_test_split） ------------------------
    train_mask = full["date"].dt.year.isin((2023, 2024))
    test_mask = full["date"].dt.year == 2025
    train = full.loc[train_mask].reset_index(drop=True)
    test = full.loc[test_mask].reset_index(drop=True)
    train_dates = set(train["date"].unique())
    test_dates = set(test["date"].unique())
    overlap = bool(train_dates & test_dates)
    if overlap or train.empty or test.empty:
        raise RuntimeError("時序切分失敗：train/test 日期重疊或某一邊為空")
    split = {
        "train_years": "2023-2024", "test_years": "2025",
        "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
        "n_train_days": len(train_dates), "n_test_days": len(test_dates),
        "date_overlap": overlap,
    }
    print(f"[3/7] 切分 train={len(train)} 列 / {len(train_dates)} 天；"
          f"test={len(test)} 列 / {len(test_dates)} 天")

    # ---- 歷史同期統計：只由 train 擬合 -----------------------------------
    hist_columns = {
        "hist_mean_speed": "speed_kmh",
        "hist_mean_level": "congestion_level",
        "hist_severe_rate": "severe_flag",
        "hist_mean_heavy_share": "heavy_share",
    }
    hist_train = mf.historical_features(train, train, hist_columns, leave_one_out=True)
    hist_test = mf.historical_features(train, test, hist_columns, leave_one_out=False)
    for col in hist_train.columns:
        train[col] = hist_train[col].to_numpy()
        test[col] = hist_test[col].to_numpy()

    # ---- 決定各組特徵（缺來源直接跳過，不補 0） ---------------------------
    missing: list[dict[str, str]] = []

    def use_feature(feat: str, reason: str) -> bool:
        if feat not in train.columns or feat not in test.columns:
            missing.append({"feature": feat, "reason": reason})
            return False
        if _finite_count(train[feat]) == 0:
            missing.append({"feature": feat, "reason": reason + "；且 train 無有限值"})
            return False
        if _finite_count(test[feat]) == 0:
            missing.append({"feature": feat, "reason": reason + "；且 test 無有限值（不對 test 補值）"})
            return False
        return True

    feats_a = list(A_BASE)
    feats_b = feats_a + list(B_TIME)
    if use_feature("ylp_precip_d1", source_status["ylp_precip_d1"]):
        feats_b.append("ylp_precip_d1")
    feats_c = list(feats_b)
    for feat in C_EXTRA:
        reason = source_status.get(feat, "來源不明")
        if feat == "hist_mean_heavy_share":
            reason = "train heavy_share 歷史同期均值"
        if feat in ("event_total_days", "days_to_event_end"):
            reason = "config.HOLIDAY_EVENTS"
        if use_feature(feat, reason):
            feats_c.append(feat)
    diagnostics = {}
    for feat in feats_c:
        tr_vals = pd.to_numeric(train[feat], errors="coerce").to_numpy(dtype=float)
        te_vals = pd.to_numeric(test[feat], errors="coerce").to_numpy(dtype=float)
        diagnostics[feat] = {
            "train_finite": int(np.isfinite(tr_vals).sum()),
            "test_finite": int(np.isfinite(te_vals).sum()),
            "train_std": float(np.nanstd(tr_vals)) if np.isfinite(tr_vals).any() else None,
        }
    low_variance = [f for f, d in diagnostics.items()
                    if d["train_std"] is not None and d["train_std"] < 1e-4]
    print(f"[4/7] 特徵數 A/B/C = {len(feats_a)}/{len(feats_b)}/{len(feats_c)}；"
          f"跳過 {len(missing)} 項；低變異特徵 {low_variance}")

    y_cls_train = train["congestion_level"].astype(int)
    y_cls_test = test["congestion_level"].astype(int)
    y_reg_train = train["travel_time_min"].astype(float)
    y_reg_test = test["travel_time_min"].astype(float)

    results: dict[str, dict] = {}
    metrics: dict[str, dict[str, float]] = {}
    feature_sets = {"A": feats_a, "B": feats_b, "C": feats_c}
    for name in ("A", "B", "C"):
        t = time.perf_counter()
        feats = feature_sets[name]
        X_train = train[feats].apply(pd.to_numeric, errors="coerce").astype("float32")
        X_test = test[feats].apply(pd.to_numeric, errors="coerce").astype("float32")
        fitted = fit_one(X_train, y_cls_train, y_reg_train, X_test)
        results[name] = fitted
        metrics[name] = compute_metrics(
            y_cls_test.to_numpy(dtype=int), fitted["proba"], fitted["pred"],
            y_reg_test.to_numpy(dtype=float), fitted["p50"], fitted["p90"])
        print(f"[5/7] 模型 {name} 完成 ({time.perf_counter() - t:.1f}s)")

    improvements: dict[str, dict[str, float]] = {}
    for target in ("A", "B"):
        improvements[f"C_vs_{target}"] = {
            key: _pct_change(metrics[target][key], metrics["C"][key], METRIC_DIRECTION[key])
            for key in METRIC_LABELS
        }

    # ---- 輸出 -------------------------------------------------------------
    long_rows = [{"model": model, "metric": key, "value": metrics[model][key]}
                 for model in ("A", "B", "C") for key in METRIC_LABELS]
    pd.DataFrame(long_rows).to_csv(out_dir / "ablation_metrics.csv", index=False,
                                   encoding="utf-8-sig")

    base_pred = pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d"),
        "corridor_id": test["corridor_id"],
        "time_slot_15m": test["time_slot_15m"],
        "y_true_class": y_cls_test.to_numpy(),
    })
    pred_c = results["C"]
    primary = base_pred.copy()
    primary["y_pred_class"] = pred_c["pred"]
    primary["p50"] = pred_c["p50"]
    primary["p90"] = pred_c["p90"]
    for cls in CLASS_LABELS:
        primary[f"prob_class_{cls}"] = pred_c["proba"][:, cls]
    primary.to_csv(out_dir / "predictions_2025.csv", index=False, encoding="utf-8-sig")

    all_rows = []
    for model in ("A", "B", "C"):
        part = base_pred.copy()
        part.insert(0, "model", model)
        part["y_pred_class"] = results[model]["pred"]
        part["p50"] = results[model]["p50"]
        part["p90"] = results[model]["p90"]
        for cls in CLASS_LABELS:
            part[f"prob_class_{cls}"] = results[model]["proba"][:, cls]
        all_rows.append(part)
    pd.concat(all_rows, ignore_index=True).to_csv(
        out_dir / "predictions_2025_all_models.csv", index=False, encoding="utf-8-sig")

    importance_rows = []
    tasks = [("classification", pred_c["clf"].booster_),
             ("regression_p50", pred_c["regressors"][0.5].booster_),
             ("regression_p90", pred_c["regressors"][0.9].booster_)]
    for task, booster in tasks:
        gains = booster.feature_importance(importance_type="gain")
        for feat, gain in zip(booster.feature_name(), gains):
            importance_rows.append({"model": "C", "task": task, "feature": feat,
                                    "gain": float(gain)})
    pd.DataFrame(importance_rows).sort_values(["task", "gain"], ascending=[True, False]).to_csv(
        out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_cls_test.to_numpy(dtype=int), pred_c["pred"], labels=CLASS_LABELS)
    cm_df = pd.DataFrame(cm, columns=[f"pred_{c}" for c in CLASS_LABELS])
    cm_df.insert(0, "true_class", CLASS_LABELS)
    cm_df.to_csv(out_dir / "confusion_matrix_model_C.csv", index=False, encoding="utf-8-sig")

    elapsed = time.perf_counter() - started
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 2),
        "root": str(root),
        "split": split,
        "panel": panel_meta,
        "data_sources": {
            "calendar": calendar_status,
            "weather": weather_status,
            "trends": trend_status,
            "policy": policy_status,
            "trends_keywords": trend_keywords,
            "policy_available_columns": policy_cols,
            "event_count": int(len(events)),
        },
        "feature_lists": feature_sets,
        "feature_diagnostics": diagnostics,
        "missing_features": missing,
        "model_params": {
            "classifier": CLF_PARAMS,
            "regressor_base": REG_PARAMS,
            "regressor_alphas": list(QUANTILES),
        },
        "metrics": metrics,
        "improvements_pct": improvements,
        "leakage_controls": [
            "依日期年份切分 2023–2024 train / 2025 test，未使用任何隨機切分。",
            "當日 speed_kmh、volume_15m、heavy_share、travel_time_min 只作標的，從不作為特徵。",
            "歷史同期平均車速／壅塞率／重車佔比僅由 train groupby 擬合後映射；train 使用 leave-one-out 去掉自身列。",
            "天氣（YLP/TPE）與 Google Trends 只取 D-1，3C 前序週末累積雨量只取嚴格早於 D 的最近週末。",
            "train 與 test 共用完全相同的特徵建構函式，無個別調參或個別填值。",
            "交管分鐘數來自事件前公告的 holiday_controls.csv，而非當日車流觀測；缺失時不建立特徵。",
        ],
        "predictions_file_model": "C",
        "output_files": [
            "ablation_metrics.csv", "ablation_report.md", "predictions_2025.csv",
            "predictions_2025_all_models.csv", "feature_importance.csv",
            "confusion_matrix_model_C.csv", "summary.json",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    build_report(out_dir, root, summary, metrics, improvements, missing)

    print("[6/7] 2025 test 指標（A / B / C）")
    table = pd.DataFrame(
        [[metrics[model][key] for model in ("A", "B", "C")] for key in METRIC_LABELS],
        index=[METRIC_LABELS[key] for key in METRIC_LABELS], columns=["A", "B", "C"])
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("[7/7] C 相對 A/B 改善 (%)")
    imp_table = pd.DataFrame(improvements).T
    print(imp_table.to_string(float_format=lambda v: f"{v:+.2f}"))
    if missing:
        print("跳過特徵：")
        for item in missing:
            print(f"  - {item['feature']}: {item['reason']}")
    print(f"完成，耗時 {elapsed:.1f} 秒；輸出至 {out_dir}")


if __name__ == "__main__":
    main()
