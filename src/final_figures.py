# -*- coding: utf-8 -*-
"""Step 2 五張中文敘事圖產生器（holiday_traffic_final.ipynb 的圖表來源）。

Input : outputs/phase2/{ablation_metrics.csv, predictions_2025.csv,
        feature_importance.csv, confusion_matrix_model_C.csv, summary.json}
Output: outputs/figures/fig1_*.png ... fig5_*.png（全部 dpi=300）

設計原則
--------
1. 只讀 phase2 的模型輸出，**不產生假資料**。缺檔或欄位不足時印出清楚訊息並略過該圖。
2. 中文字型：首選 Noto Sans TC；必要時嘗試下載並註冊，否則降級 Microsoft JhengHei /
   Noto Sans CJK TC。Notebook 第一格使用同一套下載邏輯。
3. 每張圖都有完整主標題、由資料算出的洞見副標題、軸標籤（含單位）與圖例／色階。

用法
----
    python src/final_figures.py                 # 正式產圖
    python src/final_figures.py --selftest      # [MOCK_TEST_ONLY] 僅在暫存目錄驗證繪圖邏輯
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

try:  # 走廊中文名／等級定義沿用全專案單一設定來源
    from config import CONGESTION_LEVELS, CORRIDORS
except Exception:  # pragma: no cover - 單獨複製腳本時仍能產圖
    CONGESTION_LEVELS = [(80, 0), (60, 1), (40, 2), (0, 3)]
    CORRIDORS = {}

PHASE2 = ROOT / "outputs" / "phase2"
FIG_DIR = ROOT / "outputs" / "figures"
DPI = 300

CLASS_LABELS = {0: "0 順暢", 1: "1 車多", 2: "2 壅塞", 3: "3 嚴重壅塞"}
LEVEL_COLORS = ["#4c9f70", "#e8c34a", "#e88b3b", "#c0392b"]
MODEL_COLORS = {"A": "#7f8c8d", "B": "#e08b3e", "C": "#2878b5"}
# noto-cjk repo 的 Static TC subset（完整的 OTF/TraditionalChinese 檔名為 NotoSansCJKtc-*）
FONT_URLS = {
    "Regular": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/TC/NotoSansTC-Regular.otf",
    "Bold": "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/TC/NotoSansTC-Bold.otf",
}
FONT_CANDIDATES = (
    "Noto Sans TC", "Microsoft JhengHei", "Noto Sans CJK TC",
    "PingFang TC", "Microsoft YaHei",
)


# ---------------------------------------------------------------------------
# 字型（Notebook 第一格使用相同邏輯）
# ---------------------------------------------------------------------------
def _installed_fonts() -> dict[str, set[str]]:
    """回傳 {字型名: {weight, ...}}，用來判斷是否有可用的常規／粗體。"""
    result: dict[str, set[str]] = {}
    try:
        for f in font_manager.fontManager.ttflist:
            result.setdefault(f.name, set()).add(str(f.weight))
    except Exception:  # pragma: no cover
        pass
    return result


def _download_noto(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for weight, url in FONT_URLS.items():
        target = target_dir / f"NotoSansTC-{weight}.otf"
        try:
            if not target.exists():
                with urlopen(url, timeout=60) as response:
                    payload = response.read()
                if payload[:4] not in (b"OTTO", b"\x00\x01\x00\x00"):
                    raise ValueError("下載內容不是有效的 OpenType 字型")
                target.write_bytes(payload)
            font_manager.fontManager.addfont(str(target))
            print(f"[字型] 已註冊 {target}")
        except Exception as exc:  # noqa: BLE001 - 離線時安靜降級
            print(f"[字型] 無法下載 {weight}（{exc}），改用系統中文字型")


def set_chinese_font(download: bool = True) -> str:
    """註冊 Noto Sans TC；失敗時降級到 JhengHei / Noto Sans CJK TC。"""
    if os.environ.get("FINAL_FIGURES_NO_FONT_DOWNLOAD") == "1":
        download = False
    installed = _installed_fonts()
    chosen = next((name for name in FONT_CANDIDATES if name in installed), None)

    # Colab 不一定預裝中文字型；本機的 Noto Sans TC 可變字體常只有 weight=100，
    # 這時補下載 static Regular/Bold，避免標題被渲染成極細字。
    need_download = download and (
        chosen is None
        or (chosen == "Noto Sans TC" and not {"400", "700"}.issubset(installed.get(chosen, set())))
    )
    if need_download:
        system_dir = Path("/usr/share/fonts/truetype/noto")
        target_dir = system_dir if system_dir.parent.is_dir() and os.access(
            str(system_dir.parent), os.W_OK
        ) else Path.home() / ".fonts"
        _download_noto(target_dir)
        installed = _installed_fonts()
        chosen = next((name for name in FONT_CANDIDATES if name in installed), None)

    if chosen:
        plt.rcParams["font.family"] = chosen
        plt.rcParams["font.sans-serif"] = [chosen, *FONT_CANDIDATES, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [*FONT_CANDIDATES, "DejaVu Sans"]
        print("[字型] 警告：找不到中文字型，中文可能顯示為方框")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    return chosen or "DejaVu Sans（中文缺字風險）"


# ---------------------------------------------------------------------------
# 通用工具：的正規化、欄位比對、讀檔錯誤訊息
# ---------------------------------------------------------------------------
def norm(text: object) -> str:
    """把欄位名稱轉成利於比對的小寫英數字（保留中文）。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", str(text).lower())


def find_col(df: pd.DataFrame, aliases: tuple[str, ...], what: str) -> str | None:
    lookup = {norm(c): c for c in df.columns}
    for alias in aliases:
        hit = lookup.get(norm(alias))
        if hit is not None:
            return hit
    return None


def read_csv_strict(path: Path, required: tuple[str, ...] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path.relative_to(ROOT) if ROOT in path.parents else path}："
            "請先執行 phase2_model.py 產生此檔；為避免假資料，本圖略過。"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"讀取 {path.name} 失敗：{exc}") from exc
    if df.empty:
        raise ValueError(f"{path.name} 是空檔，請檢查 phase2_model.py 的輸出。")
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name} 缺少必要欄位 {missing}；現有欄位：{list(df.columns)}"
            )
    return df


def summary_text(subtitle: str) -> None:
    """統一副標題位置，避免與主標題重疊。"""
    fig = plt.gcf()
    fig.text(0.5, 0.932, subtitle, ha="center", va="top", fontsize=10.5, color="#555555")


def finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout(rect=(0.005, 0.015, 0.995, 0.905))
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def model_letter(value: object) -> str:
    hit = re.search(r"\b([ABC])\b", str(value), flags=re.I)
    if hit:
        return hit.group(1).upper()
    hit = re.search(r"([ABC])", str(value), flags=re.I)
    return hit.group(1).upper() if hit else str(value).strip()


def corridor_label(cid: object) -> str:
    key = str(cid).strip()
    return CORRIDORS.get(key, {}).get("name", key)


# ---------------------------------------------------------------------------
# 圖 1：2025 測試連假 時段 × 走廊 壅塞等級熱力圖
# ---------------------------------------------------------------------------
def parse_slot_index(values: pd.Series) -> np.ndarray:
    """把 time_slot_15m 轉成 0–95 的時段索引（相容整數／HH:MM／HHMM／datetime）。"""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        hi = float(numeric.max())
        if hi <= 95.5:
            return np.floor(numeric.to_numpy(dtype=float)).astype(int)
        if hi <= 1440:
            return np.floor(numeric.to_numpy(dtype=float) / 15.0).astype(int)
        if hi <= 86400:
            return np.floor(numeric.to_numpy(dtype=float) / 900.0).astype(int)

    text = values.astype(str).str.strip()
    hit = text.str.extract(r"(?:^|[ T])([0-2]?\d):([0-5]\d)") if numeric.isna().any() else None
    if hit is not None and hit[0].notna().all():
        hours = hit[0].astype(int).to_numpy()
        minutes = hit[1].astype(int).to_numpy()
        return np.clip(hours * 4 + minutes // 15, 0, 95)

    hit = text.str.extract(r"^([0-2]?\d)([0-5]\d)$")
    if hit[0].notna().all():
        return np.clip(hit[0].astype(int).to_numpy() * 4 + hit[1].astype(int).to_numpy() // 15, 0, 95)

    raise ValueError(
        "無法解析 time_slot_15m（支援 0–95 時段索引、分鐘、HH:MM 或含日期時間字串）；"
        f"範例值：{values.head(3).tolist()}"
    )


def load_predictions(path: Path) -> pd.DataFrame:
    df = read_csv_strict(path, ("date", "corridor_id", "time_slot_15m", "y_true_class"))
    out = df.copy()
    out["_class"] = pd.to_numeric(out["y_true_class"], errors="coerce")
    if out["_class"].notna().sum() == 0:
        raise ValueError(f"{path.name} 的 y_true_class 不是數值，無法計算壅塞等級。")
    out["_class"] = out["_class"].clip(0, 3)
    out["_slot"] = parse_slot_index(out["time_slot_15m"])
    out["_hour"] = out["_slot"] // 4
    return out


def fig1_holiday_overview(phase2: Path, output: Path) -> Path:
    pred = load_predictions(phase2 / "predictions_2025.csv")
    counts = pred.groupby(["corridor_id", "_slot"]).size().unstack("_slot")
    levels = pred.groupby(["corridor_id", "_slot"])["_class"].mean().unstack("_slot")

    order = [c for c in CORRIDORS if c in levels.index] + [
        c for c in levels.index if c not in CORRIDORS
    ]
    levels = levels.reindex(index=order, columns=range(96))
    counts = counts.reindex(index=order, columns=range(96)).fillna(0)
    levels = levels.loc[levels.notna().any(axis=1)]
    counts = counts.loc[levels.index]
    if levels.empty:
        raise ValueError("predictions_2025.csv 沒有可用的 15 分鐘觀測，無法繪製熱力圖。")

    # 以每格至少 3 筆觀測篩掉稀疏噪音後找尖峰
    valid = counts >= 3
    masked = levels.where(valid)
    flat = int(np.nanargmax(masked.to_numpy())) if masked.notna().any().any() else int(
        np.nanargmax(levels.to_numpy())
    )
    row, col = np.unravel_index(flat, levels.shape)
    peak_corridor, peak_slot = levels.index[row], int(levels.columns[col])
    cell = pred[(pred["corridor_id"] == peak_corridor) & (pred["_slot"] == peak_slot)]
    peak_level = float(cell["_class"].mean())
    peak_rate = float((cell["_class"] >= 2).mean() * 100)
    hh, mm = divmod(peak_slot * 15, 60)
    next_hh, next_mm = divmod((peak_slot + 1) * 15, 60)
    peak_time = f"{hh:02d}:{mm:02d}–{next_hh:02d}:{next_mm:02d}"

    fig, ax = plt.subplots(figsize=(15, 7.2))
    matrix = levels.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=3, interpolation="nearest")
    image.cmap.set_bad("#eceff1")
    ax.set_xticks(np.arange(0, 96, 8), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_yticks(range(len(levels)), [corridor_label(c) for c in levels.index])
    ax.set_xlabel("當日時段（15 分鐘解析度，標示每 2 小時；時:分）", labelpad=10)
    ax.set_ylabel("測試走廊（國道 eTag 區間）", labelpad=10)
    ax.set_title("2025 測試連假｜走廊 × 時段壅塞等級熱力圖", fontsize=16, fontweight="bold", pad=14)
    summary_text(
        f"尖峰洞見：{corridor_label(peak_corridor)} 於 {peak_time} 平均等級 {peak_level:.2f}、"
        f"等級≥2 比例 {peak_rate:.0f}%（該格 n={len(cell)}）｜全域 {len(pred):,} 筆 15 分鐘觀測"
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.045)
    cbar.set_ticks([0, 1, 2, 3], labels=[CLASS_LABELS[i] for i in range(4)])
    cbar.set_label("平均壅塞等級（0–3，越高越塞）", labelpad=12)
    ax.legend(
        handles=[
            Patch(facecolor="#eceff1", label="灰色：該時段無觀測（不補值）"),
            Patch(facecolor="#c0392b", label="深紅：等級 3 嚴重壅塞"),
        ],
        loc="lower left", bbox_to_anchor=(0.0, -0.20), ncol=2, frameon=False, fontsize=9,
    )
    ax.grid(False)
    return finish(fig, output)


# ---------------------------------------------------------------------------
# 圖 2：模型 A/B/C 五項指標比較
# ---------------------------------------------------------------------------
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "macro_f1": ("macrof1", "f1macro", "macroaveragef1", "macroavgf1", "macromeanf1"),
    "recall_l2l3": (
        "recalll2l3", "l2l3recall", "recalllevel23", "recalllevel2and3", "level23recall",
        "recall23", "l2+3recall", "severerecall", "recallheavy", "recallcongested",
    ),
    "log_loss": ("logloss", "crossentropy", "ce", "negloglikelihood"),
    "brier": ("brierscore", "brierscoreloss", "brier"),
    "p50_mae": ("p50mae", "maep50", "p50maemin", "maemin", "p50maeminutes", "medianmae"),
}
METRIC_INFO = {
    "macro_f1": ("Macro F1（0–1，越高越好）", ".3f", True, 1.0, "無單位"),
    "recall_l2l3": ("等級 2+3 召回率（越高越好）", ".1f", True, 100.0, "%"),
    "log_loss": ("Log Loss（越低越好）", ".4f", False, 1.0, "無單位"),
    "brier": ("Brier Score（越低越好）", ".4f", False, 1.0, "無單位"),
    "p50_mae": ("P50 MAE（越低越好）", ".1f", False, 1.0, "分鐘"),
}


def load_ablation(path: Path) -> pd.DataFrame:
    df = read_csv_strict(path)
    model_col = find_col(df, ("model", "模型", "model_name"), "model")
    metric_col = find_col(df, ("metric", "指標", "metric_name"), "metric")
    value_col = find_col(df, ("value", "數值", "score"), "value")
    if None in (model_col, metric_col, value_col):
        raise ValueError(
            f"{path.name} 需為 long format（欄位 model, metric, value）；現有欄位：{list(df.columns)}"
        )
    out = pd.DataFrame({
        "model": df[model_col].map(model_letter),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
    })
    key = df[metric_col].map(norm)
    out["metric"] = None
    for canonical, aliases in METRIC_ALIASES.items():
        alias_keys = {norm(a) for a in aliases}
        out.loc[key.isin(alias_keys), "metric"] = canonical
    out = out.dropna(subset=["metric", "value"])
    if out.empty:
        raise ValueError(
            f"{path.name} 找不到可辨識的指標；支援 {list(METRIC_ALIASES)}；"
            f"實際 metric 值：{df[metric_col].dropna().unique().tolist()[:20]}"
        )
    return out.groupby(["model", "metric"], as_index=False)["value"].mean()


def pct_change(new: float, base: float, higher_better: bool) -> float | None:
    if not np.isfinite(new) or not np.isfinite(base) or abs(base) < 1e-12:
        return None
    return ((new - base) / abs(base) * 100) if higher_better else ((base - new) / abs(base) * 100)


def fig2_ablation_comparison(phase2: Path, output: Path) -> Path:
    abl = load_ablation(phase2 / "ablation_metrics.csv")
    metrics = [m for m in METRIC_INFO if m in set(abl["metric"])]
    if not metrics:
        raise ValueError("ablation_metrics.csv 沒有任何可比較的指標。")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    axes = axes.ravel()
    all_changes: dict[str, list[float]] = {"A": [], "B": []}
    for ax, metric in zip(axes, metrics):
        label, fmt, higher, scale, unit = METRIC_INFO[metric]
        sub = abl[abl["metric"] == metric].set_index("model")["value"]
        if metric == "recall_l2l3" and len(sub) and float(sub.max()) <= 1.5:
            sub = sub * 100  # 0–1 表示時轉成百分比顯示
        models = [m for m in ("A", "B", "C") if m in sub.index]
        values = [float(sub[m]) for m in models]
        ax.barh(range(len(models)), values, color=[MODEL_COLORS.get(m, "#95a5a6") for m in models], height=0.62)
        ax.set_yticks(range(len(models)), [f"模型 {m}" for m in models])
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(value, i, f" {value:{fmt}}", va="center", fontsize=9.5)
        if "C" in sub.index:
            notes = []
            for base in ("A", "B"):
                if base not in sub.index:
                    continue
                change = pct_change(float(sub["C"]), float(sub[base]), higher)
                if change is not None and np.isfinite(change):
                    all_changes[base].append(change)
                    notes.append(f"C vs {base} {change:+.1f}%")
            if notes:
                ax.text(0.99, 0.04, "｜".join(notes), transform=ax.transAxes, ha="right",
                        fontsize=9, color="#2c3e50",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f4f7fa", ec="#cbd5e1", alpha=0.95))
        ax.set_xlabel(f"{label}（{unit}）")
        ax.set_title(metric.replace("_", " "), fontsize=11, color="#34495e")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.legend(
        handles=[Patch(facecolor=MODEL_COLORS[m], label=f"模型 {m}") for m in ("A", "B", "C")],
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005),
    )
    avg_a = np.mean(all_changes["A"]) if all_changes["A"] else float("nan")
    avg_b = np.mean(all_changes["B"]) if all_changes["B"] else float("nan")
    avg_text = (
        f"五項平均改善：C vs A {avg_a:+.1f}%、C vs B {avg_b:+.1f}%"
        if np.isfinite(avg_a) and np.isfinite(avg_b)
        else "部分模型或指標缺席，改善百分比以圖內標註為準"
    )
    fig.suptitle("模型 A／B／C｜消融設計的五項指標比較", fontsize=18, fontweight="bold", y=0.98)
    summary_text(
        f"{avg_text}｜F1 與召回越高越好，Log Loss／Brier／MAE 越低越好（改善均以百分比表示）"
    )
    return finish(fig, output)


# ---------------------------------------------------------------------------
# 圖 3：模型 C 四等級混淆矩陣
# ---------------------------------------------------------------------------
def load_confusion(phase2: Path) -> tuple[np.ndarray, str]:
    path = phase2 / "confusion_matrix_model_C.csv"
    if not path.exists():
        pred_path = phase2 / "predictions_2025.csv"
        if pred_path.exists():  # 真實預測檔可計算，仍非假資料
            pred = read_csv_strict(pred_path)
            t = find_col(pred, ("y_true_class", "true_class", "ytrue", "y_true"), "y_true_class")
            p = find_col(pred, ("y_pred_class", "pred_class", "ypred", "y_pred"), "y_pred_class")
            if t and p:
                y_true = pd.to_numeric(pred[t], errors="coerce")
                y_pred = pd.to_numeric(pred[p], errors="coerce")
                mask = y_true.notna() & y_pred.notna()
                matrix = np.zeros((4, 4), dtype=int)
                np.add.at(matrix, (y_true[mask].astype(int).clip(0, 3), y_pred[mask].astype(int).clip(0, 3)), 1)
                return matrix, "由 predictions_2025.csv 的 y_true/y_pred_class 直接計算"
        raise FileNotFoundError(
            f"缺少 {path.name}（且無法從 predictions_2025.csv 計算）："
            "請先執行 phase2_model.py；為避免假資料，本圖略過。"
        )
    df = pd.read_csv(path)
    t_col = find_col(df, ("y_true", "true", "actual", "真實"), "y_true")
    p_col = find_col(df, ("y_pred", "pred", "predicted", "預測"), "y_pred")
    n_col = find_col(df, ("count", "n", "value", "筆數"), "count")
    if t_col and p_col and n_col:
        matrix = np.zeros((4, 4), dtype=float)
        t = pd.to_numeric(df[t_col], errors="coerce")
        p = pd.to_numeric(df[p_col], errors="coerce")
        n = pd.to_numeric(df[n_col], errors="coerce")
        mask = t.notna() & p.notna() & n.notna()
        np.add.at(matrix, (t[mask].astype(int).clip(0, 3), p[mask].astype(int).clip(0, 3)), n[mask])
        return matrix, path.name
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape == (4, 4):
        return numeric.to_numpy(dtype=float), path.name
    # crosstab 常見格式：第一欄是列標籤（true_class / y_true / '' / ...），後面 4 欄是 prediction class
    # long format 已於上方處理，因此這裡的 5 欄表視為「列標籤 + 4 個預測等級」。
    if df.shape[1] == 5 and numeric.shape[1] in (4, 5):
        return numeric.iloc[:4, -4:].to_numpy(dtype=float), path.name
    if numeric.shape[1] == 4 and numeric.shape[0] >= 4:
        return numeric.iloc[:4].to_numpy(dtype=float), path.name
    raise ValueError(
        f"{path.name} 格式無法辨識：需為 4×4 矩陣，或 long format（y_true, y_pred, count）；"
        f"實際 shape={df.shape}，欄位={list(df.columns)}"
    )


def fig3_confusion_matrix_modelC(phase2: Path, output: Path) -> Path:
    matrix, source = load_confusion(phase2)
    total = float(np.nansum(matrix))
    if total <= 0:
        raise ValueError("混淆矩陣總樣本數為 0，無法計算準確率。")
    accuracy = float(np.nansum(np.diag(matrix)) / total)
    denom = matrix[2:].sum()
    recall_l2l3 = float(matrix[2:, 2:].sum() / denom) if denom > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(9, 7.6))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    for i in range(4):
        for j in range(4):
            value = matrix[i, j]
            row_total = matrix[i].sum()
            pct = value / row_total * 100 if row_total > 0 else 0.0
            ax.text(j, i, f"{value:,.0f}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=10.5, color="white" if value > matrix.max() * 0.55 else "#1f2d3d")
    ax.set_xticks(range(4), [CLASS_LABELS[i] for i in range(4)])
    ax.set_yticks(range(4), [CLASS_LABELS[i] for i in range(4)])
    ax.set_xlabel("模型 C 預測等級（0–3）", labelpad=10)
    ax.set_ylabel("真實壅塞等級（0–3）", labelpad=10)
    ax.set_title("模型 C｜四等級壅塞混淆矩陣", fontsize=17, fontweight="bold", pad=14)
    summary_text(
        f"整體準確率 {accuracy*100:.1f}%｜等級 2+3 召回率 {recall_l2l3*100:.1f}%｜"
        f"n={total:,.0f} 筆｜來源：{source}"
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("樣本數（筆；顏色越深代表越多）", labelpad=12)
    ax.legend(
        handles=[
            Patch(facecolor="#9ecae1", label="色階：格子樣本數（筆）"),
            Patch(facecolor="white", edgecolor="#9ecae1", label="格內下排：該真實等級的 row %"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=9,
    )
    ax.grid(False)
    return finish(fig, output)


# ---------------------------------------------------------------------------
# 圖 4：P50 / P90 旅行時間區間 vs 真實值
# ---------------------------------------------------------------------------
TRUE_TT_ALIASES = (
    "y_true_travel_min", "true_travel_min", "actual_travel_min", "travel_time_true",
    "y_true_min", "y_true_tt", "tt_true", "y_travel_time", "travel_min_true",
    "true_travel_time", "y_true_travel_time", "actual_minutes", "true_min",
)


def fig4_travel_time_quantile(phase2: Path, output: Path) -> Path:
    pred = load_predictions(phase2 / "predictions_2025.csv")
    p_col = find_col(pred, ("p50", "P50", "median", "q50"), "p50") or "p50"
    q_col = find_col(pred, ("p90", "P90", "q90"), "p90") or "p90"
    for col in (p_col, q_col):
        if col not in pred.columns:
            raise ValueError(
                f"predictions_2025.csv 缺少 {col} 欄位；現有欄位：{list(pred.columns)}"
            )
    p50 = pd.to_numeric(pred[p_col], errors="coerce")
    p90 = pd.to_numeric(pred[q_col], errors="coerce")
    t_col = find_col(pred, TRUE_TT_ALIASES, "真實旅行時間")
    true = pd.to_numeric(pred[t_col], errors="coerce") if t_col else pd.Series(np.nan, index=pred.index)
    true_source = t_col or ""

    # predictions 只帶 class 與 p50/p90 時，以真實走廊面板補回旅行時間標的（同一批原始資料，非模擬值）
    if true.notna().sum() == 0:
        panel = ROOT / "data" / "processed" / "corridor_15m.parquet"
        if panel.exists():
            try:
                base = pd.read_parquet(
                    panel, columns=["date", "corridor_id", "time_slot_15m", "travel_time_min"]
                )
                base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
                merged = pred[["date", "corridor_id", "time_slot_15m"]].reset_index(drop=True).merge(
                    base, on=["date", "corridor_id", "time_slot_15m"], how="left", sort=False
                )
                joined = pd.to_numeric(merged["travel_time_min"], errors="coerce")
                if joined.notna().any():
                    true = joined
                    true_source = "travel_time_min（corridor_15m.parquet 真實值，依 date×走廊×時段連接）"
            except Exception as exc:  # noqa: BLE001
                print(f"[圖4] 無法連接真實旅行時間：{exc}")

    seconds_hint = bool(t_col and "sec" in norm(t_col)) or float(p50.median(skipna=True) or 0) > 500
    unit_note = ""
    if seconds_hint:
        p50, p90, true = p50 / 60.0, p90 / 60.0, true / 60.0
        unit_note = "（原始值判為秒，已換算為分鐘）"
    work = pd.DataFrame({
        "hour": pred["_hour"], "p50": p50, "p90": p90, "true": true,
        "date": pred["date"], "corridor": pred["corridor_id"],
    })
    work = work.dropna(subset=["p50", "p90"])
    if work.empty:
        raise ValueError("p50 / p90 欄位沒有有效數值，無法繪製區間圖。")

    has_true = work["true"].notna().any()
    if has_true:
        valid = work.dropna(subset=["true"])
        coverage = float((valid["true"] <= valid["p90"]).mean() * 100)
        mae = float((valid["p50"] - valid["true"]).abs().mean())
        margin = valid["p90"] - valid["true"]
        coverage_note = f"P90 覆蓋率 {coverage:.1f}%（真實值 ≤ P90）｜P50 MAE {mae:.1f} 分鐘"
        n_note = f"n={len(valid):,} 筆｜真實值來源：{true_source}"
    else:
        # 沒有真實值時，僅引用 ablation_metrics 已算好的指標，不自行推估
        mae_path = phase2 / "ablation_metrics.csv"
        mae_note, coverage_note2 = "P50 MAE —", "P90 覆蓋率 —"
        if mae_path.exists():
            try:
                raw = pd.read_csv(mae_path)
                mcol = find_col(raw, ("metric", "指標"), "metric")
                vcol = find_col(raw, ("value", "數值"), "value")
                looked = {norm(r[mcol]): float(r[vcol]) for _, r in raw.iterrows()}
                if "p50mae" in looked:
                    mae_note = f"P50 MAE {looked['p50mae']:.1f} 分鐘"
                if "p90coverage" in looked:
                    coverage_note2 = f"P90 覆蓋率 {looked['p90coverage']*100:.1f}%"
            except Exception:  # noqa: BLE001
                pass
        coverage_note = f"{coverage_note2}｜{mae_note}（取自 ablation_metrics.csv）"
        n_note = "缺少真實旅行時間欄位：請提供 y_true_travel_min 或 data/processed/corridor_15m.parquet"
        margin = work["p90"] - work["p50"]

    hourly = work.groupby("hour").agg(p50=("p50", "mean"), p90=("p90", "mean"),
                                      true=("true", "mean")).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), gridspec_kw={"width_ratios": [1.55, 1]})
    ax1.plot(hourly["hour"], hourly["p50"], color="#2878b5", lw=2.2, marker="o", ms=4, label="P50 中位數預測")
    ax1.plot(hourly["hour"], hourly["p90"], color="#c0392b", lw=2.0, ls="--", marker="s", ms=4, label="P90 上限預測")
    ax1.fill_between(hourly["hour"], hourly["p50"], hourly["p90"], color="#9ecae1", alpha=0.55,
                     label="P50–P90 區間（80% 區間）")
    if has_true:
        ax1.scatter(hourly["hour"], hourly["true"], color="#2c3e50", s=38, zorder=5,
                    label="真實旅行時間（每小時平均）")
    ax1.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax1.set_xlabel("當日小時（時，0–23；15 分鐘資料按時聚合）")
    ax1.set_ylabel("旅行時間（分鐘）")
    ax1.set_title("(a) 時段聚合：區間預測 vs 真實值", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    covered = margin[margin <= 0]
    missed = margin[margin > 0]
    ax2.hist(covered, bins=30, color="#4c9f70", alpha=0.85,
             label="P90 覆蓋（真實值 ≤ P90）" if has_true else "P90 − P50 ≤ 0（區間異常）")
    ax2.hist(missed, bins=30, color="#c0392b", alpha=0.75,
             label="P90 未覆蓋（真實值 > P90）" if has_true else "P90 − P50（正常區間寬度）")
    ax2.axvline(0, color="#34495e", lw=1.4, ls=":")
    ax2.set_xlabel("P90 − 真實旅行時間（分鐘；<0 代表覆蓋）" if has_true else "P90 − P50（分鐘）")
    ax2.set_ylabel("觀測筆數（筆）")
    ax2.set_title("(b) 區間誤差分布", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.25)

    fig.suptitle("模型 C｜旅行時間 P50 / P90 區間與真實值對照", fontsize=17, fontweight="bold", y=0.98)
    summary_text(f"{coverage_note}｜{n_note}{unit_note}｜陽光單一走廊聚合，非逐筆遮蔽")
    fig.text(0.5, 0.018,
             "說明：P90 覆蓋率＝真實旅行時間不高於 P90 的比例；MAE＝|P50 − 真實值| 平均值。"
             "資料不足時不補任何模擬值。",
             ha="center", fontsize=8.5, color="#7f8c8d")
    return finish(fig, output)


# ---------------------------------------------------------------------------
# 圖 5：模型 C 特徵重要性（含黃金三角標註）
# ---------------------------------------------------------------------------
GOLDEN_RULES = {
    "5C 長連假提早北返分流": (
        "5c", "earlyreturn", "returnearly", "提前北返", "北返", "回程",
        "daystoend", "daystoeventend", "eventdayidx", "eventdayindex",
    ),
    "5A 請假攻略槓桿": (
        "5a", "leaveleverage", "leavestrategy", "makeupweek", "bridgeday",
        "skipwork", "請假", "補班",
    ),
    "3A 天氣剪刀差（逃離雨天）": (
        "3a", "scissors", "剪刀差", "weathergradient", "raingradient",
        "weathercontrast", "precipdiff", "tempdiff", "weatherdiff",
        "天氣剪刀差", "降雨差", "跨區氣候",
    ),
}


def golden_tag(feature: str) -> str | None:
    key = norm(feature)
    for tag, keywords in GOLDEN_RULES.items():
        if any(norm(kw) in key for kw in keywords):
            return tag
    return None


def fig5_feature_importance(phase2: Path, output: Path) -> Path:
    df = read_csv_strict(phase2 / "feature_importance.csv")
    feat = find_col(df, ("feature", "name", "variable", "column", "特徵"), "feature")
    candidates = [c for c in df.columns if c != feat and pd.api.types.is_numeric_dtype(df[c])]
    value = find_col(
        df,
        ("permutation_importance", "importance", "gain", "mean_drop", "f1_drop", "score", "value", "split"),
        "importance",
    )
    if value not in candidates:
        value = candidates[0] if candidates else None
    if feat is None or value is None:
        raise ValueError(
            f"{feature_importance_name(phase2)} 需要特徵欄與重要度欄；現有欄位：{list(df.columns)}"
        )

    plot = df[[feat, value]].copy()
    plot[value] = pd.to_numeric(plot[value], errors="coerce")
    plot = plot.dropna().sort_values(value, ascending=False).head(18).iloc[::-1]
    if plot.empty:
        raise ValueError("feature_importance.csv 沒有有效的重要度數值。")

    tags = [golden_tag(str(x)) for x in plot[feat]]
    golden_color, other_color = "#d98324", "#3d6e9f"
    colors = [golden_color if tag else other_color for tag in tags]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(range(len(plot)), plot[value].to_numpy(), color=colors, height=0.7)
    ax.set_yticks(range(len(plot)), [str(x) for x in plot[feat]])
    for bar, val in zip(bars, plot[value]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:.4g}", va="center", fontsize=9)
    unit = "無單位（gain 累加量）" if "gain" in norm(value) else (
        "模型表現下降量" if "permutation" in norm(value) or "drop" in norm(value) else "模型歸因分數"
    )
    ax.set_xlabel(f"{value}（{unit}）", labelpad=10)
    ax.set_ylabel("模型特徵（依重要度排序，前 18 名）", labelpad=10)
    ax.set_title("模型 C｜特徵重要性與黃金三角貢獻", fontsize=17, fontweight="bold", pad=14)
    hits = sorted({tag for tag in tags if tag})
    subtitle = (
        "黃金三角入選：" + "、".join(hits) + f"（{len(hits)}/3）"
        if hits else "黃金三角特徵未以關鍵詞命中：請檢查特徵命名是否使用 5A / 5C / 3A 或中文假設名"
    )
    summary_text(f"{subtitle}｜第 1 名：{plot[feat].iloc[-1]}（{plot[value].iloc[-1]:.4g}）")
    ax.legend(
        handles=[
            Patch(facecolor=golden_color, label="黃金三角特徵（5A／5C／3A）"),
            Patch(facecolor=other_color, label="其他進入模型的特徵"),
        ],
        loc="lower right", frameon=False, fontsize=9,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    return finish(fig, output)


def feature_importance_name(phase2: Path) -> str:
    return (phase2 / "feature_importance.csv").name


# ---------------------------------------------------------------------------
# [MOCK_TEST_ONLY] 仅验证绘图逻辑；不会写入 outputs/figures
# ---------------------------------------------------------------------------
def selftest() -> int:
    import shutil

    temp = Path(tempfile.mkdtemp(prefix="final_figures_selftest_"))
    try:
        src = temp / "phase2"
        dst = temp / "figures"
        src.mkdir()
        rng = np.random.default_rng(42)
        n = 480
        corridors = ["N5_SB", "N5_NB", "N1_SB"]
        cls = rng.integers(0, 4, n)
        p50 = 18 + cls * 7 + rng.normal(0, 2, n)
        pred = pd.DataFrame({
            "date": rng.choice(["2025-04-03", "2025-04-04", "2025-10-04"], n),
            "corridor_id": rng.choice(corridors, n),
            "time_slot_15m": rng.integers(0, 96, n),
            "y_true_class": cls,
            "y_pred_class": np.clip(cls + rng.integers(-1, 2, n), 0, 3),
            "p50": p50,
            "p90": p50 + rng.uniform(4, 14, n),
            "y_true_travel_min": p50 + rng.normal(0, 5, n),
        })
        pred.to_csv(src / "predictions_2025.csv", index=False)
        rows = []
        for model, base in (("A", 1.0), ("B", 1.1), ("C", 1.2)):
            for metric, value in (("macro_f1", 0.62 * base), ("recall_l2_l3", 0.55 * base),
                                  ("log_loss", 0.81 / base), ("brier", 0.17 / base),
                                  ("p50_mae", 9.4 / base)):
                rows.append({"model": model, "metric": metric, "value": value})
        pd.DataFrame(rows).to_csv(src / "ablation_metrics.csv", index=False)
        pd.DataFrame({
            "feature": ["rain_gradient_3A", "days_to_end_5C", "leave_leverage_5A",
                        "speed_prev", "slot_sin", "holiday_flag", "temp_c", "traffic_volume"],
            "importance": [0.23, 0.19, 0.15, 0.12, 0.09, 0.08, 0.05, 0.04],
        }).to_csv(src / "feature_importance.csv", index=False)
        pd.crosstab(pred["y_true_class"], pred["y_pred_class"]).reindex(
            index=range(4), columns=range(4), fill_value=0
        ).to_csv(src / "confusion_matrix_model_C.csv")
        (src / "summary.json").write_text("{}", encoding="utf-8")

        print("[MOCK_TEST_ONLY] 以下數字僅驗證繪圖流程，絕不寫入 outputs/figures")
        results = generate(src, dst)
        for path in results:
            print(f"[selftest] {path.name} {path.stat().st_size:,} bytes")
        return 0
    finally:
        shutil.rmtree(temp, ignore_errors=True)


BUILDERS = (
    fig1_holiday_overview,
    fig2_ablation_comparison,
    fig3_confusion_matrix_modelC,
    fig4_travel_time_quantile,
    fig5_feature_importance,
)
OUTPUT_NAMES = (
    "fig1_holiday_overview.png",
    "fig2_ablation_comparison.png",
    "fig3_confusion_matrix_modelC.png",
    "fig4_travel_time_quantile.png",
    "fig5_feature_importance.png",
)


def generate(phase2: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for builder, name in zip(BUILDERS, OUTPUT_NAMES):
        try:
            made.append(builder(phase2, output / name))
            print(f"[完成] {name}")
        except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
            print(f"[略過 {name}] {exc}")
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="產生 Step 2 五張中文敘事圖（dpi=300）")
    parser.add_argument("--input-dir", type=Path, default=PHASE2, help="phase2 輸出目錄")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR, help="圖片輸出目錄")
    parser.add_argument("--selftest", action="store_true", help="[MOCK_TEST_ONLY] 暫存目錄驗證繪圖")
    args = parser.parse_args(argv)

    set_chinese_font()
    if args.selftest:
        return selftest()

    required = ("ablation_metrics.csv", "predictions_2025.csv", "feature_importance.csv",
                "confusion_matrix_model_C.csv")
    missing = [name for name in required if not (args.input_dir / name).exists()]
    if missing:
        print(f"[final_figures] 找不到 phase2 輸出：{', '.join(missing)}")
        print("[final_figures] 請先執行 python phase2_model.py；為避免假資料，本次不產生圖表。")
        return 2
    made = generate(args.input_dir, args.output_dir)
    print(f"[final_figures] 完成 {len(made)}/{len(OUTPUT_NAMES)} 張圖，輸出目錄：{args.output_dir}")
    return 0 if len(made) == len(OUTPUT_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
