# -*- coding: utf-8 -*-
"""Step 1 候選探索圖（Step 2 的 5 張報告圖另做）。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import EDA_FIGS
from src.features import Observation
from src.utils import LOGGER


def setup_chinese_font() -> str:
    candidates = ["Noto Sans TC", "Microsoft JhengHei", "Microsoft YaHei", "PingFang TC"]
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans（中文字可能缺字）"


def plot_candidate(obs: Observation, result: dict) -> str | None:
    if not obs.available or not result.get("狀態") == "可用":
        return None
    df = obs.df[["x", "y", "year"]].dropna()
    x = df["x"].astype(float).values
    y = df["y"].astype(float).values
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    if obs.kind == "continuous" and len(x) >= 3:
        ax.scatter(x, y, s=18, alpha=0.55, c=df["year"].astype(int), cmap="viridis")
        try:
            slope, intercept, r, _, _ = stats.linregress(x, y)
            grid = np.linspace(np.nanmin(x), np.nanmax(x), 50)
            ax.plot(grid, slope * grid + intercept, color="#c0392b", lw=1.8,
                    label=f"擬合線 r={r:.3f}")
            ax.legend(fontsize=8)
        except Exception:  # noqa: BLE001
            pass
        ax.set_xlabel(obs.x_name)
    else:
        med = np.median(x)
        g1 = y[x > med]
        g0 = y[x <= med]
        try:
            ax.boxplot([g0, g1], tick_labels=["低/無", "高/有"], widths=0.5, patch_artist=True)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot([g0, g1], labels=["低/無", "高/有"], widths=0.5, patch_artist=True)
        ax.set_xlabel(f"{obs.x_name}（分組）")
    ax.set_ylabel(obs.y_name)
    ax.set_title(f"{obs.hid} {obs.title}｜n={len(x)}｜effect={result.get('effect', float('nan')):.3f}\n"
                 f"p={result.get('pearson_p') if obs.kind == 'continuous' else result.get('welch_p'):.3g}"
                 f"｜字型：{font}", fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = EDA_FIGS / f"{obs.hid}_{obs.title.replace('/', '-')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.debug("產出候選圖 %s", path.name)
    return str(path.relative_to(EDA_FIGS.parent.parent))
