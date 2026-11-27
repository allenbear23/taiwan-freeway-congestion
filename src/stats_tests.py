# -*- coding: utf-8 -*-
"""24 假設的統計檢定、模型增益、跨年穩定性與可行評分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import PERM_REPEATS, RANDOM_STATE, SHAP_SAMPLE_N
from src.features import Observation
from src.utils import LOGGER

# 故事吸睛度：主觀編輯分（非數據），附理由供報告揭露
STORY: dict[str, tuple[int, str]] = {
    "1A": (5, "預報造成尖峰左移，具反直覺且可直接對應交管資訊"),
    "1B": (4, "國5/台9 分流是連假經典故事，但需省道資料"),
    "1C": (4, "搜尋領先指標可做早期預警，敘事直觀"),
    "2A": (5, "高乘載解除瞬間的洩洪效應，時間邊界清楚、易視覺化"),
    "2B": (4, "凌晨免收費壓線搶閘，政策邊界明確"),
    "2C": (3, "匝道封閉虹吸效應局部性强，需門架級資料"),
    "2D": (3, "差別費率轉移需對照平行國道，政策情境較弱"),
    "3A": (5, "『逃離雨天』氣候剪刀差，跨域故事性最高"),
    "3B": (5, "宜蘭下雨轉塞桃竹，反直覺跨區連動"),
    "3C": (4, "掃墓延後釋放，文化+氣候雙重機制"),
    "3D": (3, "降雨容量崩跌屬工程臨界點，但需高解析雨量"),
    "4A": (5, "出國潮抽乾國旅，宏觀視角吸睛"),
    "4B": (4, "送機潮引爆國2，跨運具連動清楚"),
    "4C": (4, "高鐵滿載外溢至國1，跨運具替代明確"),
    "5A": (4, "請假攻略槓桿，貼近民眾生活"),
    "5B": (3, "補班後補償反彈，效應較間接"),
    "5C": (4, "連假越長越早北返，日曆拓撲好懂"),
    "6A": (4, "長途旅次脆弱度，微觀機理扎實"),
    "6B": (3, "重車爬坡干擾，工程意義強但故事性中等"),
    "6C": (5, "崩盤前兆訊號，預警價值高"),
    "7A": (3, "農民曆吉日有趣但效應可能很小"),
    "7B": (4, "演唱會散場衝擊，事件明確"),
    "8A": (4, "回堵衝擊波延遲，空間物理故事完整"),
    "8B": (3, "國5與台9同步崩潰，需省道資料"),
}


def _finite(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _score_sig(p: float, effect: float) -> int:
    if not np.isfinite(p):
        return 1
    if p < 0.001 and effect >= 0.5:
        return 5
    if p < 0.01 and effect >= 0.3:
        return 4
    if p < 0.05:
        return 3
    if p < 0.10:
        return 2
    return 1


def _quantile_score(value: float, edges: tuple[float, ...]) -> int:
    if not np.isfinite(value):
        return 1
    for i, e in enumerate(edges):
        if value <= e:
            return i + 1
    return 5


def evaluate(obs: Observation) -> dict:
    """回傳單一假設的完整評分紀錄。"""
    base = {
        "id": obs.hid, "維度": obs.dimension, "假設": obs.title,
        "X": obs.x_name, "Y": obs.y_name, "類型": obs.kind,
        "所需資料": ", ".join(obs.required), "n": 0, "狀態": "資料不足",
        "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan,
        "welch_t": np.nan, "welch_p": np.nan, "mw_u": np.nan, "mw_p": np.nan,
        "cohens_d": np.nan, "effect": np.nan, "gain": np.nan, "shap_mean_abs": np.nan,
        "stability": np.nan, "顯著性": 1, "模型增益": 1, "穩定度": 1, "故事性": STORY[obs.hid][0],
        "故事理由": STORY[obs.hid][1], "總分": np.nan, "備註": obs.note,
    }
    if not obs.available:
        return base

    df = obs.df[["x", "y", "year", "unit"]].dropna()
    x = _finite(df["x"])
    y = _finite(df["y"])
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        base["n"] = len(x)
        base["備註"] = (base["備註"] + "；變異不足").strip("；")
        return base
    base["n"] = len(x)

    effect = np.nan
    p_main = np.nan
    try:
        if obs.kind == "continuous":
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            base.update(pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp)
            effect = abs(sr) if np.isfinite(sr) else abs(pr)
            p_main = sp if np.isfinite(sp) else pp
        else:
            g1 = y[x == 1] if np.any(x == 1) else y[x > np.median(x)]
            g0 = y[x == 0] if np.any(x == 0) else y[x <= np.median(x)]
            if len(g1) >= 3 and len(g0) >= 3:
                t, tp = stats.ttest_ind(g1, g0, equal_var=False)
                u, up = stats.mannwhitneyu(g1, g0, alternative="two-sided")
                pooled = np.sqrt((np.var(g1, ddof=1) + np.var(g0, ddof=1)) / 2)
                d = (np.mean(g1) - np.mean(g0)) / pooled if pooled > 0 else 0.0
                base.update(welch_t=t, welch_p=tp, mw_u=u, mw_p=up, cohens_d=d)
                effect = abs(d)
                p_main = up if np.isfinite(up) else tp
    except Exception as e:  # noqa: BLE001
        base["備註"] = f"檢定失敗：{e}"

    # --- 模型增益：LightGBM + permutation importance / SHAP ---
    gain = np.nan
    shap_abs = np.nan
    try:
        from lightgbm import LGBMRegressor
        from sklearn.inspection import permutation_importance
        model = LGBMRegressor(
            n_estimators=60, num_leaves=7, max_depth=3, learning_rate=0.1,
            min_child_samples=3, verbosity=-1, random_state=RANDOM_STATE,
        )
        model.fit(x.reshape(-1, 1), y)
        r = permutation_importance(model, x.reshape(-1, 1), y, n_repeats=PERM_REPEATS,
                                  random_state=RANDOM_STATE, scoring="neg_mean_squared_error")
        var_y = np.var(y)
        gain = float(np.mean(r.importances_mean) / var_y) if var_y > 0 else 0.0
        try:
            import shap
            sample = min(SHAP_SAMPLE_N, len(x))
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x[:sample].reshape(-1, 1))
            shap_abs = float(np.mean(np.abs(sv)))
        except Exception as e:  # noqa: BLE001
            LOGGER.debug("SHAP 跳過：%s", e)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("模型增益失敗 %s：%s", obs.hid, e)

    # --- 跨年穩定性：各年效果方向是否一致 ---
    try:
        signs = []
        overall_sign = np.sign(stats.spearmanr(x, y).statistic if obs.kind == "continuous"
                               else (np.mean(g1) - np.mean(g0)))
        for yr, g in df.groupby("year"):
            if len(g) < 4:
                continue
            if obs.kind == "continuous":
                s = stats.spearmanr(g["x"], g["y"]).statistic
            else:
                xx = g["x"].values
                s = np.mean(g["y"].values[xx > np.median(xx)]) - np.mean(g["y"].values[xx <= np.median(xx)])
            if np.isfinite(s) and s != 0:
                signs.append(np.sign(s) == overall_sign)
        stability = float(np.mean(signs)) if signs else 0.5
    except Exception:  # noqa: BLE001
        stability = 0.5

    base.update(
        effect=effect, gain=gain, shap_mean_abs=shap_abs, stability=stability,
        顯著性=_score_sig(p_main, effect),
        模型增益=_quantile_score(gain, (0.01, 0.03, 0.08, 0.15)),
        穩定度=5 if stability >= 0.9 else 4 if stability >= 0.75 else 3 if stability >= 0.6 else 2 if stability >= 0.4 else 1,
        狀態="可用",
    )
    base["總分"] = round(
        0.35 * base["顯著性"] + 0.25 * base["模型增益"] +
        0.20 * base["穩定度"] + 0.20 * base["故事性"], 3,
    )
    return base


def evaluate_all(observations: list[Observation]) -> pd.DataFrame:
    rows = [evaluate(o) for o in observations]
    return pd.DataFrame(rows).sort_values("總分", ascending=False, na_position="last").reset_index(drop=True)
