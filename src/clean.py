# -*- coding: utf-8 -*-
"""零車速清洗、5→15 分鐘聚合、里程加權走廊面板。

鐵則：
  1. TrafficVolume == 0 且 SpaceMeanSpeed == 0 -> 視為缺失，不得當紫爆。
  2. 先以相鄰時段線性插值（最多 INTERPOLATION_LIMIT 個 5 分鐘時階），
     仍缺失則以 FREE_FLOW_SPEED 補齊，並逐項記錄筆數。
  3. 小客車(31) 算主車速；重車(42,5) 只算混流佔比。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from config import (
    CORRIDORS, FREE_FLOW_SPEED, HEAVY_TYPES, INTERPOLATION_LIMIT,
    PASSENGER_TYPE, SLOT_MINUTES, SLOTS_PER_DAY, ZERO_ZERO_AS_MISSING,
)
from src.utils import LOGGER, congestion_level, corridor_section_km, match_corridor, parse_gantry_id


class CleaningAudit:
    def __init__(self) -> None:
        self.raw_rows = 0
        self.zero_zero_rows = 0
        self.interpolated_rows = 0
        self.freeflow_filled_rows = 0
        self.section_rows_dropped = 0
        self.inactive_section_rows_dropped = 0
        self.by_corridor: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    def to_dict(self) -> dict:
        return {
            "raw_rows": self.raw_rows,
            "zero_zero_rows": self.zero_zero_rows,
            "interpolated_rows": self.interpolated_rows,
            "freeflow_filled_rows": self.freeflow_filled_rows,
            "section_rows_dropped": self.section_rows_dropped,
            "inactive_section_rows_dropped": self.inactive_section_rows_dropped,
            "by_corridor": {k: dict(v) for k, v in self.by_corridor.items()},
        }


def clean_speeds(df: pd.DataFrame, audit: CleaningAudit) -> pd.DataFrame:
    """在 5 分鐘尺度做缺失標記 → 插值 → 自由流補值。"""
    out = df.copy()
    audit.raw_rows += len(out)
    zero_mask = (out["TrafficVolume"].fillna(0) == 0) & (out["SpaceMeanSpeed"].fillna(0) == 0)
    out["_was_zero_zero"] = zero_mask
    if ZERO_ZERO_AS_MISSING:
        audit.zero_zero_rows += int(zero_mask.sum())
        out.loc[zero_mask, "SpaceMeanSpeed"] = np.nan

    out = out.sort_values(["GantryFrom", "GantryTo", "VehicleType", "TimeInterval"])
    key = ["GantryFrom", "GantryTo", "VehicleType"]

    def _fill(g: pd.DataFrame) -> pd.DataFrame:
        s = g["SpaceMeanSpeed"]
        missing = s.isna()
        s2 = s.interpolate(method="linear", limit=INTERPOLATION_LIMIT, limit_direction="both")
        g = g.copy()
        g["_interp_flag"] = missing & s2.notna()
        g["_ff_flag"] = s2.isna()
        g["SpaceMeanSpeed"] = s2.fillna(FREE_FLOW_SPEED)
        return g

    # include_groups=False 讓分組鍵不重複帶入；reset_index 會把鍵從 MultiIndex 還原。
    out = out.groupby(key, group_keys=True, sort=False).apply(_fill, include_groups=False)
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()
    audit.interpolated_rows += int(out["_interp_flag"].sum()) if "_interp_flag" in out else 0
    audit.freeflow_filled_rows += int(out["_ff_flag"].sum()) if "_ff_flag" in out else 0
    return out


def _add_geometry(df: pd.DataFrame, overrides: dict[str, float] | None) -> pd.DataFrame:
    out = df.copy()
    out["corridor_id"] = [match_corridor(a) for a in out["GantryFrom"]]
    to_corridor = [match_corridor(b) for b in out["GantryTo"]]
    out.loc[[a != b for a, b in zip(out["corridor_id"], to_corridor)], "corridor_id"] = None
    out["dist_km"] = [
        corridor_section_km(a, b, overrides) for a, b in zip(out["GantryFrom"], out["GantryTo"])
    ]
    gf = out["GantryFrom"].map(parse_gantry_id)
    out["km_from"] = [g["km"] if g else np.nan for g in gf]
    return out


def aggregate(df: pd.DataFrame, overrides: dict[str, float] | None = None,
              audit: CleaningAudit | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """回傳 (走廊 15 分面板, 區間 15 分面板, 區間 5 分面板, 清洗稽核)。"""
    audit = audit or CleaningAudit()
    cleaned = clean_speeds(df, audit)
    # 全日零流量的門架區間視為「未啟用」，不納入走廊加權（否則會被 90km/h 灌入假自由流）
    vol_sum = cleaned.groupby(["GantryFrom", "GantryTo", "VehicleType"])["TrafficVolume"].transform("sum")
    inactive = vol_sum.fillna(0) == 0
    audit.inactive_section_rows_dropped += int(inactive.sum())
    cleaned = cleaned[~inactive].copy()
    geo = _add_geometry(cleaned, overrides)
    before = len(geo)
    geo = geo[geo["corridor_id"].notna() & geo["dist_km"].notna()].copy()
    audit.section_rows_dropped += before - len(geo)
    if geo.empty:
        empty = pd.DataFrame(columns=[
            "date", "time_slot_15m", "corridor_id", "speed_kmh", "volume_15m",
            "travel_time_min", "heavy_share", "congestion_level", "n_sections",
        ])
        five_empty = pd.DataFrame(columns=[
            "date", "time_slot_5m", "corridor_id", "GantryFrom", "GantryTo",
            "km_from", "speed_kmh", "volume_15m",
        ])
        return empty, empty, five_empty, audit.to_dict()

    geo["date"] = geo["TimeInterval"].dt.normalize()
    geo["time_slot_15m"] = geo["TimeInterval"].dt.hour * 4 + geo["TimeInterval"].dt.minute // 15

    # --- 區間 x 15 分鐘（小客車）---
    passenger = geo[geo["VehicleType"] == PASSENGER_TYPE].copy()
    passenger = passenger.copy()
    passenger["_wv"] = passenger["SpaceMeanSpeed"] * passenger["TrafficVolume"].clip(lower=0)
    passenger["_plain"] = passenger["SpaceMeanSpeed"]
    sec = passenger.groupby(
        ["date", "time_slot_15m", "corridor_id", "GantryFrom", "GantryTo", "dist_km", "km_from"],
        as_index=False,
    ).agg(
        speed_plain=("_plain", "mean"),
        weighted_speed=("_wv", "sum"),
        volume_15m=("TrafficVolume", "sum"),
        zero_zero_count=("SpaceMeanSpeed", "size"),
    )
    sec["speed_kmh"] = np.where(
        sec["volume_15m"] > 0, sec["weighted_speed"] / sec["volume_15m"], sec["speed_plain"]
    )
    sec["travel_time_min"] = np.where(
        sec["speed_kmh"] > 0, sec["dist_km"] / sec["speed_kmh"] * 60,
        sec["dist_km"] / FREE_FLOW_SPEED * 60,
    )

    # --- 走廊 x 15 分鐘：里程加權車速 + 區間旅行時間加總 ---
    sec["weighted_speed"] = sec["speed_kmh"] * sec["dist_km"]
    corridor = sec.groupby(["date", "time_slot_15m", "corridor_id"], as_index=False).agg(
        weighted_speed=("weighted_speed", "sum"),
        total_dist=("dist_km", "sum"),
        travel_time_min=("travel_time_min", "sum"),
        volume_mean=("volume_15m", "mean"),
        n_sections=("GantryFrom", "nunique"),
    )
    corridor["speed_kmh"] = corridor["weighted_speed"] / corridor["total_dist"]
    corridor["volume_15m"] = corridor["volume_mean"]

    # 混流佔比（重車 / 全部車種），只用 42、5
    mix = geo.groupby(["date", "time_slot_15m", "corridor_id", "VehicleType"], as_index=False).agg(
        v=("TrafficVolume", "sum"),
    )
    mix["is_heavy"] = mix["VehicleType"].isin(HEAVY_TYPES)
    mix["is_pass"] = mix["VehicleType"] == PASSENGER_TYPE
    heavy = mix[mix["is_heavy"]].groupby(["date", "time_slot_15m", "corridor_id"])["v"].sum()
    total = mix[mix["is_pass"]].groupby(["date", "time_slot_15m", "corridor_id"])["v"].sum()
    heavy_share = (heavy / total.replace(0, np.nan)).rename("heavy_share").reset_index()
    corridor = corridor.merge(heavy_share, on=["date", "time_slot_15m", "corridor_id"], how="left")
    corridor["heavy_share"] = corridor["heavy_share"].fillna(0.0).clip(0, 1)

    # 清洗稽核依走廊拆分
    if "_was_zero_zero" in geo.columns:
        z = geo[geo["_was_zero_zero"]]
        for cid, cnt in z.groupby("corridor_id").size().items():
            audit.by_corridor[cid]["zero_zero_rows"] += int(cnt)

    corridor["congestion_level"] = corridor["speed_kmh"].map(congestion_level).astype("Int64")
    corridor = corridor[[
        "date", "time_slot_15m", "corridor_id", "speed_kmh", "volume_15m",
        "travel_time_min", "heavy_share", "congestion_level", "n_sections",
    ]].sort_values(["date", "corridor_id", "time_slot_15m"]).reset_index(drop=True)

    sec = sec[[
        "date", "time_slot_15m", "corridor_id", "GantryFrom", "GantryTo",
        "km_from", "dist_km", "speed_kmh", "volume_15m", "travel_time_min",
    ]].sort_values(["date", "corridor_id", "km_from", "time_slot_15m"]).reset_index(drop=True)

    five = passenger[[
        "date", "TimeInterval", "corridor_id", "GantryFrom", "GantryTo",
        "km_from", "SpaceMeanSpeed", "TrafficVolume",
    ]].rename(columns={"SpaceMeanSpeed": "speed_kmh", "TrafficVolume": "volume_5m"}).copy()
    five["time_slot_5m"] = five["TimeInterval"].dt.hour * 12 + five["TimeInterval"].dt.minute // 5
    five = five[[
        "date", "time_slot_5m", "corridor_id", "GantryFrom", "GantryTo",
        "km_from", "speed_kmh", "volume_5m",
    ]].sort_values(["date", "corridor_id", "km_from", "time_slot_5m"]).reset_index(drop=True)

    LOGGER.info("聚合完成：走廊 %d 列、區間 %d 列、5分 %d 列、原始 %d 列",
                len(corridor), len(sec), len(five), audit.raw_rows)
    return corridor, sec, five, audit.to_dict()
