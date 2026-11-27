# -*- coding: utf-8 -*-
"""共用工具：log、重試下載、快取、時間刻度、走廊比對。"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests

from config import (
    CORRIDORS, DOWNLOAD_RETRIES, REQUEST_TIMEOUT, SLOT_MINUTES,
    SLOTS_PER_DAY, TISV_BASE,
)

LOGGER = logging.getLogger("eda")


def setup_logger(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    LOGGER.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(h)


def cache_path(key: str, suffix: str = ".parquet") -> Path:
    """以 key 建立穩定檔名的快取路徑。"""
    from config import CACHE
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:60]
    return CACHE / f"{safe}_{digest}{suffix}"


def http_get_text(url: str, retries: int = DOWNLOAD_RETRIES) -> str:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            LOGGER.warning("下載失敗 %s（%d/%d）：%s", url, i + 1, retries, e)
            time.sleep(2 ** i + random.random())
    raise RuntimeError(f"下載失敗：{url}") from last


def http_download(url: str, dest: Path, retries: int = DOWNLOAD_RETRIES) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last = None
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            LOGGER.info("下載完成 %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            LOGGER.warning("下載失敗 %s（%d/%d）：%s", url, i + 1, retries, e)
            time.sleep(2 ** i + random.random())
    raise RuntimeError(f"下載失敗：{url}") from last


def slot_of(ts: pd.Timestamp) -> int:
    """把時間戳記轉成 0..95 的 15 分鐘時段索引。"""
    return int(ts.hour * (60 // SLOT_MINUTES) + ts.minute // SLOT_MINUTES)


def slot_to_hhmm(slot: int) -> str:
    return f"{slot // 4:02d}:{(slot % 4) * SLOT_MINUTES:02d}"


def daterange(start: str | datetime, end: str | datetime) -> Iterator[datetime]:
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    for i in range((e - s).days + 1):
        yield (s + timedelta(days=i)).to_pydatetime()


def parse_gantry_id(gantry_id: str) -> dict | None:
    """解析 eTag 門架 ID。失敗回 None（例如未編碼的臨時門架）。"""
    if not isinstance(gantry_id, str):
        return None
    m = re.match(r"^(\d{2})([FH])(?:R)?(\d{4})([NSEW])$", gantry_id.strip())
    if not m:
        return None
    road, kind, num, direction = m.groups()
    return {
        "gantry_id": gantry_id,
        "road": road,
        "kind": kind,
        "km": int(num) / 10.0,
        "direction": direction,
    }


def match_corridor(gantry_id: str, spec: dict | None = None) -> str | None:
    """比對單一門架是否屬於指定走廊；未指定 spec 時比對所有走廊。"""
    g = parse_gantry_id(gantry_id)
    if g is None:
        return None
    candidates = {spec["name"]: spec} if spec else CORRIDORS
    for cid, c in candidates.items():
        if g["road"] == c["road"] and g["direction"] == c["direction"] \
                and g["kind"] in c["gantry_kinds"] and c["km_min"] <= g["km"] <= c["km_max"]:
            return cid
    return None


def corridor_section_km(from_id: str, to_id: str, overrides: dict[str, float] | None = None) -> float | None:
    """以門架里程差推估區間長度；可用官方對照表覆蓋。"""
    if overrides:
        if from_id in overrides and to_id in overrides:
            d = abs(overrides[to_id] - overrides[from_id])
            if d > 0:
                return d
    a, b = parse_gantry_id(from_id), parse_gantry_id(to_id)
    if not a or not b or a["road"] != b["road"]:
        return None
    d = abs(b["km"] - a["km"])
    # 國5 雪隧等長區間可達 20+ km，上限設 30 km；過大代表跨域/編碼異常
    return d if 0 < d <= 30 else None


def load_gantry_overrides(path: Path | None = None) -> dict[str, float]:
    """可選：data/external/gantry_km.csv（gantry_id,km）。"""
    if path is None:
        from config import EXTERNAL
        path = EXTERNAL / "gantry_km.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["gantry_id"].astype(str), pd.to_numeric(df["km"], errors="coerce")))


def congestion_level(speed: float) -> int:
    if pd.isna(speed):
        return -1
    if speed >= 80:
        return 0
    if speed >= 60:
        return 1
    if speed >= 40:
        return 2
    return 3


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def md_table(df: pd.DataFrame, columns: list[str] | None = None, float_digits: int = 3) -> str:
    """輸出可比對的 Markdown 表格（給報告與螢幕同步使用）。"""
    d = df[columns] if columns else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:.{float_digits}f}")
        d[c] = d[c].astype(str).str.replace("|", "\\|")
    head = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    rows = ["| " + " | ".join(r) + " |" for r in d.astype(str).values.tolist()]
    return "\n".join([head, sep, *rows])
