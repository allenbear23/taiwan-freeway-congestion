# -*- coding: utf-8 -*-
"""高公局 TISVCLOUD TDCS M05A / M06A 下載與串流解析。

站台行為（2026-11 實測）：
  * 最近 35 天：/history/TDCS/M05A/YYYYMMDD/HH/TDCS_M05A_YYYYMMDD_HHMMSS.csv
  * 較早日期：/history/TDCS/M05A/M05A_YYYYMMDD.tar.gz（M06A 同理）
  * M05A/M06A 僅含國道 1/3/5 門架，沒有國道 2。
"""
from __future__ import annotations

import io
import re
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import RAW, RECENT_DAYS, TISV_BASE
from src.utils import LOGGER, http_download, http_get_text

M05A_COLUMNS = [
    "TimeInterval", "GantryFrom", "GantryTo", "VehicleType",
    "SpaceMeanSpeed", "TrafficVolume",
]
M06A_COLUMNS = [
    "VehicleType", "DetectionTime_O", "GantryID_O", "DetectionTime_D",
    "GantryID_D", "TripLength", "TripEnd", "TripInformation",
]
FILE_RE = re.compile(r"^TDCS_(M05A|M06A)_(\d{8})_(\d{6})\.csv$")


def dataset_root(dataset: str) -> str:
    return f"{TISV_BASE}/{dataset}"


def is_recent(day: datetime) -> bool:
    return (datetime.now() - day).days <= RECENT_DAYS


def ensure_day(dataset: str, day: datetime) -> dict:
    """確保單日原始檔存在，回傳描述子供串流解析。"""
    ds = day.strftime("%Y%m%d")
    dest = RAW / dataset / f"{dataset}_{ds}.tar.gz"
    if not is_recent(day):
        url = f"{dataset_root(dataset)}/{dataset}_{ds}.tar.gz"
        return {"kind": "tar", "path": http_download(url, dest), "date": ds}

    # 最近 35 天：逐小時抓 csv，存到 RAW/<dataset>/recent/<date>/<hh>/ ...
    folder = RAW / dataset / "recent" / ds
    csvs = sorted(folder.glob("**/*.csv"))
    if not csvs:
        index_url = f"{dataset_root(dataset)}/{ds}/"
        html = http_get_text(index_url)
        hours = sorted(set(re.findall(r"href=\"(\d{2})/\"", html)))
        for hh in hours:
            hh_html = http_get_text(f"{index_url}{hh}/")
            names = sorted(set(re.findall(rf"href=\"(TDCS_{dataset}_{ds}_{hh}\d{{4}}\.csv)\"", hh_html)))
            for name in names:
                p = folder / hh / name
                http_download(f"{index_url}{hh}/{name}", p)
        csvs = sorted(folder.glob("**/*.csv"))
    return {"kind": "dir", "path": folder, "files": csvs, "date": ds}


def _read_csv(buf: io.BufferedReader, dataset: str) -> pd.DataFrame:
    cols = M05A_COLUMNS if dataset == "M05A" else M06A_COLUMNS
    try:
        df = pd.read_csv(buf, names=cols, header=None, dtype=str, encoding="utf-8",
                         engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        buf.seek(0)
        df = pd.read_csv(buf, names=cols, header=None, dtype=str, encoding="latin-1",
                         engine="python", on_bad_lines="skip")
    return df


def iter_day_files(desc: dict, dataset: str) -> Iterator[tuple[str, io.BufferedReader]]:
    """逐檔產出 (檔名, 可讀串流)，避免整日解壓到磁碟。"""
    if desc["kind"] == "tar":
        with tarfile.open(desc["path"], "r:gz") as tf:
            for m in tf:
                basename = m.name.rsplit("/", 1)[-1]
                if not m.isfile() or FILE_RE.match(basename) is None:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                yield basename, f
    else:
        for p in desc["files"]:
            if FILE_RE.match(p.name) is None:
                continue
            yield p.name, p.open("rb")


def read_m05a_day(desc: dict, roads: set[str] | None = None) -> pd.DataFrame:
    """讀取單日 M05A；roads 可用於早期過濾（例如 {'05','01'}）。"""
    chunks: list[pd.DataFrame] = []
    for name, buf in iter_day_files(desc, "M05A"):
        df = _read_csv(buf, "M05A")
        if df.empty:
            continue
        df = df[df["GantryFrom"].astype(str).str[:2].isin(roads)] if roads else df
        chunks.append(df)
    if not chunks:
        return pd.DataFrame(columns=M05A_COLUMNS)
    out = pd.concat(chunks, ignore_index=True)
    out["TimeInterval"] = pd.to_datetime(out["TimeInterval"], errors="coerce")
    out["VehicleType"] = pd.to_numeric(out["VehicleType"], errors="coerce").astype("Int64")
    out["SpaceMeanSpeed"] = pd.to_numeric(out["SpaceMeanSpeed"], errors="coerce")
    out["TrafficVolume"] = pd.to_numeric(out["TrafficVolume"], errors="coerce")
    out["date"] = out["TimeInterval"].dt.normalize()
    out["slot"] = out["TimeInterval"].dt.hour * 4 + out["TimeInterval"].dt.minute // 15
    LOGGER.debug("M05A %s：%d 列", desc["date"], len(out))
    return out


def read_m06a_day(desc: dict, max_rows: int | None = None) -> pd.DataFrame:
    """讀取單日 M06A 旅次；檔案很大，預設由 max_rows 抽樣。"""
    chunks: list[pd.DataFrame] = []
    n = 0
    for name, buf in iter_day_files(desc, "M06A"):
        df = _read_csv(buf, "M06A")
        if df.empty:
            continue
        chunks.append(df)
        n += len(df)
        if max_rows and n >= max_rows:
            break
    if not chunks:
        return pd.DataFrame(columns=M06A_COLUMNS)
    out = pd.concat(chunks, ignore_index=True)
    for c in ("VehicleType", "TripLength"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("DetectionTime_O", "DetectionTime_D"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["date"] = out["DetectionTime_O"].dt.normalize()
    if max_rows and len(out) > max_rows:
        out = out.sample(max_rows, random_state=42).reset_index(drop=True)
    return out
