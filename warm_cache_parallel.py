# -*- coding: utf-8 -*-
"""平行下載 + 解析 M05A，寫入每日快取供 eda_screener --skip-download 秒讀。

Windows  multiprocessing 需 module-level function + __main__ guard。
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import PROCESSED  # noqa: E402
from eda_screener import target_dates  # noqa: E402
from src.clean import CleaningAudit, aggregate  # noqa: E402
from src.tdcs import ensure_day, read_m05a_day  # noqa: E402
from src.utils import load_gantry_overrides  # noqa: E402

ROADS = {"01", "03", "05"}


def warm_one(ds: str) -> tuple[str, str]:
    day_dir = PROCESSED / "days" / ds
    cached = day_dir / "corridor.parquet"
    if cached.exists() and (day_dir / "audit.json").exists():
        return ds, "cached"
    try:
        desc = ensure_day("M05A", pd.Timestamp(ds))
        raw = read_m05a_day(desc, roads=ROADS)
        if raw.empty:
            return ds, "empty"
        audit = CleaningAudit()
        c, s, f, a = aggregate(raw, overrides=load_gantry_overrides(), audit=audit)
        if c.empty:
            return ds, "no_corridor"
        day_dir.mkdir(parents=True, exist_ok=True)
        c.to_parquet(cached, index=False)
        s.to_parquet(day_dir / "section.parquet", index=False)
        f.to_parquet(day_dir / "five.parquet", index=False)
        (day_dir / "audit.json").write_text(
            json.dumps(a, ensure_ascii=False), encoding="utf-8")
        return ds, "ok"
    except Exception as e:  # noqa: BLE001
        return ds, f"error:{e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    args = ap.parse_args()
    todo = [d.strftime("%Y%m%d") for d in target_dates(args.years, True)]
    print(f"目標 {len(todo)} 天，workers={args.workers}", flush=True)
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(warm_one, ds): ds for ds in todo}
        for fut in as_completed(futures):
            ds, status = fut.result()
            done += 1
            if status.startswith("error"):
                failed += 1
            if done % 10 == 0 or status.startswith("error"):
                print(f"[{done}/{len(todo)}] {ds} {status}", flush=True)
    print(f"DONE failed={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
