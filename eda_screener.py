# -*- coding: utf-8 -*-
"""Step 1：資料下載解析 → 零車速清洗 → 24 假設地毯式檢定 → 可行性排行榜。

執行範例：
  python eda_screener.py --years 2023 2024 2025
  python eda_screener.py --skip-download --years 2024
  python eda_screener.py --fetch-trends --years 2024 2025
  python eda_screener.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import (  # noqa: E402
    HOLIDAY_EVENTS, OUTPUTS, PROCESSED, WEEKEND_CONTROL_MONTHS,
)
from src.clean import CleaningAudit, aggregate  # noqa: E402
from src.external import collect_all  # noqa: E402
from src.features import FeatureBuilder  # noqa: E402
from src.plots import plot_candidate  # noqa: E402
from src.report import golden_triangle, render, write_report  # noqa: E402
from src.stats_tests import evaluate_all  # noqa: E402
from src.tdcs import ensure_day, read_m05a_day, read_m06a_day  # noqa: E402
from src.utils import LOGGER, daterange, load_gantry_overrides, md_table, setup_logger  # noqa: E402

ROADS = {"01", "02", "03", "05"}


def target_dates(years: list[int], with_controls: bool) -> list[pd.Timestamp]:
    days: set[pd.Timestamp] = set()
    for e in HOLIDAY_EVENTS:
        if e["year"] in years:
            days.update(pd.to_datetime(list(daterange(e["start"], e["end"]))))
            # 連假前後各 1 日
            days.add(pd.Timestamp(e["start"]) - pd.Timedelta(days=1))
            days.add(pd.Timestamp(e["end"]) + pd.Timedelta(days=1))
    if with_controls:
        for y in years:
            for m in WEEKEND_CONTROL_MONTHS:
                for day in daterange(f"{y}-{m:02d}-01", f"{y}-{m:02d}-28"):
                    if day.weekday() >= 5:
                        days.add(pd.Timestamp(day).normalize())
    return sorted(days)


def build_panels(dates: list[pd.Timestamp], skip_download: bool,
                 m06a_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, str]:
    audit_total = CleaningAudit()
    corridors, sections, fives = [], [], []
    for i, day in enumerate(dates, 1):
        ds = day.strftime("%Y%m%d")
        LOGGER.info("[%d/%d] M05A %s", i, len(dates), ds)
        day_dir = PROCESSED / "days" / ds
        cached = day_dir / "corridor.parquet"
        try:
            if skip_download and cached.exists():
                import json
                c = pd.read_parquet(cached)
                s = pd.read_parquet(day_dir / "section.parquet")
                f = pd.read_parquet(day_dir / "five.parquet")
                a = json.loads((day_dir / "audit.json").read_text(encoding="utf-8"))
                for k in ("raw_rows", "zero_zero_rows", "interpolated_rows",
                          "freeflow_filled_rows", "section_rows_dropped",
                          "inactive_section_rows_dropped"):
                    setattr(audit_total, k, getattr(audit_total, k) + a.get(k, 0))
                for cid, v in a.get("by_corridor", {}).items():
                    for kk, vv in v.items():
                        audit_total.by_corridor[cid][kk] += vv
                LOGGER.debug("使用快取 %s", ds)
            else:
                desc = ensure_day("M05A", day) if not skip_download else None
                if desc is None:
                    from config import RAW
                    p = RAW / "M05A" / f"M05A_{ds}.tar.gz"
                    desc = {"kind": "tar", "path": p, "date": ds} if p.exists() else None
                if desc is None:
                    LOGGER.warning("缺檔，略過 %s", ds)
                    continue
                raw = read_m05a_day(desc, roads=ROADS)
                if raw.empty:
                    LOGGER.warning("無資料，略過 %s", ds)
                    continue
                c, s, f, a = aggregate(raw, overrides=load_gantry_overrides(), audit=audit_total)
                if not c.empty:
                    day_dir.mkdir(parents=True, exist_ok=True)
                    c.to_parquet(cached, index=False)
                    s.to_parquet(day_dir / "section.parquet", index=False)
                    f.to_parquet(day_dir / "five.parquet", index=False)
                    (day_dir / "audit.json").write_text(
                        __import__("json").dumps(a, ensure_ascii=False), encoding="utf-8")
            if not c.empty:
                corridors.append(c)
                sections.append(s)
                fives.append(f)
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("處理 %s 失敗：%s", ds, e)

    panel = pd.concat(corridors, ignore_index=True) if corridors else pd.DataFrame()
    sec = pd.concat(sections, ignore_index=True) if sections else pd.DataFrame()
    five = pd.concat(fives, ignore_index=True) if fives else pd.DataFrame()

    m06a, m06a_status = pd.DataFrame(), "DISABLED: --m06a-days 0"
    if m06a_days > 0:
        sample = dates[:m06a_days]
        parts = []
        for day in sample:
            try:
                desc = ensure_day("M06A", day)
                parts.append(read_m06a_day(desc, max_rows=150_000))
                LOGGER.info("M06A %s 抽樣 %d 列", day.strftime("%Y%m%d"), len(parts[-1]))
            except Exception as e:  # noqa: BLE001
                LOGGER.warning("M06A %s 失敗：%s", day.strftime("%Y%m%d"), e)
        if parts:
            m06a = pd.concat(parts, ignore_index=True)
            m06a_status = f"OK（抽樣 {len(sample)} 天）" if len(m06a) else "API_ERROR"
    return panel, sec, five, audit_total.to_dict(), m06a, m06a_status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--no-controls", action="store_true", help="不下載一般週末對照組")
    ap.add_argument("--skip-download", action="store_true", help="只用本機快取")
    ap.add_argument("--fetch-trends", action="store_true", help="嘗試以 pytrends 抓取（可能被封鎖）")
    ap.add_argument("--m06a-days", type=int, default=0, help="抽樣抓取 M06A 的天數（檔案很大）")
    ap.add_argument("--selftest", action="store_true", help="只執行 [MOCK_TEST_ONLY] 單元測試")
    args = ap.parse_args(argv)

    setup_logger()
    if args.selftest:
        import unittest
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

    dates = target_dates(args.years, not args.no_controls)
    LOGGER.info("目標日期 %d 天（%s ~ %s）", len(dates), dates[0].date(), dates[-1].date())

    panel, sec, five, audit, m06a, m06a_status = build_panels(dates, args.skip_download, args.m06a_days)
    if panel.empty:
        LOGGER.error("沒有任何走廊面板資料，請確認下載或指定年份。")
        return 2

    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED / "corridor_15m.parquet", index=False)
    sec.to_parquet(PROCESSED / "section_15m.parquet", index=False)
    five.to_parquet(PROCESSED / "section_5m.parquet", index=False)

    trend_start = f"{min(args.years)-1}-12-01"
    trend_end = f"{max(args.years)}-12-31"
    external = collect_all(args.years, list(panel["date"].unique()),
                           fetch_trends_flag=args.fetch_trends,
                           trend_start=trend_start, trend_end=trend_end)
    builder = FeatureBuilder(panel, sec, five, external, args.years, m06a, m06a_status)
    observations = builder.build_all()
    scores = evaluate_all(observations)

    fig_paths = []
    result_map = {r["id"]: r for r in scores.to_dict("records")}
    for o in observations:
        p = plot_candidate(o, result_map[o.hid])
        if p:
            fig_paths.append(p)

    golden = golden_triangle(scores)
    sources = {
        "M05A (ETC)": "OK" if audit.get("raw_rows", 0) > 0 else "MANUAL_NEEDED",
        "M06A (ETC)": m06a_status,
        **{k: v for k, (_, v) in external.items()},
    }
    text = render(scores, audit, sources, panel, args.years, fig_paths, golden)
    write_report(text)

    print("\n===== 清理前後對比 =====")
    print(md_table(pd.DataFrame([
        {"項目": k, "值": v} for k, v in audit.items() if k != "by_corridor"
    ])))
    print("\n===== 可行性排行榜 Top 10 =====")
    ok = scores[scores["狀態"] == "可用"].head(10)
    print(md_table(ok[["id", "假設", "n", "effect", "gain", "stability", "總分"]]) if len(ok) else "無可用候選")
    print("\n===== 黃金三角 Top 3 =====")
    print(md_table(golden))
    print(f"\n完整報告：{OUTPUTS / 'feasibility_report.md'}")
    print(f"候選圖數量：{len(fig_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
