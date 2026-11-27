# -*- coding: utf-8 -*-
"""抓人事行政總處「政府行政機關辦公日曆表」CSV（data.gov.tw 資料集 14718）。"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# 西元年份 -> data.gov.tw / dgpa 檔案 ID（112/113/114 年）
FILES = {
    2023: "dd89fe8c-25f5-4035-a31e-7c14b44a74d7",
    2024: "777152e9-fdd1-4a61-876c-2733e7692538",
    2025: "22f9fcbc-fbb2-4387-8bcf-73b2279666c2",
}
OUT = Path("data/external/calendar")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for year, fid in FILES.items():
        url = ("https://www.dgpa.gov.tw/FileConversion?"
               f"filename=dgpa/files/202407/{fid}.csv&nfix=&"
               f"name={year}年中華民國政府行政機關辦公日曆表.csv")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fp = OUT / f"office_calendar_{year}.csv"
        fp.write_bytes(r.content)
        print(f"{year}: {len(r.content)} bytes -> {fp}")


if __name__ == "__main__":
    main()
