# -*- coding: utf-8 -*-
"""獨立抓取 Google Trends 並落檔（避免 PowerShell 命令列中文編碼問題）。"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.external import fetch_trends

KEYWORDS = ("國5塞車", "高乘載", "替代道路", "台9線", "宜蘭住宿")

if __name__ == "__main__":
    df, status = fetch_trends(KEYWORDS, "TW", "2022-12-01", "2025-12-31")
    print(status, df.shape)
    if not df.empty:
        print(df.tail(3).to_string())
