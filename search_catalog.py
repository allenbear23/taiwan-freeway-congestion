# -*- coding: utf-8 -*-
"""在 data.gov.tw 全部資料集清單中依關鍵字搜尋。"""
from __future__ import annotations

import sys

import pandas as pd

p = r"C:\Users\user\AppData\Local\Temp\opencode\datagov_catalog.csv"
cols = ["資料集識別碼", "資料集名稱", "資料提供屬性", "檔案格式", "資料下載網址", "資料集描述", "提供機關"]
df = pd.read_csv(p, usecols=cols, dtype=str, low_memory=False)
out_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".md") else None
kws = sys.argv[1:] if out_path is None else sys.argv[2:]
buf = []
def emit(s: str) -> None:
    buf.append(s)
    print(s)
blob = (df["資料集名稱"].fillna("") + " " + df["資料集描述"].fillna("") + " " + df["資料集識別碼"].fillna(""))
mask = pd.Series(False, index=df.index)
for kw in kws:
    mask |= blob.str.contains(kw, case=False, na=False)
out = df[mask].copy()
emit(f"MATCHES {len(out)}")
for _, r in out.iterrows():
    emit(f"ID {r['資料集識別碼']} | {r['資料集名稱']}")
    emit(f"  ORG {r['提供機關']} | FMT {str(r['檔案格式'])[:120]}")
    emit(f"  URL {str(r['資料下載網址'])[:500].replace(chr(10), ' ')}")
    emit(f"  DESC {str(r['資料集描述'])[:160].replace(chr(10), ' ')}")
if out_path:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))
    print("WROTE", out_path)
