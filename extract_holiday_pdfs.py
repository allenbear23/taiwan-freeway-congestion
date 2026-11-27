# -*- coding: utf-8 -*-
"""抽出高公局連假疏導措施 PDF 連結（ASCII 輸出）。"""
from __future__ import annotations

from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

url = "https://www.freeway.gov.tw/Upload/Html/201943202/index.html"
r = requests.get(url, timeout=30)
r.encoding = "utf-8-sig"
soup = BeautifulSoup(r.text, "html.parser")
for a in soup.find_all("a", href=True):
    href = a["href"]
    if ".pdf" not in href.lower():
        continue
    full = urljoin(url, href)
    print(quote(full, safe=":/?=&%"))
