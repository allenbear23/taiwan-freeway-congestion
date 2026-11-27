# -*- coding: utf-8 -*-
import sys
import requests
for u in sys.argv[1:]:
    try:
        r = requests.get(u, timeout=30, stream=True)
        print(r.status_code, r.headers.get("content-type"), r.headers.get("content-length"), u)
    except Exception as e:
        print("ERR", e, u)
