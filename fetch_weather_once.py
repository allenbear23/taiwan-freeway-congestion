# -*- coding: utf-8 -*-
"""抓 Open-Meteo Archive（ERA5 再分析，免登入）逐時氣象，供 3A/3B/3C/3D 使用。

資料真實來源：https://open-meteo.com/en/docs/historical-weather-api
輸出：data/external/weather/hourly_{TPE,YLP,TAY}.csv（datetime, precip_mm, temp_c）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

STATIONS = {
    "TPE": (25.0330, 121.5654),  # 台北
    "YLP": (24.9918, 121.6300),  # 宜蘭
    "TAY": (24.9936, 121.3010),  # 桃園
}
START, END = "2022-12-01", "2025-12-31"
URL = "https://archive-api.open-meteo.com/v1/archive"
OUT = Path("data/external/weather")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in STATIONS.items():
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": START, "end_date": END,
            "hourly": "precipitation,temperature_2m",
            "timezone": "Asia/Taipei",
        }
        r = requests.get(URL, params=params, timeout=120)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(h["time"]),
            "precip_mm": pd.to_numeric(h["precipitation"], errors="coerce"),
            "temp_c": pd.to_numeric(h["temperature_2m"], errors="coerce"),
        })
        fp = OUT / f"hourly_{name}.csv"
        df.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)} 列 -> {fp}")


if __name__ == "__main__":
    main()
