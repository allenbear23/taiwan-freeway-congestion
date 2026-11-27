# -*- coding: utf-8 -*-
"""[MOCK_TEST_ONLY] 僅驗證清洗與聚合邏輯，絕不作為研究結論。"""
from __future__ import annotations

import unittest

import pandas as pd

from src.clean import CleaningAudit, aggregate


def mock_rows() -> pd.DataFrame:
    """[MOCK_TEST_ONLY] 最小的 M05A 結構样本。"""
    rows = []
    for minute, speed, vol in [(0, 85, 10), (5, 0, 0), (10, 75, 12), (15, 70, 9)]:
        rows.append([pd.Timestamp(f"2024-02-09 06:{minute:02d}"),
                     "05F0000S", "05F0040S", 31, speed, vol])
    rows.append([pd.Timestamp("2024-02-09 06:00"), "05F0000S", "05F0040S", 42, 60, 2])
    rows.append([pd.Timestamp("2024-02-09 06:00"), "99F0000S", "99F0123S", 31, 80, 5])
    return pd.DataFrame(rows, columns=[
        "TimeInterval", "GantryFrom", "GantryTo", "VehicleType",
        "SpaceMeanSpeed", "TrafficVolume",
    ])


class TestClean(unittest.TestCase):
    def test_zero_zero_is_missing_then_interpolated(self):
        audit = CleaningAudit()
        c, s, f, result = aggregate(mock_rows(), audit=audit)
        self.assertEqual(result["zero_zero_rows"], 1, "[MOCK_TEST_ONLY] 應偵測一筆 0/0")
        self.assertEqual(result["interpolated_rows"], 1, "[MOCK_TEST_ONLY] 應由前後補回一筆")
        self.assertEqual(result["freeflow_filled_rows"], 0)
        self.assertGreaterEqual(result["section_rows_dropped"], 1, "[MOCK_TEST_ONLY] 非走廊應丟棄")
        self.assertFalse(c.empty)

    def test_no_purple_burst_from_zero_zero(self):
        audit = CleaningAudit()
        c, s, f, result = aggregate(mock_rows(), audit=audit)
        n5 = c[c["corridor_id"] == "N5_SB"]
        self.assertFalse((n5["congestion_level"] == 3).any(),
                         "[MOCK_TEST_ONLY] 0/0 不得被當成紫爆")

    def test_15min_slots(self):
        c, s, f, result = aggregate(mock_rows(), audit=CleaningAudit())
        self.assertTrue(c["time_slot_15m"].between(0, 95).all())
        self.assertLessEqual(c["time_slot_15m"].max(), 95)


if __name__ == "__main__":
    unittest.main()
