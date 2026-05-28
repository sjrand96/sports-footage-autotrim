"""YouTube source IDs used by data pipeline scripts (fetch, train/test split).

For ``train_test_split.py``:
- Set ``TRAIN_SOURCE_IDS`` and ``TEST_SOURCE_IDS`` to assign whole sources to train or test.
- Leave both empty to use ``SOURCE_IDS`` with a random 80/20 per-source split instead.
"""

SOURCE_IDS: list[str] = [
    "jZ18INu4LQc",
    "vq3CZAx3GnM",
    "GRuOSrz3kdY",
    "1rXZJyVXUHU",
    "6rRMEXuLAng",
    "Fr3ue3w5QRY",
    "ANwMhMfcwGM",
    "2crSZaHIBaY",
    "Y0l5bl2Bp-M",
    "Gzx01gOde80",
    "xDEe3bIX628",
    "RCbQVAISMcU",
    "TRt7udwisVU",
]

# Whole-source train/test assignment (used when either list is non-empty).
TRAIN_SOURCE_IDS: list[str] = [
    "jZ18INu4LQc",
    "vq3CZAx3GnM",
    "GRuOSrz3kdY",
    "1rXZJyVXUHU",
    "6rRMEXuLAng",
    "Fr3ue3w5QRY",
    "ANwMhMfcwGM",
    "2crSZaHIBaY",
]
TEST_SOURCE_IDS: list[str] = [
    "Y0l5bl2Bp-M",
    "Gzx01gOde80",
    "xDEe3bIX628",
]
