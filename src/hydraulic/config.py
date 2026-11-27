"""Dataset layout for the UCI 'Condition monitoring of hydraulic systems' data.

17 sensor files, one row per 60 s load cycle (2,205 cycles), sampled at
different rates. profile.txt holds the condition labels per cycle.
"""

from __future__ import annotations

# sensor name -> samples per cycle (sampling rate x 60 s)
SENSORS: dict[str, int] = {
    # pressure [bar], 100 Hz
    "PS1": 6000, "PS2": 6000, "PS3": 6000, "PS4": 6000, "PS5": 6000, "PS6": 6000,
    # motor power [W], 100 Hz
    "EPS1": 6000,
    # volume flow [l/min], 10 Hz
    "FS1": 600, "FS2": 600,
    # temperature [degC], 1 Hz
    "TS1": 60, "TS2": 60, "TS3": 60, "TS4": 60,
    # vibration [mm/s], 1 Hz
    "VS1": 60,
    # cooling efficiency [%], cooling power [kW], efficiency factor [%], 1 Hz
    "CE": 60, "CP": 60, "SE": 60,
}

N_CYCLES = 2205

# columns of profile.txt
PROFILE_COLUMNS = ["cooler", "valve", "pump", "accumulator", "stable"]

# the four components this project classifies, with their label meanings
TARGETS: dict[str, dict[int, str]] = {
    "cooler": {3: "close to total failure", 20: "reduced efficiency", 100: "full efficiency"},
    "valve": {100: "optimal", 90: "small lag", 80: "severe lag", 73: "close to total failure"},
    "pump": {0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
    "accumulator": {
        130: "optimal pressure",
        115: "slightly reduced pressure",
        100: "severely reduced pressure",
        90: "close to total failure",
    },
}

# Time-blocked evaluation.
# Consecutive cycles on the rig are near-copies (a valve state lasts ~15 cycles,
# an accumulator state ~180), so any split that puts neighbouring cycles on
# both sides leaks. The cycle sequence is cut into N_TIME_BLOCKS contiguous
# blocks. Every TEST_EVERY-th block is the held-out test set, spread over the
# whole run so that every condition level appears in it. PURGE_CYCLES cycles on
# each side of a test block are dropped from training, so no training cycle
# sits right next to a test cycle.
N_TIME_BLOCKS = 20
TEST_EVERY = 5          # blocks 2, 7, 12, 17 -> 20 % of cycles
TEST_OFFSET = 2
PURGE_CYCLES = 15

# How models are selected per target.
# "purged": the training cycles are cut into contiguous blocks, folds take
#   every k-th block, and cycles next to a validation block are left out of
#   that fold's training data. Used for every component.
# "blocked" (plain time-ordered folds) and "stratified" (shuffled folds) are
# kept for comparison.
TARGET_CV: dict[str, str] = {
    "cooler": "purged",
    "valve": "purged",
    "pump": "purged",
    "accumulator": "purged",
}

# whole-cycle statistics
FEATURE_STATS = ("mean", "std", "min", "max")
# the cycle is split into this many equal time windows; the mean of each
# window describes the signal's shape (e.g. how fast pressure builds up)
N_WINDOWS = 6
# trend and roughness features
SHAPE_STATS = ("slope", "absdiff")
