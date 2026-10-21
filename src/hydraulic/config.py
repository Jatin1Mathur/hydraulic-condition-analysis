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

# How models are selected per target.
# "blocked": time-ordered folds (consecutive cycles stay together), so the score
#   reflects unseen operating periods, not near-duplicate neighbouring cycles.
# "stratified": shuffled folds. Used where a label occurs in only a few long
#   blocks of time (cooler: 3 blocks, accumulator: 12), so a held-out time
#   block would contain a class the model never saw.
TARGET_CV: dict[str, str] = {
    "cooler": "stratified",
    "valve": "blocked",
    "pump": "blocked",
    "accumulator": "stratified",
}

# whole-cycle statistics
FEATURE_STATS = ("mean", "std", "min", "max")
# the cycle is split into this many equal time windows; the mean of each
# window describes the signal's shape (e.g. how fast pressure builds up)
N_WINDOWS = 6
# trend and roughness features
SHAPE_STATS = ("slope", "absdiff")
