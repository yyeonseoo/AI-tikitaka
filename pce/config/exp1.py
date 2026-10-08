"""Experiment 1 (find hidden high-PCE combinations) settings. Combination bins come from config/v2.py."""

CUTOFF_YEAR = 2017  # keep 2018-19 untouched for experiment 2
MIN_DEVICES = 3
MIN_PAPERS = 2
MIN_CANDIDATES = 100  # stop and ask if fewer candidates than this

HIDE_SHARES = [0.05, 0.10, 0.20]  # top share of candidates (by mean PCE) hidden from training = answers
TOP_NS = [10, 20, 50]

PREDICT_AT = {"year": 2017, "scan_direction": "Reversed"}  # pinned when scoring candidates
N_BOOT = 10  # paper-level bootstrap LightGBMs for the uncertainty score
KAPPAS = [0.5, 1, 2]  # score = mean + kappa * std
N_RANDOM = 1000
SEED = 0

# Chemistry constraint (pending mentee check): report with and without
EXCLUDE_DMSO_BIN = "0.35-0.9"
TEMP_RANGE = (70, 150)  # degC, applied to the 10-degree bin start
