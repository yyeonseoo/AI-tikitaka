"""v2 settings: which variables go in, how categories are grouped, and the tuning grid.
Change things here, not in models/v2/model_v2.py."""

# Variable groups (names are the derived columns built in model_v2.make_features)
PROCESS = ["solvent", "dmso_frac", "antisolvent", "anneal_temp", "anneal_time", "additive", "solvent_annealing"]
CONDITION = ["arch", "backcontact", "flexible", "etl", "htl"]  # provisional: still under review
CORRECTION = ["year", "scan_direction"]
GROUPS = {"process": PROCESS, "condition": CONDITION, "correction": CORRECTION}

# Not used, kept here so the decision is visible (see reports/eda_report.md sections 7 and 9-5)
EXCLUDED = ["perovskite_thickness", "atmosphere", "relative_humidity", "precursor_concentration",
            "Voc", "Jsc", "FF", "hysteresis_index"]

NUMERIC = ["dmso_frac", "anneal_temp", "anneal_time", "solvent_annealing", "flexible", "year"]
# everything else in GROUPS is categorical; missing values become this category
MISSING = "missing"

# keep the N most frequent values (counted on train rows), the rest -> "other"
TOP_N = {"solvent": 6, "antisolvent": 7, "additive": 5, "backcontact": 5, "etl": 6, "htl": 6}

# When predicting for a recommendation, pin the correction variables to these values
PREDICT_AT = {"year": 2019, "scan_direction": "Reversed"}

# Cleaning
SUN_RANGE = (90, 110)  # mW/cm2, 1 sun
PCE_CONSISTENCY_TOL = 1.0  # %p, |Voc*Jsc*FF - PCE|

# Split / CV
TEST_SIZE = 0.2
SEED = 0
CV_FOLDS = 5

# Light tuning grid (GroupKFold on train only)
GRID = {
    "rf": [{"min_samples_leaf": m, "max_features": f} for m in (1, 5) for f in (0.33, 1.0)],
    "lgbm": [{"num_leaves": n, "n_estimators": e, "learning_rate": 0.05} for n in (15, 63) for e in (300, 800)],
    "hgb": [{"max_leaf_nodes": n, "learning_rate": lr, "max_iter": 300} for n in (15, 63) for lr in (0.05, 0.1)],
}

# Experiment 1/2 combination bins
DMSO_BINS = [-0.001, 0.0, 0.35, 0.9, 1.0]  # 0 / (0, 0.35] / (0.35, 0.9] / (0.9, 1]
TEMP_BIN = 10  # degC
TIME_BIN = 5  # min
MIN_DEVICES = 3
SPLIT_YEAR = 2017  # experiment 2: train <= 2017, evaluate 2018-2019
EVAL_YEARS = (2018, 2019)
