"""Summarize the original RAPID paper's model results (Pendleton et al. 2020, ML_Logs.zip from ipendlet/MLScripts).
For every model and feature set: leave-one-amine-out (LOO) vs standard train/test split, success = crystal score 4.
Usage: python rapid/original_baselines.py   (-> rapid/results_original_baselines.md)"""
import zipfile
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from util import md  # noqa: E402
SRC = ROOT / "data" / "MLScripts" / "temp_densityconc" / "ML_Logs.zip"
OUT = ROOT / "results_original_baselines.md"
COLS = {"matthewCoef_success mean": "MCC", "f1_success mean": "F1", "precision_success mean": "precision", "recall_success mean": "recall"}


def read(z, name):
    rows = []
    for f in z.namelist():
        if f.endswith("/" + name) and "__MACOSX" not in f:
            d = pd.read_csv(z.open(f)).rename(columns={"Unnamed: 0": "data", "Unnamed: 1": "features"})
            rows.append(d.assign(model=f.split("/")[-2]))
    return pd.concat(rows)[["model", "data", "features", *COLS]].rename(columns=COLS)


def main():
    with zipfile.ZipFile(SRC) as z:
        loo, std = read(z, "LeaveOneOut_Summary.csv"), read(z, "StandardTestTrain_Summary.csv")
    T = loo.merge(std, on=["model", "data", "features"], suffixes=(" LOO", " split"))
    best = T.sort_values("MCC LOO", ascending=False).groupby("model").head(1).round(3)
    OUT.write_text("# Original RAPID baselines (Pendleton et al. 2020, ML_Logs.zip)\n\n"
                   "Success = crystal score 4. LOO = leave-one-amine-out (mean over held-out amines); split = standard train/test.\n"
                   "data: sampled = stratified sample, all = full data. Random classifier MCC = 0.\n\n"
                   f"## best LOO per model\n{md(best, index=False)}\n\n## all\n{md(T.round(3), index=False)}\n",
                   encoding="utf-8")
    print(best[["model", "data", "features", "MCC LOO", "MCC split", "F1 LOO", "F1 split"]].to_string(index=False))


if __name__ == "__main__":
    main()
