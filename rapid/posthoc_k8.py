"""Post-hoc check (after step 4, not in any plan): with k = 8 shots of a new amine, (1) does the model help when no shot
succeeded, (2) is its gain just "conditions near a successful shot"? Compares against that nearest-success rule and
re-scores without exact repeats of shot conditions. Same splits, tuning and shot orders as steps 3-4.
Usage: python rapid/posthoc_k8.py   (prints a summary; numbers recorded in rapid/README.md)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3 import AMINE, REPS, SEED, auc, gain, prepare, shots_runs, tune_b  # noqa: E402

COND = ["_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid", "_rxn_temperatureC_actual_bulk", "_rxn_reactiontimeS", "_rxn_stirrateRPM"]
K = 8


def one(train, s, g, cfg, w, fold, X):
    y = s.y.to_numpy()
    key = s[COND].round(3).astype(str).agg("|".join, axis=1).to_numpy()
    Z = s[["_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid"]].to_numpy()
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    rows = []
    for r in range(REPS):
        perm = np.random.default_rng([SEED, 7, fold, r, len(s)]).permutation(len(s))  # same shot orders as steps 3-4
        p = shots_runs(train, s, cfg, w, perm, [K], X)[K]
        shot, rest = perm[:K], perm[K:]
        nodup = rest[~np.isin(key[rest], key[shot])]
        succ = shot[y[shot] == 1]
        # nearest-success rule: rank by distance (standardized concentrations) to the closest successful shot
        near = -np.min(np.linalg.norm(Z[:, None, :] - Z[None, succ, :], axis=2), axis=1) if len(succ) else np.zeros(len(s))
        rows.append({"amine": g, "rep": r, "succ_in_shots": int(y[shot].sum()),
                     "auc": auc(y[rest], p[rest]), "gain": gain(y[rest], p[rest]),
                     "auc_nodup": auc(y[nodup], p[nodup]), "gain_nodup": gain(y[nodup], p[nodup]),
                     "auc_near": auc(y[rest], near[rest]), "gain_near": gain(y[rest], near[rest]),
                     "dup_share": 1 - len(nodup) / len(rest)})
    return rows


def main():
    d, feat, X = prepare()
    amines = np.array(sorted(d[AMINE].unique()))
    rng = np.random.default_rng(SEED)
    outer = {g: i % 5 for i, g in enumerate(rng.permutation(amines))}
    out = []
    for f in range(5):
        test_a = [g for g in amines if outer[g] == f]
        rest_a = rng.permutation([g for g in amines if outer[g] != f])
        n_val = int(round(len(rest_a) * 0.25))
        val_a, tr_a = rest_a[:n_val], rest_a[n_val:]
        cfg, w = tune_b(d[d[AMINE].isin(tr_a)], val_a, d, f, X)
        train = d[d[AMINE].isin(np.concatenate([tr_a, val_a]))]
        res = Parallel(n_jobs=-1)(delayed(one)(train, d[d[AMINE] == g].reset_index(drop=True), g, cfg, w, f, X)
                                  for g in test_a if d.loc[d[AMINE] == g, "y"].sum() > 0)
        out += [x for r in res for x in r]
    R = pd.DataFrame(out)
    R["has_succ"] = R.succ_in_shots > 0
    cols = ["auc", "gain", "auc_nodup", "gain_nodup", "auc_near", "gain_near", "dup_share"]
    print("draws by 'any success in the 8 shots':", R.groupby("has_succ").size().to_dict(),
          "| amines with at least one all-fail draw:", R[~R.has_succ].amine.nunique())
    print("amine-weighted means:\n", R.groupby(["amine", "has_succ"])[cols].mean().groupby(level="has_succ").mean().round(3).to_string())


if __name__ == "__main__":
    main()
