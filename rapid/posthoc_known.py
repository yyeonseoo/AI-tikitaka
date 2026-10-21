"""Post-hoc check (not in any plan): for amines already tested, is the step-3 A recommender (LightGBM, AUC 0.88) better
than simple rules that only look at the same amine's training experiments?
  nearest success: rank test conditions by distance to the closest 4-point training condition of that amine
  neighbour mean : 4-point share among the 5 closest training conditions of that amine
Same split, model and test rows as step 3 A. Distance = standardized PbI2, amine and formic acid concentrations.
Usage: python rapid/posthoc_known.py   (prints a summary; numbers recorded in rapid/LOG.md)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3 import AMINE, SEED, auc, gain, part_a, prepare  # noqa: E402

CONC = ["_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid"]
N_BOOT = 1000


def main():
    d, feat, X = prepare()
    _, _, best, sizes, T = part_a(d, X)  # step 3 A: test rows with LightGBM predictions p
    trainval = d.drop(index=T.index)
    rows = []
    for g, te in T.groupby(AMINE):
        if te.y.sum() == 0:
            continue
        tr = trainval[trainval[AMINE] == g]
        mu, sd = tr[CONC].mean(), tr[CONC].std() + 1e-9
        Zt, Zr = ((te[CONC] - mu) / sd).to_numpy(), ((tr[CONC] - mu) / sd).to_numpy()
        D = np.linalg.norm(Zt[:, None] - Zr[None], axis=2)
        succ = tr.y.to_numpy() == 1
        near = -D[:, succ].min(1) if succ.any() else np.zeros(len(te))
        nb = tr.y.to_numpy()[np.argsort(D, axis=1)[:, :5]].mean(1)
        y = te.y.to_numpy()
        rows.append({"amine": g, **{f"{m}_{k}": f(y, p) for m, p in (("lgbm", te.p.to_numpy()), ("near", near), ("nbmean", nb))
                                    for k, f in (("auc", auc), ("gain", gain))}})
    R = pd.DataFrame(rows).set_index("amine")
    idx = [np.random.default_rng([SEED, 3, b]).integers(0, len(R), len(R)) for b in range(N_BOOT)]

    def ci(v):
        v = v.to_numpy(float)
        b = np.array([np.nanmean(v[i]) for i in idx])
        return f"{np.nanmean(v):+.3f} [{np.nanquantile(b, 0.025):+.3f}, {np.nanquantile(b, 0.975):+.3f}]"
    print(f"amines {len(R)}, split {sizes}")
    for k in ("auc", "gain"):
        print(k, "| LightGBM", ci(R[f"lgbm_{k}"]), "| 성공 근처", ci(R[f"near_{k}"]), "| 이웃 평균", ci(R[f"nbmean_{k}"]))
        print("   LightGBM - 성공 근처", ci(R[f"lgbm_{k}"] - R[f"near_{k}"]), "| LightGBM - 이웃 평균", ci(R[f"lgbm_{k}"] - R[f"nbmean_{k}"]))


if __name__ == "__main__":
    main()
