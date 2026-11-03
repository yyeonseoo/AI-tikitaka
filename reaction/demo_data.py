"""Precompute everything the local demo (reaction/demo/index.html) shows, so the page needs no server.
For each of the 15 Suzuki pairs: LightGBM trained without that pair ranks its 384 conditions (round 0 = one-shot
recommendation), then 9 rounds of 4 experiments re-train on the revealed yields (plan 5 greedy, but starting from the
model's top 4 instead of 4 random picks). Also: exact random expectation, history-average order, RDKit 3D coordinates.
Usage: python reaction/demo_data.py   (-> reaction/demo/data.js)"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model2 import desc_sm  # noqa: E402
from model5 import BATCH, N_MAX, NEAR, ROUNDS, history_order, lgbm_ensemble  # noqa: E402
from prepare import load_sm  # noqa: E402

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "demo" / "data.js"
COND = ["ligand", "base", "solvent"]


def sequential(X, y, hist, target):
    """Round 0 ranks with history only; each later round adds the 4 newest results. Returns order and per-round predictions."""
    threadpool_limits(1)
    done, preds = [], []
    for _ in range(ROUNDS + 1):
        trn = np.concatenate([hist, done]).astype(int)
        mu, _ = lgbm_ensemble(X[trn], y[trn], X[target], "warm")
        preds.append(np.clip(mu, 0, 100))
        rest = [i for i in np.argsort(-mu, kind="stable") if target[i] not in done]
        done += [target[i] for i in rest[:BATCH]]
    return [int(np.flatnonzero(target == t)[0]) for t in done], preds


def mol3d(smiles):
    if not smiles:
        return None
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(m, randomSeed=7) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            pass
        return Chem.MolToMolBlock(m)
    except Exception:
        return None


def main():
    d = load_sm()
    y = d.y.to_numpy(float)
    key = d[["electrophile", "nucleophile"]].agg(" + ".join, axis=1).to_numpy()
    pairs = list(pd.unique(key))
    X = desc_sm(d).to_numpy(float)
    runs = Parallel(n_jobs=-1)(delayed(sequential)(X, y, np.flatnonzero(key != p), np.flatnonzero(key == p)) for p in pairs)
    hist = history_order(d)

    out_pairs = []
    for p, (order, preds) in zip(pairs, runs):
        rows = d[key == p]
        yp = rows.y.to_numpy(float)
        best, n = yp.max(), len(yp)
        m = int((yp >= best - NEAR).sum())
        ys = np.sort(yp)
        rand = [sum(comb(i - 1, k - 1) / comb(n, k) * ys[i - 1] for i in range(k, n + 1)) for k in range(1, N_MAX + 1)]
        ai_best = np.maximum.accumulate(yp[order])
        hist_best = np.maximum.accumulate(hist[p][0])
        e, nu = rows.electrophile.iloc[0], rows.nucleophile.iloc[0]
        others = [q for q in pairs if q != p]
        out_pairs.append({
            "id": p, "electrophile": e, "nucleophile": nu,
            "conds": rows[COND].to_numpy().tolist(), "y": np.round(yp, 1).tolist(),
            "preds": [np.round(q, 1).tolist() for q in preds], "order": order,
            "best": round(best, 1), "nearCount": m,
            "aiFound": int(np.argmax(ai_best >= best - NEAR) + 1) if (ai_best >= best - NEAR).any() else None,
            "randomExpected": round((n + 1) / (m + 1), 1),
            "curves": {"ai": np.round(ai_best, 1).tolist(), "random": np.round(rand, 1).tolist(), "history": np.round(hist_best, 1).tolist()},
            "familiar": {"sameElectrophile": sum(q.startswith(e + " +") for q in others),
                         "sameNucleophile": sum(q.endswith("+ " + nu) for q in others),
                         "roleSwapped": nu == "2d, Bromide"},
        })
        print(f"  {p}: best {best:.1f}, AI near-best at {out_pairs[-1]['aiFound']}, random {out_pairs[-1]['randomExpected']}", flush=True)

    smi = pd.read_csv(ROOT / "reagents_sm.csv", keep_default_na=False)
    mols = {f"{r.slot}|{r['name']}": {"slot": r.slot, "name": r["name"], "smiles": r.smiles, "note": r.note, "mol": mol3d(r.smiles)}
            for _, r in smi.iterrows()}
    print("  3D failed:", [k for k, v in mols.items() if v["mol"] is None])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.DEMO = " + json.dumps({"pairs": out_pairs, "mols": mols}, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
