"""Experiment 1, cluster bootstrap with two simple baselines next to v2, on the same resamples as exp1_bootstrap.py
(same seed per replicate -> same papers drawn, same candidates, same hidden answers).
- Ridge: v2's input variables, one-hot categoricals, standardized numerics, median fill + missing flags;
  alpha chosen per replicate by paper-level GroupKFold(5) inside the training data (lowest MSE).
- kNN: for each candidate, mean of the k=5 nearest combinations still in the training data (process-variable
  distance only, no device conditions); the candidate itself is never its own neighbour.
v2 is rerun on the same replicates and checked against the earlier bootstrap_runs CSV.
Usage: python experiments/exp1/exp1_bootstrap_baselines.py [R=200] [earlier bootstrap_runs_<timestamp>.csv]
Outputs: experiments/exp1/bootstrap_baselines_<timestamp>.csv, experiments/exp1/results_bootstrap_baselines.md,
progress every 10 replicates in experiments/exp1/boot_progress.log."""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from experiments.exp1.exp1_hide_top import DMSO_ORDER, PARTS, candidate_inputs, combo_mean, fed_values, load_past
from experiments.exp1.exp1_bootstrap import SETTINGS, SHARE, candidates, lgbm, log, metrics, resample, tie_key
from models.v2.model_v2 import ALL, Prep
from src.features import md

OUT = Path(__file__).resolve().parent
ALPHAS = [0.1, 1, 10, 100]
K_NN = 5
METHODS = ["v2", "Ridge", f"kNN k={K_NN}"]
PAIRS = [("v2", "Ridge"), ("v2", METHODS[2]), ("Ridge", METHODS[2])]


# ---- Ridge ----
class RidgeModel:
    def __init__(self, alpha, features=ALL):
        self.alpha, self.features = alpha, features

    def _enc(self, X):
        t = self.prep.transform(X)  # categoricals use the train-fitted category lists, so columns stay fixed
        return pd.get_dummies(t, columns=[c for c in t if isinstance(t[c].dtype, pd.CategoricalDtype)], dtype=float)

    def fit(self, X, y):
        self.prep = Prep(self.features, impute=True).fit(X)  # median fill + missing flags
        Z = self._enc(X)
        self.cols = Z.columns
        self.num = [c for c in Z if c in C.NUMERIC]  # standardize numeric inputs only; one-hot and flags stay 0/1
        self.mu, self.sd = Z[self.num].mean(), Z[self.num].std().replace(0, 1)
        self.m = Ridge(alpha=self.alpha).fit(self._scale(Z), y)
        return self

    def _scale(self, Z):
        Z = Z.reindex(columns=self.cols, fill_value=0.0)
        Z[self.num] = (Z[self.num] - self.mu) / self.sd
        return Z.values

    def predict(self, X):
        return self.m.predict(self._scale(self._enc(X)))


def ridge(train, features=ALL):
    mse = {}
    for a in ALPHAS:
        err = []
        for i, j in GroupKFold(C.CV_FOLDS).split(train, groups=train["doi"]):
            p = RidgeModel(a, features).fit(train.iloc[i], train.pce.iloc[i]).predict(train.iloc[j])
            err.append(np.mean((p - train.pce.iloc[j].values) ** 2))
        mse[a] = np.mean(err)
    alpha = min(mse, key=mse.get)
    return RidgeModel(alpha, features).fit(train, train.pce), alpha


# ---- kNN on process variables ----
def parts(keys):
    p = pd.Series(keys, index=keys).str.split(" / ", expand=True)
    p.columns = PARTS
    return p.astype(object)  # plain numpy object arrays broadcast; pandas string arrays don't


def knn_scores(cand, train, tie):
    known = train.dropna(subset=["key"]).groupby("key").pce.mean()  # every combination still in training
    pc, pk = parts(cand.index), parts(known.index)
    span = {v: max(pd.concat([pc[v], pk[v]]).astype(float).pipe(lambda s: s.max() - s.min()), 1) for v in ("temp", "time")}
    D = np.zeros((len(pc), len(pk)))
    for v in ("solvent", "antisolvent", "additive", "solvent_annealing"):
        D += pc[v].to_numpy()[:, None] != pk[v].to_numpy()[None, :]
    D += np.abs(pc.dmso.map(DMSO_ORDER).to_numpy(float)[:, None] - pk.dmso.map(DMSO_ORDER).to_numpy(float)[None, :]) / 3
    for v in ("temp", "time"):
        D += np.abs(pc[v].to_numpy(float)[:, None] - pk[v].to_numpy(float)[None, :]) / span[v]
    D[pc.index.to_numpy(object)[:, None] == pk.index.to_numpy(object)[None, :]] = np.inf  # never its own neighbour
    ktie = tie_key(pk.index)
    out = []
    for i in range(len(pc)):
        nn = np.lexsort((ktie, D[i]))[:K_NN]  # nearest first, fixed tie key
        out.append(known.values[nn].mean())
    return pd.Series(out, index=cand.index)


def replicate(F, rep, seed):
    s_resample, _, _ = np.random.SeedSequence(seed).spawn(3)  # same spawn as exp1_bootstrap.replicate -> same resample
    D = F.assign(orig_doi=F.doi) if rep < 0 else resample(F, np.random.default_rng(s_resample))
    cand = candidates(D)
    k = int(round(len(cand) * SHARE))
    answers = set(cand.pce.nlargest(k).index)
    tie = tie_key(cand.index)
    X = candidate_inputs(D[D.key.isin(cand.index)])
    assert fed_values(X) == {v: [str(x)] for v, x in E.PREDICT_AT.items()}
    answer_papers = set(D.loc[D.key.isin(answers), "orig_doi"])
    rows = []
    for setting in SETTINGS:
        train = D[~D.key.isin(answers)] if setting == SETTINGS[0] else D[~D.orig_doi.isin(answer_papers)]
        rm, alpha = ridge(train)
        scores = {"v2": combo_mean(X, lgbm(train, train.pce).predict(X)),
                  "Ridge": combo_mean(X, rm.predict(X)),
                  METHODS[2]: knn_scores(cand, train, tie)}
        for name, sc in scores.items():
            rows.append({"rep": rep, "seed": seed, "setting": setting, "method": name, "n_candidates": len(cand),
                         "n_answers": k, "train_rows": len(train), "ridge_alpha": alpha if name == "Ridge" else np.nan,
                         **metrics(sc, cand, answers, tie)})
    return rows


def summarize(R):
    point = R[R.rep == -1].set_index(["setting", "method"]).top10_share
    B = R[R.rep >= 0]
    S = B.groupby(["setting", "method"]).top10_share.agg(
        R="size", mean="mean", lo=lambda v: v.quantile(0.025), hi=lambda v: v.quantile(0.975)).reset_index()
    S.insert(3, "point (original data)", [point.get((s, m), np.nan) for s, m in zip(S.setting, S.method)])
    w = B.pivot_table(index=["rep", "setting"], columns="method", values="top10_share")
    P = []
    for a, b in PAIRS:
        for s, g in (w[a] - w[b]).groupby(level="setting"):
            P.append({"setting": s, "comparison": f"{a} - {b}", "R": len(g), "mean diff": g.mean(),
                      "2.5%": g.quantile(0.025), "97.5%": g.quantile(0.975), "higher": (g > 0).mean(),
                      "equal": (g == 0).mean(), "lower": (g < 0).mean()})
    alpha = B[B.method == "Ridge"].ridge_alpha.value_counts().sort_index()
    return S, pd.DataFrame(P), alpha


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    earlier = OUT / Path(sys.argv[2]).name if len(sys.argv) > 2 else None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = OUT / f"bootstrap_baselines_{stamp}.csv"
    F = load_past()
    log(f"baselines start {stamp}: R={n}, {len(F)} rows <= {E.CUTOFF_YEAR}")
    t0 = time.time()
    runs = replicate(F, -1, E.SEED)
    for rep in range(n):
        runs += replicate(F, rep, E.SEED + rep)
        pd.DataFrame(runs).to_csv(raw, index=False)
        if (rep + 1) % 10 == 0:
            log(f"baselines replicate {rep + 1}/{n} done, elapsed {(time.time() - t0) / 60:.1f} min")
    R = pd.DataFrame(runs)
    S, P, alpha = summarize(R)

    check = ""
    if earlier is not None and earlier.exists():  # same resamples -> v2 must reproduce the earlier run exactly
        old = pd.read_csv(earlier)
        a = R[R.method == "v2"].set_index(["rep", "setting"])[["n_candidates", "n_answers", "top10_share"]]
        b = old[old.method == "v2"].set_index(["rep", "setting"])[["n_candidates", "n_answers", "top10_share"]]
        j = a.join(b, rsuffix="_old", how="inner")
        same = (j.n_candidates == j.n_candidates_old) & (j.n_answers == j.n_answers_old) & np.isclose(j.top10_share, j.top10_share_old)
        check = f"v2 reproduces {earlier.name}: {same.sum()}/{len(j)} replicate x setting rows identical"
    (OUT / "results_bootstrap_baselines.md").write_text(
        f"# exp1 baselines on the cluster bootstrap (generated by experiments/exp1/exp1_bootstrap_baselines.py)\n\n"
        f"raw: {raw.name}; R={n}; main metric = share of hidden top-10% answers in the top 10% of the ranking "
        f"(random = {SHARE}). {check}\n\n## methods\n{md(S.round(3), index=False)}\n\n"
        f"## paired differences (same replicates)\n{md(P.round(3), index=False)}\n\n"
        f"## Ridge alpha chosen (count over replicates x settings)\n{md(alpha.to_frame('count'))}\n", encoding="utf-8")
    log(f"baselines finished in {(time.time() - t0) / 60:.1f} min")
    print(check)
    for _, r in S.iterrows():
        print(f"{r.setting:24s} {r.method:10s} point {r['point (original data)']:.2f}  boot {r['mean']:.2f} ({r.lo:.2f}-{r.hi:.2f})")


if __name__ == "__main__":
    main()
