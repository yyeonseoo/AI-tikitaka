"""Experiment 1, paper-level cluster bootstrap.
Each replicate resamples papers (<= 2017) with replacement, re-derives the candidate combinations and the hidden
top-10% answers inside the resample, retrains every method without the answers, and scores the candidates.
Usage: python experiments/exp1/exp1_bootstrap.py [R_single=200] [R_ensemble=100]
       python experiments/exp1/exp1_bootstrap.py bootstrap_runs_<timestamp>.csv   (rebuild the summary only, no training)
  replicates 0..R_ensemble-1 run every method (shared, so comparisons are paired);
  replicates R_ensemble..R_single-1 run only the single-model methods (v2, B, A2).
Outputs: experiments/exp1/bootstrap_runs_<timestamp>.csv (raw, one row per replicate x setting x method),
experiments/exp1/results_bootstrap.md (summary), progress every 10 replicates in experiments/exp1/boot_progress.log."""
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from experiments.exp1.exp1_hide_top import BEST, PAST, candidate_inputs, combo_mean, fed_values, load_past
from models.v2.model_v2 import ALL, Model, Prep
from src.features import md

OUT = Path(__file__).resolve().parent
LOG = OUT / "boot_progress.log"
IMPUTE = BEST["missing"] == "median+flag"
SHARE = 0.10  # answers = top 10% of candidates; main metric = share of answers in the top 10% of the ranking
AUX_SHARE = 0.20
LOW_PCE = 5.0
SETTINGS = ["as designed", "D1: hide answer papers"]
SINGLE = ["v2", f"B drop PCE<{LOW_PCE:g}", "A2 within-paper ranker"]
ENSEMBLE = ["ensemble k=0", "ensemble k=1"]


def log(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S} {text}\n")


def tie_key(keys):
    return np.array([int(hashlib.sha1(k.encode()).hexdigest()[:12], 16) for k in keys])


def resample(F, rng):
    """Papers drawn with replacement; a paper drawn twice becomes two papers (DOI suffix), orig_doi keeps the source."""
    groups = F.groupby("doi").indices
    papers = np.array(list(groups))
    drawn = rng.choice(papers, len(papers), replace=True)
    parts = []
    seen = {}
    for p in drawn:
        seen[p] = seen.get(p, 0) + 1
        parts.append(F.iloc[groups[p]].assign(orig_doi=p, doi=f"{p}#{seen[p]}"))
    return pd.concat(parts, ignore_index=True)


def candidates(F):
    g = F.dropna(subset=["key"]).groupby("key").agg(n=("pce", "size"), papers=("orig_doi", "nunique"), pce=("pce", "mean"))
    return g[(g.n >= E.MIN_DEVICES) & (g.papers >= E.MIN_PAPERS)]  # copies of one paper count as one paper


def lgbm(train, y):
    return Model("lgbm", BEST["params"], IMPUTE, ALL).fit(train, y)


def ranker(train):
    t = train[train.groupby("doi").doi.transform("size") >= 2].sort_values("doi", kind="stable")
    prep = Prep(ALL, IMPUTE).fit(t)
    label = t.pce.round().clip(0, 30).astype(int)  # grade = rounded PCE; default label_gain 2^g - 1, g = 0..30
    r = LGBMRanker(objective="lambdarank", random_state=C.SEED, verbose=-1, **BEST["params"])
    r.fit(prep.transform(t), label, group=t.groupby("doi", sort=False).size().values)
    return lambda X: r.predict(prep.transform(X))


def score_methods(train, X, ensemble, rng):
    s = {"v2": combo_mean(X, lgbm(train, train.pce).predict(X))}
    t = train[train.pce >= LOW_PCE]
    s[SINGLE[1]] = combo_mean(X, lgbm(t, t.pce).predict(X))
    s[SINGLE[2]] = combo_mean(X, ranker(train)(X))
    if ensemble:
        groups = train.groupby("doi").indices
        papers = np.array(list(groups))
        preds = []
        for _ in range(E.N_BOOT):
            b = train.iloc[np.concatenate([groups[p] for p in rng.choice(papers, len(papers), replace=True)])]
            preds.append(combo_mean(X, lgbm(b, b.pce).predict(X)))
        p = pd.concat(preds, axis=1)
        s[ENSEMBLE[0]] = p.mean(axis=1)
        s[ENSEMBLE[1]] = p.mean(axis=1) + 1 * p.std(axis=1)
    return s


def metrics(score, cand, answers, tie):
    order = pd.DataFrame({"s": score.loc[cand.index], "t": tie}, index=cand.index) \
        .sort_values(["s", "t"], ascending=[False, True]).index
    n, k = len(cand), len(answers)
    is_ans = order.isin(answers)
    best = cand.loc[list(answers)].pce.idxmax()
    return {"top10_share": is_ans[:k].sum() / k,  # top 10% of ranking vs 10% answers: precision = recall
            "top20_recall": is_ans[:int(round(n * AUX_SHARE))].sum() / k,
            "best_rank_pct": (order.get_loc(best) + 1) / n}


def replicate(F, rep, ensemble, seed):
    """One replicate. rep = -1 means the original data (point estimate, no resampling)."""
    s_resample, s_model, s_random = np.random.SeedSequence(seed).spawn(3)
    D = F.assign(orig_doi=F.doi) if rep < 0 else resample(F, np.random.default_rng(s_resample))
    cand = candidates(D)
    k = int(round(len(cand) * SHARE))
    answers = set(cand.pce.nlargest(k).index)
    tie = tie_key(cand.index)
    X = candidate_inputs(D[D.key.isin(cand.index)])
    assert fed_values(X) == {v: [str(x)] for v, x in E.PREDICT_AT.items()}
    rows = []
    base = {"rep": rep, "seed": seed, "n_candidates": len(cand), "n_answers": k}
    perm = pd.Series(np.random.default_rng(s_random).permutation(len(cand)).astype(float), index=cand.index)
    rows.append({**base, "setting": "-", "method": "random (1 permutation)", **metrics(perm, cand, answers, tie)})
    answer_papers = set(D.loc[D.key.isin(answers), "orig_doi"])
    for setting in SETTINGS:
        train = D[~D.key.isin(answers)] if setting == SETTINGS[0] else D[~D.orig_doi.isin(answer_papers)]
        for name, sc in score_methods(train, X, ensemble, np.random.default_rng(s_model)).items():
            rows.append({**base, "setting": setting, "method": name, "train_rows": len(train),
                         **metrics(sc, cand, answers, tie)})
    return rows


def summarize(runs, n_ens):
    point = runs[runs.rep == -1].set_index(["setting", "method"])
    boot = runs[runs.rep >= 0]
    out = []
    for (setting, method), g in boot.groupby(["setting", "method"]):
        v = g.top10_share
        row = {"setting": setting, "method": method, "R": len(g),
               "point (original data)": point.top10_share.get((setting, method), np.nan),
               "boot mean": v.mean(), "2.5%": v.quantile(0.025), "97.5%": v.quantile(0.975),
               "x random": v.mean() / SHARE, "top20 recall mean": g.top20_recall.mean(),
               "best rank pct mean": g.best_rank_pct.mean()}
        out.append(row)
    S = pd.DataFrame(out)

    def paired(a, b, sub):
        w = sub.pivot_table(index=["rep", "setting"], columns="method", values="top10_share")
        w = w.dropna(subset=[a, b])
        d = (w[a] - w[b]).groupby(level="setting")
        return pd.DataFrame({"comparison": f"{a} - {b}", "R": d.size(), "mean diff": d.mean(),
                             "2.5%": d.quantile(0.025), "97.5%": d.quantile(0.975),
                             "share of reps higher": (w[a] > w[b]).groupby(level="setting").mean()})
    P = []
    for m in SINGLE[1:]:
        P.append(paired(m, "v2", boot[boot.setting != "-"]))  # all reps where both ran (single methods: R_single)
    for m in ENSEMBLE:
        P.append(paired(m, "v2", boot[(boot.setting != "-") & (boot.rep < n_ens)]))
    P.append(paired(ENSEMBLE[1], ENSEMBLE[0], boot[(boot.setting != "-") & (boot.rep < n_ens)]))  # pure kappa effect
    return S, pd.concat(P).reset_index()


def write_summary(R, raw, r_single, r_ens):
    S, P = summarize(R, r_ens)
    dist = R[R.rep >= 0].drop_duplicates("rep")[["n_candidates", "n_answers"]].describe().round(1)
    res = OUT / "results_bootstrap.md"
    res.write_text(
        f"# exp1 paper-level cluster bootstrap (generated by experiments/exp1/exp1_bootstrap.py)\n\n"
        f"raw: {raw.name}; R_single={r_single}, R_ensemble={r_ens}; main metric = share of the hidden top-10% "
        f"answers inside the top 10% of the ranking (random = {SHARE})\n\n"
        f"## candidates / answers per replicate\n{md(dist)}\n\n## methods (top-10% share)\n{md(S.round(3), index=False)}\n\n"
        f"## paired differences (same replicates)\n{md(P.round(3), index=False)}\n", encoding="utf-8")
    return S, P, res


def main():
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):  # rebuild the summary from saved raw runs, no training
        raw = OUT / Path(sys.argv[1]).name
        R = pd.read_csv(raw)
        _, _, res = write_summary(R, raw, R.rep.max() + 1, R[R.method == ENSEMBLE[0]].rep.max() + 1)
        print(f"summary rebuilt from {raw.name} -> {res.relative_to(ROOT)}")
        return
    r_single = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    r_ens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = OUT / f"bootstrap_runs_{stamp}.csv"
    F = load_past()
    log(f"start {stamp}: R_single={r_single}, R_ensemble={r_ens}, hyperparameters {PAST.name}, "
        f"{len(F)} rows / {F.doi.nunique()} papers <= {E.CUTOFF_YEAR}")
    t0 = time.time()
    runs = replicate(F, -1, True, E.SEED)
    log(f"original data done in {time.time() - t0:.0f}s")
    for rep in range(r_single):
        runs += replicate(F, rep, rep < r_ens, E.SEED + rep)
        pd.DataFrame(runs).to_csv(raw, index=False)  # keep raw results even if interrupted
        if (rep + 1) % 10 == 0:
            el = time.time() - t0
            log(f"replicate {rep + 1}/{r_single} done, elapsed {el / 60:.1f} min")
    S, P, res = write_summary(pd.DataFrame(runs), raw, r_single, r_ens)
    log(f"finished in {(time.time() - t0) / 60:.1f} min -> {res.name}, {raw.name}")
    print(f"done: {res.relative_to(ROOT)}, {raw.relative_to(ROOT)}")
    for _, r in S[S.setting != "-"].iterrows():
        print(f"{r.setting:24s} {r.method:24s} point {r['point (original data)']:.2f}  boot {r['boot mean']:.2f}"
              f" ({r['2.5%']:.2f}-{r['97.5%']:.2f})")


if __name__ == "__main__":
    main()
