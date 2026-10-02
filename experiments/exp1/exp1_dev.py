"""Experiment 1, round 2: same setup as exp1.py (<= 2017, hide top 5/10/20%, past-only hyperparameters,
same candidate inputs with year 2017 / scan Reverse) with more ranking methods, leakage settings D1/D2,
combination-bootstrap 95% CIs of unique hits and paired differences against v2 on the same resamples.
Usage: python experiments/exp1/exp1_dev.py   (tables -> experiments/exp1/results_dev.md, plot -> figures/)"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from experiments.exp1.exp1 import (BEST, PAST, candidate_inputs, candidates, combo_mean, evaluate, fed_values, load_past,
                                   rank, streams)
from models.v2.model_v2 import ALL, Model, Prep
from src.features import md

OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results_dev.md"
IMPUTE = BEST["missing"] == "median+flag"
GP_ROWS = 3000
N_CI = 1000
LOW_PCE = 5.0
REF = "LGBM mean (k=0)"  # paired differences are taken against this (= v2)
SETTINGS = ["as designed", "D1: hide answer papers", "D2: conditions fixed"]


def out(text="", console=False):
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(str(text) + "\n")
    if console:
        print(text, flush=True)


def lgbm(train, y, features=ALL):
    return Model("lgbm", BEST["params"], IMPUTE, features).fit(train, y)


# ---- methods: each takes training rows and returns {name: scorer}; scorer(X) -> score per combination,
#      where X always comes from candidate_inputs (one input builder for every method) ----
def m_lgbm(train, rng):
    m = lgbm(train, train.pce)
    return {REF: lambda X: combo_mean(X, m.predict(X))}


def m_boot(train, rng):
    groups = train.groupby("doi").indices
    papers = np.array(list(groups))
    ms = []
    for _ in range(E.N_BOOT):
        b = train.iloc[np.concatenate([groups[p] for p in rng.choice(papers, len(papers), replace=True)])]
        ms.append(lgbm(b, b.pce))

    def score(X):
        p = pd.concat([combo_mean(X, m.predict(X)) for m in ms], axis=1)
        return p.mean(axis=1) + 1 * p.std(axis=1)
    return {"LGBM boot k=1": score}


def multi_paper(train):
    return train[train.groupby("doi").doi.transform("size") >= 2]


def m_relative(train, rng):
    t = multi_paper(train)
    m = lgbm(t, t.pce - t.groupby("doi").pce.transform("median"))  # within-paper relative efficiency
    return {"A1 within-paper relative": lambda X: combo_mean(X, m.predict(X))}


def m_ranker(train, rng):
    t = multi_paper(train).sort_values("doi", kind="stable")
    prep = Prep(ALL, IMPUTE).fit(t)
    label = t.pce.round().clip(0, 30).astype(int)  # relevance grade; only compared inside each paper
    r = LGBMRanker(objective="lambdarank", random_state=C.SEED, verbose=-1, **BEST["params"])
    r.fit(prep.transform(t), label, group=t.groupby("doi", sort=False).size().values)
    return {"A2 within-paper ranker": lambda X: combo_mean(X, r.predict(prep.transform(X)))}


def m_no_low(train, rng):
    t = train[train.pce >= LOW_PCE]
    m = lgbm(t, t.pce)
    return {f"B drop PCE<{LOW_PCE:g}": lambda X: combo_mean(X, m.predict(X))}


def m_gp(train, rng):
    s = train.sample(min(GP_ROWS, len(train)), random_state=C.SEED)  # ponytail: exact GP is O(n^3); 3000-row sample
    prep = Prep(ALL, impute=True).fit(s)  # GP needs no NaN: median fill + missing flags

    def enc(X):
        t = prep.transform(X)
        return pd.get_dummies(t, columns=[c for c in t if isinstance(t[c].dtype, pd.CategoricalDtype)], dtype=float)
    Xs = enc(s)
    mu, sd = Xs.mean(), Xs.std().replace(0, 1)
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(3.0) + WhiteKernel(1.0), normalize_y=True,
                                  random_state=C.SEED).fit(((Xs - mu) / sd).values, s.pce.values)

    def pred(X):
        return gp.predict(((enc(X).reindex(columns=Xs.columns, fill_value=0) - mu) / sd).values, return_std=True)
    return {"C GP mean": lambda X: combo_mean(X, pred(X)[0]),
            "C GP mean+1sd": lambda X: combo_mean(X, np.add(*pred(X)))}


METHODS = [m_lgbm, m_boot, m_relative, m_ranker, m_no_low, m_gp]


def joint_bootstrap(scores, c, answers, rng):
    """Resample candidate combinations once per replicate and score every method on that same resample.
    Hits = number of distinct answers among the top N slots (a duplicated answer counts once)."""
    keys = np.array(c.index)
    tie = c.tie.values
    is_ans = np.isin(keys, list(answers))
    S = {m: s.loc[keys].values for m, s in scores.items()}
    hits = {m: {n: np.empty(N_CI) for n in E.TOP_NS} for m in S}
    for r in range(N_CI):
        i = rng.integers(0, len(keys), len(keys))
        for m, s in S.items():
            o = i[np.lexsort((tie[i], -s[i]))]  # score desc, fixed tie key
            for n in E.TOP_NS:
                top = o[:n]
                hits[m][n][r] = len(set(top[is_ans[top]]))
    return hits


def fmt(v, lo, hi, digits=0):
    return f"{v:.{digits}f} ({lo:.0f} to {hi:.0f})" if digits == 0 else f"{v:.{digits}f} ({lo:.1f} to {hi:.1f})"


def main():
    RESULTS.write_text("# exp1 round 2 results (generated by experiments/exp1/exp1_dev.py)\n\n", encoding="utf-8")
    rs = streams()
    F = load_past()
    c = candidates(F)
    rows = F[F.key.isin(c.index)]
    out(f"hyperparameters: {PAST.relative_to(ROOT)}; candidates: {len(c)} (rows {len(rows)}); "
        f"years {F.year.min():.0f}-{F.year.max():.0f}", console=True)

    tables, sums, curves = [], {}, {}
    for share in E.HIDE_SHARES:
        k = int(round(len(c) * share))
        answers = set(c.pce.nlargest(k).index)
        best = c.loc[list(answers)].pce.idxmax()
        base = F[~F.key.isin(answers)]
        strict = F[~F.doi.isin(set(F[F.key.isin(answers)].doi))]
        common = {v: base[v].mode().iloc[0] for v in C.CONDITION}
        perms = [evaluate(list(rs["random"].permutation(c.index)), answers, best) for _ in range(E.N_RANDOM)]
        tables.append({"hidden": f"{share:.0%}", "setting": "-", "method": "random (1000)",
                       **{f"hits@{n}": fmt(np.mean([p[f'hits@{n}'] for p in perms]),
                                           *np.percentile([p[f"hits@{n}"] for p in perms], [2.5, 97.5]), 1)
                          for n in E.TOP_NS},
                       "best rank": round(np.mean([p["best_answer_rank"] for p in perms]), 1), "diff@20 vs v2": "",
                       "_h20": np.mean([p["hits@20"] for p in perms])})
        fitted = {}
        for setting, train in [(SETTINGS[0], base), (SETTINGS[1], strict), (SETTINGS[2], None)]:
            if train is not None:  # D2 reuses the as-designed models; only the candidate inputs change
                t0 = time.time()
                fitted = {}
                for m in METHODS:
                    fitted |= m(train, rs["boot"])
                print(f"  hide {share:.0%} / {setting}: fitted in {time.time() - t0:.0f}s", flush=True)
                if setting == SETTINGS[0]:
                    fitted_base = fitted
            else:
                fitted = fitted_base
            X = candidate_inputs(rows, common if setting == SETTINGS[2] else None)
            fed = fed_values(X)
            assert fed == {k_: [str(v)] for k_, v in E.PREDICT_AT.items()}, fed
            out(f"hide {share:.0%} / {setting}: values fed to every method: {fed}")
            scores = {name: scorer(X).loc[c.index] for name, scorer in fitted.items()}
            hits = joint_bootstrap(scores, c, answers, rs["ci"])
            for name, sc in scores.items():
                r = evaluate(rank(sc, c.tie), answers, best)
                d20 = hits[name][20] - hits[REF][20]
                sums.setdefault((setting, name), {"point": 0, "draws": np.zeros(N_CI)})
                sums[(setting, name)]["point"] += r["hits@20"]
                sums[(setting, name)]["draws"] += d20
                tables.append({"hidden": f"{share:.0%}", "setting": setting, "method": name,
                               **{f"hits@{n}": fmt(r[f"hits@{n}"], *np.percentile(hits[name][n], [2.5, 97.5]))
                                  for n in E.TOP_NS},
                               "best rank": r["best_answer_rank"],
                               "diff@20 vs v2": "" if name == REF else fmt(d20.mean(), *np.percentile(d20, [2.5, 97.5]), 1),
                               "_h20": r["hits@20"]})
                curves[(share, setting, name)] = r["curve"]

    R = pd.DataFrame(tables)
    show = [col for col in R.columns if not col.startswith("_")]
    out("\nhits@N = distinct answers in the top N (95% CI from 1000 combination resamples);"
        " diff@20 vs v2 = paired difference on the same resamples (mean, 95% CI)")
    for share in E.HIDE_SHARES:
        h = f"{share:.0%}"
        for setting in ["-"] + SETTINGS:
            t = R[(R.hidden == h) & (R.setting == setting)]
            if len(t):
                out(f"\n## hide top {h} — {setting if setting != '-' else 'random baseline'}\n"
                    + md(t[show].drop(columns=["hidden", "setting"]), index=False))

    rnd = R[R.setting == "-"]["_h20"].sum()
    rows_s = []
    for (setting, name), v in sums.items():
        d = v["draws"]
        rows_s.append({"setting": setting, "method": name, "hits@20 summed": v["point"],
                       "diff vs v2 (summed, 95% CI)": "" if name == REF else fmt(d.mean(), *np.percentile(d, [2.5, 97.5]), 1),
                       "P(diff > 0)": "" if name == REF else f"{(d > 0).mean():.2f}"})
    S = pd.DataFrame(rows_s)
    out(f"\n## hits@20 summed over hide 5/10/20% (answers 10+20+40; random {rnd:.1f})\n" + md(S, index=False))

    out("\n=== exp1 round 2 summary: hits@20 summed over hide 5/10/20% "
        f"(random {rnd:.1f}); paired diff vs v2 with 95% CI ===", console=True)
    piv = S.pivot(index="method", columns="setting", values="hits@20 summed")[SETTINGS]
    dif = S.pivot(index="method", columns="setting", values="diff vs v2 (summed, 95% CI)")[SETTINGS]
    for m in piv.index:
        out(f"{m:26s} " + "  ".join(f"{s.split(':')[0]:>11s} {piv.loc[m, s]:>3.0f} {dif.loc[m, s]:>18s}" for s in SETTINGS),
            console=True)

    fig, axes = plt.subplots(1, len(E.HIDE_SHARES), figsize=(5 * len(E.HIDE_SHARES), 4))
    for ax, share in zip(axes, E.HIDE_SHARES):
        for (sh, setting, name), cv in curves.items():
            if sh == share and setting == SETTINGS[1]:
                ax.plot(np.arange(1, len(cv) + 1), cv, label=name, lw=1.3)
        k = int(round(len(c) * share))
        ax.plot([0, len(c)], [0, k], "k--", lw=1, label="random")
        ax.set_xlim(0, 60)
        ax.set_ylim(0, k + 0.5)
        ax.set_title(f"hide top {share:.0%} ({k}), D1: answer papers hidden", fontsize=9)
        ax.set_xlabel("top N recommended")
        ax.set_ylabel("answers found")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "hits_curve_dev.png", dpi=110)
    plt.close(fig)
    out(f"details: {RESULTS.relative_to(ROOT)}", console=True)


if __name__ == "__main__":
    main()
