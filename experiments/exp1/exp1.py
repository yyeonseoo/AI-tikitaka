"""Experiment 1: hide the best process combinations from training and check whether each ranking
method puts them near the top. Uses only papers up to 2017 and hyperparameters chosen on that period
(models/v2/best_params_past.json). Also estimates a noise ceiling inside the same period.
Usage: python experiments/exp1/exp1.py
Console gets a summary; full tables go to experiments/exp1/results.md, plots to experiments/exp1/figures/."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from models.v2.model_v2 import ALL, Model, clean, combo_key, make_features
from src.data import split_from_file
from src.features import md

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
RESULTS = OUT / "results.md"
PAST = ROOT / "models" / "v2" / "best_params_past.json"  # chosen on <= 2017 only; best_params.json is never used here
if not PAST.exists():
    raise SystemExit(f"{PAST.relative_to(ROOT)} missing: run python models/v2/tune_past.py first")
BEST = json.loads(PAST.read_text(encoding="utf-8"))
PARTS = ["solvent", "dmso", "antisolvent", "additive", "solvent_annealing", "temp", "time"]  # order used in combo_key
DMSO_ORDER = {"0": 0, "0-0.35": 1, "0.35-0.9": 2, "0.9-1": 3}


def out(text="", console=False):
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(str(text) + "\n")
    if console:
        print(text, flush=True)


def streams():
    """Independent random streams: paper bootstrap / random baseline / CI resampling. Models use C.SEED."""
    boot, rand, ci = (np.random.default_rng(s) for s in np.random.SeedSequence(E.SEED).spawn(3))
    return {"boot": boot, "random": rand, "ci": ci}


def v2_model(features=ALL):
    return Model(BEST["model"], BEST["params"], BEST["missing"] == "median+flag", features)


def load_past():
    """<= 2017 rows only; nothing after the cutoff ever reaches a model or a score."""
    F = make_features(clean()[0])
    F["key"] = combo_key(F)
    F = F[F.year <= E.CUTOFF_YEAR].reset_index(drop=True)
    assert F.year.max() <= E.CUTOFF_YEAR
    return F


def candidates(F):
    g = F.dropna(subset=["key"]).groupby("key").agg(n=("pce", "size"), papers=("doi", "nunique"), pce=("pce", "mean"))
    c = g[(g.n >= E.MIN_DEVICES) & (g.papers >= E.MIN_PAPERS)].copy()
    parts = c.index.to_series().str.split(" / ", expand=True)
    assert parts.shape[1] == len(PARTS), "a category value contains ' / '"
    c[PARTS] = parts.values
    c["allowed"] = (c.dmso != E.EXCLUDE_DMSO_BIN) & c.temp.astype(float).between(*E.TEMP_RANGE)
    c["tie"] = [int(hashlib.sha1(k.encode()).hexdigest()[:12], 16) for k in c.index]  # fixed tie-break per combination
    return c


def candidate_inputs(rows, conditions=None):
    """The one place candidate rows become model inputs: real rows of each combination, correction variables
    pinned (year 2017, scan Reverse), optionally condition variables overwritten (diagnostic D2)."""
    X = rows.assign(**E.PREDICT_AT)
    return X.assign(**conditions) if conditions else X


def fed_values(X):
    return {v: sorted(map(str, X[v].unique())) for v in E.PREDICT_AT}


def combo_mean(X, pred):
    return pd.Series(pred, index=X.index).groupby(X["key"]).mean()


def follow_best(c, ref):
    """Distance on process variables to the nearest reference (best known) combination; smaller = better."""
    num = pd.DataFrame({"dmso": c.dmso.map(DMSO_ORDER) / 3,
                        "temp": c.temp.astype(float), "time": c.time.astype(float)})
    for k in ("temp", "time"):
        num[k] = (num[k] - num[k].min()) / max(num[k].max() - num[k].min(), 1)
    cat = c[["solvent", "antisolvent", "additive", "solvent_annealing"]]
    d = [((cat != cat.loc[r]).sum(axis=1) + (num - num.loc[r]).abs().sum(axis=1)) for r in ref]
    return -pd.concat(d, axis=1).min(axis=1)


def rank(score, tie):
    """Highest score first; ties broken by the fixed per-combination key."""
    return list(pd.DataFrame({"s": score, "t": tie.loc[score.index]}).sort_values(["s", "t"], ascending=[False, True]).index)


def evaluate(order, answers, best_answer):
    hits = np.cumsum([k in answers for k in order])
    return {f"hits@{n}": int(hits[min(n, len(order)) - 1]) for n in E.TOP_NS} | {
        "best_answer_rank": order.index(best_answer) + 1, "curve": hits}


def random_baseline(keys, answers, best, rng):
    rand = [evaluate(list(rng.permutation(keys)), answers, best) for _ in range(E.N_RANDOM)]
    return {key: np.mean([x[key] for x in rand], axis=0) for key in rand[0]}


def run_share(F, c, share, rs):
    k = int(round(len(c) * share))
    answers = set(c.pce.nlargest(k).index)
    train = F[~F.key.isin(answers)]
    X = candidate_inputs(F[F.key.isin(c.index)])
    scores = {"v2 mean": combo_mean(X, v2_model().fit(train, train.pce).predict(X))}
    groups = train.groupby("doi").indices
    papers = np.array(list(groups))
    boot = []
    for _ in range(E.N_BOOT):
        idx = np.concatenate([groups[p] for p in rs["boot"].choice(papers, len(papers), replace=True)])
        b = train.iloc[idx]
        boot.append(combo_mean(X, v2_model().fit(b, b.pce).predict(X)))
    boot = pd.concat(boot, axis=1)
    for kappa in E.KAPPAS:
        scores[f"v2 + {kappa}*sd"] = boot.mean(axis=1) + kappa * boot.std(axis=1)
    known = c[~c.index.isin(answers)]
    scores["follow best (invalid, see report)"] = follow_best(c, known.pce.nlargest(k).index)

    res = {}
    for constrained in (False, True):
        cc = c[c.allowed] if constrained else c
        ans = answers & set(cc.index)
        if not ans:
            continue
        best = cc.loc[list(ans)].pce.idxmax()
        r = {"random (mean of 1000)": random_baseline(list(cc.index), ans, best, rs["random"])}
        for name, s in scores.items():
            r[name] = evaluate(rank(s.loc[cc.index], cc.tie), ans, best)
        res[constrained] = (r, len(cc), len(ans))
    return k, res, scores, boot, fed_values(X)


def diagnostics(F, c, share):
    """Leakage checks for 'v2 mean' (not part of the requested design):
    D1) hide every row of the papers behind the answers, not just the answer combinations;
    D2) score all candidates with the same (most common) condition variables, so only process variables differ."""
    k = int(round(len(c) * share))
    answers = set(c.pce.nlargest(k).index)
    best = c.loc[list(answers)].pce.idxmax()
    rows = F[F.key.isin(c.index)]
    base = F[~F.key.isin(answers)]
    papers = set(F[F.key.isin(answers)].doi)
    strict = F[~F.doi.isin(papers)]
    common = {v: base[v].mode().iloc[0] for v in C.CONDITION}
    res = {}
    for name, train, X in [("as designed", base, candidate_inputs(rows)),
                           ("D1: hide answer papers", strict, candidate_inputs(rows)),
                           ("D2: conditions fixed to most common", base, candidate_inputs(rows, common))]:
        r = evaluate(rank(combo_mean(X, v2_model().fit(train, train.pce).predict(X)), c.tie), answers, best)
        res[name] = {"train rows": len(train), **{key: v for key, v in r.items() if key != "curve"}}
    return pd.DataFrame(res).T, len(papers), common


def noise_ceiling(F):
    """Inside <= 2017 only. Train/test roles from splits/doi_split.csv. For test rows whose combination also appears
    in train papers: predict PCE as the mean of the train papers' devices in that combination; compare with v2
    (past hyperparameters) trained on the same train papers, on the same rows."""
    tr_i, te_i = split_from_file(F["doi"])
    tr, te = F.iloc[tr_i], F.iloc[te_i].copy()
    te["v2"] = v2_model().fit(tr, tr.pce).predict(te)
    te["v2_process"] = v2_model(C.PROCESS).fit(tr, tr.pce).predict(te)
    other = tr.dropna(subset=["key"]).groupby("key").pce.agg(["mean", "count"])
    t = te.dropna(subset=["key"]).join(other, on="key")
    t = t[t["count"] > 0]
    rows = []
    for name, p in [("train mean", np.full(len(t), tr.pce.mean())), ("v2 (process variables only)", t.v2_process),
                    ("v2", t.v2), ("same combo, train papers' mean", t["mean"])]:
        rows.append((name, len(t), t.doi.nunique(), mean_absolute_error(t.pce, p), r2_score(t.pce, p)))
    v2_all = (mean_absolute_error(te.pce, te.v2), r2_score(te.pce, te.v2), len(te))
    return pd.DataFrame(rows, columns=["predictor", "test rows", "papers", "MAE", "R2"]), v2_all, t


def main():
    FIG.mkdir(exist_ok=True)
    RESULTS.write_text("# exp1 results (generated by experiments/exp1/exp1.py)\n\n", encoding="utf-8")
    rs = streams()
    F = load_past()
    c = candidates(F)
    out(f"hyperparameters: {PAST.relative_to(ROOT)} ({BEST['model']}, {BEST['missing']}, {BEST['params']})", console=True)
    out(f"rows <= {E.CUTOFF_YEAR}: {len(F)} ({F.doi.nunique()} papers, years {F.year.min():.0f}-{F.year.max():.0f});"
        f" candidates (>= {E.MIN_DEVICES} devices, >= {E.MIN_PAPERS} papers): {len(c)};"
        f" with chemistry constraint: {c.allowed.sum()}", console=True)
    if len(c) < E.MIN_CANDIDATES:
        out(f"fewer than {E.MIN_CANDIDATES} candidates: stopping", console=True)
        return
    out("candidate mean PCE: " + ", ".join(f"{q} {v:.1f}" for q, v in c.pce.quantile([0.5, 0.8, 0.9, 0.95]).items()))

    summary, curves = [], {}
    for share in E.HIDE_SHARES:
        k, res, scores, boot, fed = run_share(F, c, share, rs)
        out(f"\n## hide top {share:.0%} ({k} combinations, mean PCE >= {c.pce.nlargest(k).min():.1f}%)")
        out(f"values fed to the models for candidates: {fed}")
        for constrained, (r, n_c, n_ans) in res.items():
            label = "with chemistry constraint" if constrained else "no constraint"
            rand = r["random (mean of 1000)"]
            tab = pd.DataFrame([{
                "method": name, **{f"hits@{n}": round(float(v[f"hits@{n}"]), 2) for n in E.TOP_NS},
                **{f"x random @{n}": round(float(v[f"hits@{n}"]) / float(rand[f"hits@{n}"]), 2) for n in E.TOP_NS},
                "best answer rank": round(float(v["best_answer_rank"]), 1)} for name, v in r.items()])
            out(f"\n### {label}: {n_c} candidates, {n_ans} answers\n" + md(tab, index=False))
            summary.append(tab.assign(hidden=f"{share:.0%}", constraint=constrained))
            curves[(share, constrained)] = (r, n_c, n_ans)
        top = c.pce.nlargest(k)
        out(f"\nSpearman(v2 predicted, actual mean PCE) over all candidates: {scores['v2 mean'].corr(c.pce, method='spearman'):.2f};"
            f" over hidden answers only: {scores['v2 mean'].loc[top.index].corr(top, method='spearman'):.2f}")
        out(f"bootstrap sd per combination: median {boot.std(axis=1).median():.2f}, answers {boot.std(axis=1).loc[top.index].median():.2f}")

    fig, axes = plt.subplots(2, len(E.HIDE_SHARES), figsize=(5 * len(E.HIDE_SHARES), 8))
    for j, share in enumerate(E.HIDE_SHARES):
        for i, constrained in enumerate((False, True)):
            ax = axes[i, j]
            if (share, constrained) not in curves:
                ax.axis("off")
                continue
            r, n_c, n_ans = curves[(share, constrained)]
            x = np.arange(1, n_c + 1)
            for name, v in r.items():
                ax.plot(x, v["curve"], lw=2.2 if name.startswith("random") else 1.4,
                        ls="--" if name.startswith("random") else "-", label=name,
                        color="gray" if name.startswith("random") else None)
            ax.set_xlim(0, min(60, n_c))
            ax.set_ylim(0, n_ans + 0.5)
            ax.set_title(f"hide top {share:.0%} ({n_ans} answers), {'constrained' if constrained else 'no constraint'}",
                         fontsize=9)
            ax.set_xlabel("top N recommended")
            ax.set_ylabel("answers found")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "hits_curve.png", dpi=110)
    plt.close(fig)

    dg, n_papers, common = diagnostics(F, c, 0.10)
    out(f"\n## diagnostics: hide top 10%, 'v2 mean' (answers come from {n_papers} papers)\n" + md(dg))
    out(f"D2 uses: { {k: str(v) for k, v in common.items()} }")

    nc, v2_all, t = noise_ceiling(F)
    out(f"\n## noise ceiling inside <= {E.CUTOFF_YEAR} (train/test roles from splits/doi_split.csv;"
        f" 'other papers' = train papers only)\n" + md(nc.round(3), index=False))
    out(f"v2 on all <= {E.CUTOFF_YEAR} test rows ({v2_all[2]}): MAE {v2_all[0]:.3f}, R2 {v2_all[1]:.3f}")
    out(f"train-paper devices per test row's combination: median {t['count'].median():.0f}")

    s = pd.concat(summary)
    h = s[(s.hidden == "10%") & (~s.constraint)].set_index("method")
    out("\n=== exp1 summary (hide top 10%, no constraint) ===", console=True)
    for m, v in h.iterrows():
        out(f"{m:36s} hits@10 {v['hits@10']:>5}  hits@20 {v['hits@20']:>5}  hits@50 {v['hits@50']:>5}"
            f"  best answer rank {v['best answer rank']}", console=True)
    for name, v in dg.iterrows():
        out(f"diagnostic {name:36s} hits@20 {v['hits@20']}  best answer rank {v['best_answer_rank']}", console=True)
    n = nc.set_index("predictor")
    out(f"noise ceiling (<= {E.CUTOFF_YEAR}, {nc['test rows'].iloc[0]} test rows): train-paper combo mean R2 {n.R2.iloc[3]:.2f},"
        f" v2 process-only R2 {n.R2.iloc[1]:.2f}, v2 R2 {n.R2.iloc[2]:.2f}", console=True)
    out(f"details: {RESULTS.relative_to(ROOT)}  figure: {(FIG / 'hits_curve.png').relative_to(ROOT)}", console=True)


if __name__ == "__main__":
    main()
