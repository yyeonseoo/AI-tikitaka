"""Experiment 1 (corrected): find hidden high-efficiency combinations inside <= 2017 by cross-fitting
(plan: experiments/PLAN_corrected_reruns.md; replaces the first exp1 scripts, whose non-answer candidates stayed in training).
Candidates are split into 5 folds (rank-interleaved by actual efficiency); each fold is hidden in turn, every method is
trained (and tuned) on the rest and scores only the hidden fold, so no candidate is scored by a model that saw it.
Uncertainty: paper-level bootstrap of <= 2017; every replicate re-derives candidates and folds and refits everything.
Usage: python pce/experiments/exp1/exp1_crossfit.py [R=100] [workers=8]     (python ... time  -> time one replicate)
Replicates are appended to bootstrap_exp1_partial.csv, so an interrupted run resumes where it stopped.
Outputs: experiments/exp1/results_exp1.md, bootstrap_exp1_<timestamp>.csv, figures/exp1_summary.png;
progress in experiments/exp1/exp1_progress.log (not committed)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker process; workers run replicates in parallel
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from src.evaluation import (NAME, combo_table, eligible, fit_methods, fmt, load_data, metrics, paired, pin,
                            resample_papers, tie_key)
from src.features import md

OUT = Path(__file__).resolve().parent
LOG = OUT / "exp1_progress.log"
PARTIAL = OUT / "bootstrap_exp1_partial.csv"
YEAR, FOLDS = 2017, 5
SETTINGS = {"rows": "조합 행만 숨김", "papers": "조합을 낸 논문 전체 숨김"}
METRICS = ["spearman_dev", "spearman_paper", "hit_dev", "hit_paper", "gain_dev", "best_pct"]
RANDOM = {"spearman_dev": 0, "spearman_paper": 0, "hit_dev": 0.10, "hit_paper": 0.10, "gain_dev": 0, "best_pct": 0.5}


def log(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S} {text}\n")


def crossfit(D, setting, rng):
    """Score every candidate with models that never saw it. D: rows with key, pce, doi, orig, rid."""
    cand = eligible(D)
    actual = D[D.key.isin(cand)].groupby("key").pce.mean()
    order = pd.DataFrame({"p": actual, "t": tie_key(actual.index)}).sort_values(["p", "t"], ascending=[False, True]).index
    fold = pd.Series(np.arange(len(order)) % FOLDS, index=order)  # answers spread evenly over the folds
    preds, knns = [], []
    for f in range(FOLDS):
        keys = fold.index[fold == f]
        hide = D.key.isin(keys)
        if setting == "papers":
            hide = D.orig.isin(D.loc[hide, "orig"])
        train = D[~hide].reset_index(drop=True)
        ev = D[D.key.isin(keys)]
        P, knn, _ = fit_methods(train, pin(ev, YEAR), keys, rng, groups_col="orig")
        preds.append(pd.concat([ev[["key", "pce", "doi", "rid", "orig"]], P], axis=1))
        knns.append(knn)
    return combo_table(pd.concat(preds), cand), pd.concat(knns)


def replicate(rep):
    """rep = -1: original data; rep >= 0: one paper-level bootstrap replicate. Returns metric rows."""
    F = load_data(YEAR)
    base = F.assign(orig=F.doi)
    ss_boot, *ss_fit = np.random.SeedSequence([E.SEED, 1, rep + 1]).spawn(1 + len(SETTINGS))
    D = base if rep < 0 else resample_papers(base, np.random.default_rng(ss_boot))
    rows = []
    for (s, _), ss in zip(SETTINGS.items(), ss_fit):
        m, n = metrics(*crossfit(D, s, np.random.default_rng(ss)))
        rows += [{"rep": rep, "setting": s, "method": k, "n_candidates": n, **m[k]} for k in NAME]
    return rows


def report(R, stamp, raw):
    n_rep = R[R.rep >= 0].rep.nunique()
    out, figs = [], []
    for s, label in SETTINGS.items():
        Rs = R[R.setting == s]
        point = Rs[Rs.rep == -1].set_index("method")[METRICS]
        pooled = Rs[Rs.rep >= 0]
        S = []
        for m in NAME:
            g = pooled[pooled.method == m]
            S.append({"method": NAME[m], **{mt: fmt(point.loc[m, mt], g[mt].quantile(0.025), g[mt].quantile(0.975)) for mt in METRICS},
                      "boot mean (spearman_dev)": round(g.spearman_dev.mean(), 3)})
        P = paired(pooled, METRICS, point)
        main = P[P.metric == "spearman_dev"]
        out.append(f"## {label} ({s}); candidates on original data: {int(Rs[Rs.rep == -1].n_candidates.iloc[0])}\n\n"
                   f"### methods\n{md(pd.DataFrame(S), index=False)}\n\n"
                   f"### paired comparisons, main metric (spearman_dev)\n{md(main.drop(columns='metric').round(3), index=False)}\n\n"
                   f"### paired comparisons, all metrics\n{md(P.round(3), index=False)}\n")
        figs.append((label, point, pooled, main))

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for i, (label, point, pooled, main) in enumerate(figs):
        t = pooled.groupby("method").spearman_dev.agg(lo=lambda v: v.quantile(0.025), hi=lambda v: v.quantile(0.975)).reindex(list(NAME)[::-1])
        pt = point.spearman_dev.reindex(t.index)
        ax[i, 0].barh([NAME[m] for m in t.index], pt, xerr=[np.clip(pt - t.lo, 0, None), np.clip(t.hi - pt, 0, None)], capsize=3,
                      color=["#DD8452" if m == "knn" else "#4C72B0" if "full" in m or m.startswith("ens") else "#A0B4D0" for m in t.index])
        ax[i, 0].axvline(0, color="#C44E52", ls="--", lw=1)
        ax[i, 0].set(xlabel="순위 일치도 (점 = 원래 데이터, 선 = 재표집 95% 구간; 무작위 = 0)", xlim=(-0.3, 0.9), title=f"({'ab'[i]}) {label}: 방법별")
        mm = main.iloc[::-1]
        ax[i, 1].errorbar(mm.point, range(len(mm)), xerr=[np.clip(mm.point - mm["2.5%"], 0, None), np.clip(mm["97.5%"] - mm.point, 0, None)],
                          fmt="o", capsize=3, color="#4C72B0")
        ax[i, 1].axvline(0, color="#C44E52", ls="--", lw=1)
        ax[i, 1].set_yticks(range(len(mm)), [f"{q}\n{c.replace('−', '-')}" for q, c in zip(mm.question, mm.comparison)], fontsize=7)
        ax[i, 1].set(xlabel="순위 일치도 차이 (앞 - 뒤)", title=f"({'cd'[i]}) {label}: 짝지은 비교")
    fig.suptitle(f"실험 1 수정판: 2017년까지 안에서 숨긴 조합 채점 (교차 채점 5묶음, 논문 재표집 {n_rep}회)", fontsize=11)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "exp1_summary.png", dpi=120)
    plt.close(fig)

    (OUT / "results_exp1.md").write_text(
        "# experiment 1 results, corrected (generated by experiments/exp1/exp1_crossfit.py)\n\n"
        f"<= 2017 only; cross-fitting with {FOLDS} candidate folds; R={n_rep} paper-level bootstrap replicates, each refitting and "
        f"re-tuning every method; raw {raw.name}. Cells: original-data point [2.5%, 97.5%] of the replicates. "
        f"Random: {RANDOM}. best_pct = rank percentile of the best actual combination (1 = ranked first).\n\n" + "\n".join(out),
        encoding="utf-8")


def main():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    if sys.argv[1:2] == ["time"]:
        t0 = time.time()
        replicate(0)
        print(f"one replicate: {(time.time() - t0) / 60:.1f} min (single thread)")
        return
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    done = set(pd.read_csv(PARTIAL).rep) if PARTIAL.exists() else set()
    todo = [r for r in range(-1, n_rep) if r not in done]
    print(f"{len(done)} replicates already done, {len(todo)} to run with {workers} workers")
    log(f"start: R={n_rep}, todo {len(todo)}, workers {workers}")
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(replicate, r) for r in todo]
        for i, f in enumerate(as_completed(futures), 1):
            pd.DataFrame(f.result()).to_csv(PARTIAL, mode="a", header=not PARTIAL.exists(), index=False)
            if i % 10 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} replicates, {(time.time() - t0) / 60:.1f} min")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    R = pd.read_csv(PARTIAL)
    R = R[R.rep < n_rep]
    raw = OUT / f"bootstrap_exp1_{stamp}.csv"
    R.to_csv(raw, index=False)
    PARTIAL.unlink()
    report(R, stamp, raw)
    log(f"finished in {(time.time() - t0) / 60:.1f} min")
    print(open(OUT / "results_exp1.md", encoding="utf-8").read().split("### paired comparisons, all")[0][:3000])


if __name__ == "__main__":
    main()
