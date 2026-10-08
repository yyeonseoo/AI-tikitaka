"""Experiment 3, phase 1 (corrected): matched-input comparison on temporal backtests inside <= 2017
(plan: experiments/exp3/PLAN_phase1.md; replaces the first phase-1 script, which used hyperparameters tuned on
data that included its evaluation years).
For each evaluation year Y in 2015-2017: train on papers before Y (hyperparameters chosen inside those years only),
score Y's process combinations, compare rank agreement with the actual combination efficiency.
Usage: python pce/experiments/exp3/exp3_backtest.py [R=1000]
Outputs: experiments/exp3/results_backtest.md, bootstrap_backtest_<timestamp>.csv, figures/backtest_summary.png;
progress in experiments/exp3/backtest_progress.log (not committed)."""
import sys
import time
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
from src.evaluation import (CFG_MIN, NAME, combo_table, config_label, config_spearman, eligible,
                            fit_methods, fmt, load_data, metrics, paired, pin, resample_papers)
from src.features import md

OUT = Path(__file__).resolve().parent
LOG = OUT / "backtest_progress.log"
YEARS = [2015, 2016, 2017]
METRICS = ["spearman_dev", "spearman_paper", "hit_dev", "gain_dev"]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def log(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S} {text}\n")
    print(text, flush=True)


def run_year(F, year, n_rep):
    ss_fit, ss_boot = np.random.SeedSequence([E.SEED, year]).spawn(2)
    train = F[F.year < year].reset_index(drop=True)
    ev = F[F.year == year].dropna(subset=["key"]).reset_index(drop=True)
    t0 = time.time()
    P, knn, chosen = fit_methods(train, pin(ev, train.year.max()), ev.key.unique(), np.random.default_rng(ss_fit))
    log(f"{year}: trained on {len(train)} rows ({train.doi.nunique()} papers) in {time.time() - t0:.0f}s; chosen {chosen}")
    base = pd.concat([ev[["key", "pce", "doi", "rid"]].assign(orig=ev.doi, cfg=config_label(ev)), P], axis=1)
    fixed = eligible(base)
    cfg_keys = {}
    for cfg, g in base.groupby("cfg"):
        k = eligible(g)
        if len(k) >= CFG_MIN:
            cfg_keys[cfg] = k
    rows = []

    def record(rep, R):
        m_fixed, n_fixed = metrics(combo_table(R, fixed), knn)
        m_re, n_re = metrics(combo_table(R, eligible(R)), knn)
        cs = config_spearman(R, cfg_keys)
        for m in NAME:
            rows.append({"year": year, "rep": rep, "method": m, "n_fixed": n_fixed, "n_rederived": n_re,
                         **m_fixed[m], "spearman_rederived": m_re[m]["spearman_dev"],
                         "spearman_config": cs.get(m, np.nan), "n_config_groups": cs.get("n_groups", 0)})
    record(-1, base)
    rng = np.random.default_rng(ss_boot)
    for rep in range(n_rep):
        record(rep, resample_papers(base, rng))
        if (rep + 1) % 100 == 0:
            log(f"  {year}: resample {rep + 1}/{n_rep}, {(time.time() - t0) / 60:.1f} min")
    info = {"train_rows": len(train), "train_papers": train.doi.nunique(), "eval_rows": len(ev), "eval_papers": ev.doi.nunique(),
            "fixed_candidates": len(fixed), "config_groups": len(cfg_keys), "chosen": str(chosen)}
    return rows, info


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    F = load_data(2017)  # phase 1 must not touch 2018+
    t0 = time.time()
    log(f"start {stamp}: R={n_rep}, {len(F)} rows <= 2017")
    runs, info = [], {}
    for y in YEARS:
        r, inf = run_year(F, y, n_rep)
        runs += r
        info[y] = inf
    R = pd.DataFrame(runs)
    raw = OUT / f"bootstrap_backtest_{stamp}.csv"
    R.to_csv(raw, index=False)

    allm = METRICS + ["spearman_rederived", "spearman_config"]
    point_y = R[R.rep == -1].set_index(["year", "method"])
    point = R[R.rep == -1].groupby("method")[allm].mean()  # average of the three years
    pooled = R[R.rep >= 0].groupby(["rep", "method"])[allm].mean().reset_index()

    S = []
    for m in NAME:
        g = pooled[pooled.method == m]
        S.append({"method": NAME[m], **{mt: fmt(point.loc[m, mt], g[mt].quantile(0.025), g[mt].quantile(0.975)) for mt in allm},
                  "boot mean (spearman_dev)": round(g.spearman_dev.mean(), 3)})
    S = pd.DataFrame(S)
    Y = []
    for y in YEARS:
        for m in NAME:
            g = R[(R.year == y) & (R.rep >= 0) & (R.method == m)]
            Y.append({"year": y, "method": NAME[m], "spearman_dev": fmt(point_y.loc[(y, m), "spearman_dev"],
                                                                         g.spearman_dev.quantile(0.025), g.spearman_dev.quantile(0.975))})
    Y = pd.DataFrame(Y).pivot(index="method", columns="year", values="spearman_dev").reindex([NAME[m] for m in NAME])

    P = paired(pooled, allm, point)
    main = P[P.metric == "spearman_dev"]

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    t = pooled.groupby("method").spearman_dev.agg(lo=lambda v: v.quantile(0.025), hi=lambda v: v.quantile(0.975)).reindex(list(NAME)[::-1])
    pt = point.spearman_dev.reindex(t.index)
    ax[0].barh([NAME[m] for m in t.index], pt, xerr=[np.clip(pt - t.lo, 0, None), np.clip(t.hi - pt, 0, None)], capsize=3,  # point can sit outside the resample band
               color=["#DD8452" if m == "knn" else "#4C72B0" if "full" in m or m.startswith("ens") else "#A0B4D0" for m in t.index])
    ax[0].axvline(0, color="#C44E52", ls="--", lw=1)
    ax[0].set(xlabel="순위 일치도 (세 해 평균; 점 = 원래 데이터, 선 = 재표집 95% 구간; 무작위 = 0)", xlim=(-0.2, 0.9),
              title="(a) 방법별 순위 일치도")
    mm = main.iloc[::-1]
    ax[1].errorbar(mm.point, range(len(mm)), xerr=[np.clip(mm.point - mm["2.5%"], 0, None), np.clip(mm["97.5%"] - mm.point, 0, None)], fmt="o", capsize=3,
                   color="#4C72B0")
    ax[1].axvline(0, color="#C44E52", ls="--", lw=1)
    ax[1].set_yticks(range(len(mm)), [f"{q}\n{c.replace('−', '-')}" for q, c in zip(mm.question, mm.comparison)],
                     fontsize=7)  # the Korean font has no U+2212 minus sign
    ax[1].set(xlabel="순위 일치도 차이 (앞 - 뒤)", title="(b) 짝지은 비교 (구간이 0을 넘으면 차이 확인)")
    fig.suptitle(f"실험 3 1단계: 전 해까지 학습 → 다음 해 평가 (2015·2016·2017, 연도별 논문 재표집 {n_rep}회)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "backtest_summary.png", dpi=120)
    plt.close(fig)

    (OUT / "results_backtest.md").write_text(
        "# experiment 3 phase 1 results (generated by experiments/exp3/exp3_backtest.py)\n\n"
        f"<= 2017 only; R={n_rep} per year; raw {raw.name}. Cells: original-data point (mean of the three years) "
        "[2.5%, 97.5%] of the pooled resamples. spearman_rederived = candidates re-derived per resample (sensitivity); "
        "spearman_config = fixed device configuration (exploratory).\n\n"
        f"## folds\n{md(pd.DataFrame(info).T)}\n\n## methods (pooled over 2015-2017)\n{md(S, index=False)}\n\n"
        f"## rank agreement by year\n{md(Y)}\n\n"
        f"## paired comparisons, main metric (spearman_dev)\n{md(main.drop(columns='metric').round(3), index=False)}\n\n"
        f"## paired comparisons, all metrics\n{md(P.round(3), index=False)}\n", encoding="utf-8")
    log(f"finished in {(time.time() - t0) / 60:.1f} min")
    print(md(main[["comparison", "point", "2.5%", "97.5%", "judgement"]].round(3), index=False))


if __name__ == "__main__":
    main()
