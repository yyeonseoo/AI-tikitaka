"""Experiment 3, phase 1: rolling backtests inside <= 2017 to try graded ranking metrics before any new plan
touches 2018+. Folds: train <= Y-1, evaluate year Y, for Y = 2015, 2016, 2017.
Metrics use every candidate combination (not only the top-10% answers):
  spearman   rank agreement between the model score and the actual combination mean PCE (random = 0)
  captured   (actual mean PCE of the recommended top 10% - candidates' mean) / (best possible top 10% - candidates' mean)
             (random = 0, perfect = 1)
  rec_pce    actual mean PCE of the recommended top 10% (for reading in %)
  ndcg       NDCG over the top 10% with relevance = actual mean PCE - lowest candidate mean (random ~ its own level)
  hit        old metric: share of the actual top 10% inside the recommended top 10%
  near_miss  mean process distance from each recommended combination to the closest actual top-10% combination
Usage: python experiments/exp3/exp3_backtest.py   (-> experiments/exp3/results_backtest.md, bootstrap_backtest_<ts>.csv,
figures/backtest_summary.png). Nothing after 2017 is loaded."""
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from experiments.exp1.exp1_bootstrap import tie_key
from experiments.exp1.exp1_bootstrap_baselines import parts, ridge
from experiments.exp1.exp1_hide_top import DMSO_ORDER
from experiments.exp2.exp2_future import PROC, knn_score, lgbm
from models.v2.model_v2 import clean, combo_key, make_features
from src.features import md

OUT = Path(__file__).resolve().parent
EVAL_YEARS = [2015, 2016, 2017]
SHARE, N_REP, HIGH = 0.10, 500, 15.0
METRICS = ["spearman", "captured", "rec_pce", "ndcg", "hit", "near_miss"]
NAME = {"lgbm": "LightGBM", "lgbm_detrended": "LightGBM, 연도 보정 타깃", "lgbm_process": "LightGBM, 공정+보정",
        "ridge": "선형 회귀", "knn": "비슷한 조합 평균", "random": "무작위"}
if "Malgun Gothic" in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class Detrended:
    """LightGBM on PCE minus the median PCE of the same publication year (removes the year-level trend)."""

    def fit(self, train):
        y = train.pce - train.groupby("year").pce.transform("median")
        self.m = lgbm().fit(train, y)
        return self

    def predict(self, X):
        return self.m.predict(X)


def process_distance(keys):
    p = parts(pd.Index(keys))
    D = np.zeros((len(p), len(p)))
    for v in ("solvent", "antisolvent", "additive", "solvent_annealing"):
        D += p[v].to_numpy()[:, None] != p[v].to_numpy()[None, :]
    d = p.dmso.map(DMSO_ORDER).to_numpy(float)
    D += np.abs(d[:, None] - d[None, :]) / 3
    for v in ("temp", "time"):
        x = p[v].to_numpy(float)
        D += np.abs(x[:, None] - x[None, :]) / max(x.max() - x.min(), 1)
    return pd.DataFrame(D, index=p.index, columns=p.index)


def score_metrics(score, actual, tie, dist):
    """score/actual: per-candidate Series on the same index."""
    n = len(actual)
    k = max(1, int(round(n * SHARE)))
    order = pd.DataFrame({"s": score.values, "t": tie}, index=actual.index).sort_values(["s", "t"], ascending=[False, True]).index
    rec = order[:k]
    best = actual.nlargest(k)
    base = actual.mean()
    rel = actual - actual.min()
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = (np.sort(rel.values)[::-1][:k] * disc).sum()
    return {"spearman": pd.Series(score.values, index=actual.index).corr(actual, method="spearman"),
            "captured": (actual.loc[rec].mean() - base) / (best.mean() - base) if best.mean() > base else np.nan,
            "rec_pce": actual.loc[rec].mean(),
            "ndcg": (rel.loc[rec].values * disc).sum() / idcg if idcg > 0 else np.nan,
            "hit": rec.isin(best.index).sum() / k,
            "near_miss": dist.loc[rec, best.index].min(axis=1).mean()}


def run_fold(F, year, rng):
    train = F[F.year < year].reset_index(drop=True)
    ev = F[F.year == year].reset_index(drop=True)
    models = {"lgbm": lgbm().fit(train, train.pce), "lgbm_detrended": Detrended().fit(train),
              "lgbm_process": lgbm(PROC).fit(train, train.pce), "ridge": ridge(train)[0]}
    X = ev.assign(year=year - 1, scan_direction="Reversed")  # pinned to the last training year, as in exp1/exp2
    pred = {m: mod.predict(X) for m, mod in models.items()}
    keys = ev.key.dropna().unique()
    knn = knn_score(keys, train)
    dist = process_distance(keys)
    groups = ev.groupby("doi").indices
    papers = np.array(list(groups))
    hi = ev.pce.values >= HIGH
    auc = {m: roc_auc_score(hi, models[m].predict(ev)) for m in models}  # device level, actual inputs
    rows = []
    for rep in range(-1, N_REP):
        if rep < 0:
            idx = np.arange(len(ev))
            copy = ev.doi.values
        else:
            drawn = rng.choice(papers, len(papers), replace=True)
            idx = np.concatenate([groups[p] for p in drawn])
            seen = {}
            copy = []
            for p in drawn:
                seen[p] = seen.get(p, 0) + 1
                copy += [f"{p}#{seen[p]}"] * len(groups[p])
        R = pd.DataFrame({"key": ev.key.values[idx], "pce": ev.pce.values[idx], "orig": ev.doi.values[idx],
                          "ridx": idx}).dropna(subset=["key"])
        g = R.groupby("key")
        cand = pd.DataFrame({"n": g.size(), "papers": g.orig.nunique(), "pce": g.pce.mean()})
        cand = cand[(cand.n >= E.MIN_DEVICES) & (cand.papers >= E.MIN_PAPERS)]
        if len(cand) < 10:
            continue
        rows_c = R[R.key.isin(cand.index)]
        tie = tie_key(cand.index)
        scores = {m: pd.Series(p[rows_c.ridx.values], index=rows_c.index).groupby(rows_c.key.values).mean().loc[cand.index]
                  for m, p in pred.items()}
        scores["knn"] = knn.loc[cand.index]
        scores["random"] = pd.Series(rng.permutation(len(cand)).astype(float), index=cand.index)
        d = dist.loc[cand.index, cand.index]
        for m, s in scores.items():
            rows.append({"year": year, "rep": rep, "method": m, "n_candidates": len(cand),
                         "n_answers": max(1, int(round(len(cand) * SHARE))),
                         **score_metrics(s, cand.pce, tie, d)})
    return rows, {"train_rows": len(train), "eval_rows": len(ev), "eval_papers": ev.doi.nunique(), **{f"AUC {m}": a for m, a in auc.items()}}


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    F = make_features(clean()[0])
    F["key"] = combo_key(F)
    F = F[F.year <= 2017].reset_index(drop=True)
    assert F.year.max() <= 2017, "phase 1 must not touch 2018+"
    t0 = time.time()
    runs, info = [], {}
    for i, y in enumerate(EVAL_YEARS):
        r, inf = run_fold(F, y, np.random.default_rng(np.random.SeedSequence(E.SEED + i)))
        runs += r
        info[y] = inf
        print(f"  {y}: {inf['train_rows']} train rows -> {inf['eval_rows']} eval rows, {(time.time() - t0) / 60:.1f} min", flush=True)
    R = pd.DataFrame(runs)
    raw = OUT / f"bootstrap_backtest_{stamp}.csv"
    R.to_csv(raw, index=False)

    B = R[R.rep >= 0]
    pooled = B.groupby(["rep", "method"])[METRICS].mean().reset_index()  # average of the three years per resample
    rows = []
    for (label, frame) in [("pooled 2015-2017", pooled)] + [(str(y), B[B.year == y]) for y in EVAL_YEARS]:
        for m, g in frame.groupby("method"):
            row = {"years": label, "method": NAME[m]}
            for mt in METRICS:
                v = g[mt].dropna()
                row[mt] = f"{v.mean():.3f} ({v.quantile(0.025):.3f}–{v.quantile(0.975):.3f})"
            rows.append(row)
    S = pd.DataFrame(rows)

    # paired vs random and vs LightGBM on the pooled resamples
    W = pooled.pivot(index="rep", columns="method")
    P = []
    for mt in METRICS:
        for a, b in [(m, "random") for m in NAME if m != "random"] + [(m, "lgbm") for m in NAME if m not in ("lgbm", "random")]:
            d = (W[mt][a] - W[mt][b]).dropna()
            lo, hi = d.quantile(0.025), d.quantile(0.975)
            P.append({"metric": mt, "comparison": f"{NAME[a]} − {NAME[b]}", "mean": d.mean(), "2.5%": lo, "97.5%": hi,
                      "judgement": "evidence insufficient" if lo <= 0 <= hi else ("higher" if lo > 0 else "lower")})
    P = pd.DataFrame(P)
    # how clearly does each metric separate LightGBM from random? (mean gap / bootstrap sd of the gap)
    sep = P[P.comparison == f"{NAME['lgbm']} − {NAME['random']}"].assign(
        separation=lambda t: [((W[mt]["lgbm"] - W[mt]["random"]).mean() / (W[mt]["lgbm"] - W[mt]["random"]).std())
                              for mt in t.metric])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, mt, title, ref in [(ax[0], "spearman", "순위 일치도 (Spearman, 무작위 = 0)", 0),
                              (ax[1], "captured", "추천 상위 10%가 얻은 효율 이득 비율 (무작위 = 0, 최선 = 1)", 0),
                              (ax[2], "hit", "기존 지표: 정답 적중 비율 (무작위 = 0.10)", 0.10)]:
        g = pooled.groupby("method")[mt]
        t = pd.DataFrame({"m": g.mean(), "lo": g.quantile(0.025), "hi": g.quantile(0.975)}).reindex(list(NAME)[::-1])
        a.barh([NAME[k] for k in t.index], t.m, xerr=[t.m - t.lo, t.hi - t.m], capsize=3,
               color=["#C44E52" if k == "random" else "#4C72B0" if k == "lgbm" else "#A0B4D0" for k in t.index])
        a.axvline(ref, color="#C44E52", ls="--", lw=1)
        a.set_title(title, fontsize=10)
    fig.suptitle("실험 3 연습: 2015·2016·2017년을 각각 그 전 해까지로 학습해 평가 (세 해 평균, 논문 재표집 500회)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "backtest_summary.png", dpi=120)
    plt.close(fig)

    cand = R[R.rep == -1].drop_duplicates("year")[["year", "n_candidates", "n_answers"]]
    (OUT / "results_backtest.md").write_text(
        "# experiment 3 phase 1: rolling backtests inside <= 2017 (generated by experiments/exp3/exp3_backtest.py)\n\n"
        f"raw: {raw.name}; {N_REP} paper resamples per year; candidate inputs pinned to the last training year and "
        f"scan Reverse; hyperparameters models/v2/best_params_past.json (chosen on <= 2017 train papers, so they have "
        f"seen part of these evaluation years)\n\n## folds\n{md(pd.DataFrame(info).T.round(3))}\n\n"
        f"## candidates / answers (original data)\n{md(cand, index=False)}\n\n"
        f"## metrics (mean, 95% interval)\n{md(S, index=False)}\n\n"
        f"## how clearly each metric separates LightGBM from random (pooled; separation = mean gap / sd of gap)\n"
        f"{md(sep[['metric', 'mean', '2.5%', '97.5%', 'judgement', 'separation']].round(3), index=False)}\n\n"
        f"## paired differences (pooled over the three years)\n{md(P.round(3), index=False)}\n", encoding="utf-8")
    print(md(sep[["metric", "mean", "2.5%", "97.5%", "judgement", "separation"]].round(3), index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> experiments/exp3/results_backtest.md")


if __name__ == "__main__":
    main()
