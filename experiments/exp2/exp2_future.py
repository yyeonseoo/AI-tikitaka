"""Experiment 2: train on papers up to a cutoff year, then rank the process combinations of later papers.
Plan (committed before running): experiments/exp2/PLAN.md.
Usage:
  python experiments/exp2/exp2_future.py --smoke   dry run inside <= 2017 (train <= 2015, evaluate 2016-2017)
  python experiments/exp2/exp2_future.py           real run (train <= 2017, evaluate 2018-2019), 1000 resamples
Outputs (real run): experiments/exp2/results_exp2.md, bootstrap_exp2_<timestamp>.csv, figures/;
smoke run writes to experiments/exp2/smoke/."""
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from experiments.exp1.exp1_bootstrap import tie_key
from experiments.exp1.exp1_bootstrap_baselines import parts, ridge
from experiments.exp1.exp1_hide_top import BEST, DMSO_ORDER, PAST, candidate_inputs, fed_values
from models.v2.model_v2 import ALL, Model, clean, combo_key, make_features
from src.features import md

OUT = Path(__file__).resolve().parent
SHARE, AUX_SHARE, HIGH, LOW_PCE, K_NN, N_ENS = 0.10, 0.20, 15.0, 5.0, 5, 10
PROC = C.PROCESS + C.CORRECTION  # "공정+보정": 7 process + 2 correction variables (correction pinned at prediction)
IMPUTE = BEST["missing"] == "median+flag"
NAME = {"lgbm": "LightGBM (주 방법)", "lgbm_no_fail": "LightGBM, 실패 소자 제외", "ens_mean": "LightGBM 10개 평균",
        "ens_bonus": "LightGBM 10개 + 불확실성", "ridge": "선형 회귀", "lgbm_process": "LightGBM, 공정+보정",
        "ridge_process": "선형 회귀, 공정+보정", "knn": "비슷한 조합 평균 (kNN)", "random": "무작위"}
REGRESSORS = ["lgbm", "lgbm_no_fail", "ens_mean", "ridge", "lgbm_process", "ridge_process"]
PAIRS = [(m, "lgbm") for m in NAME if m not in ("lgbm", "random")] + [("knn", "lgbm_process"), ("ens_bonus", "ens_mean")]
if "Malgun Gothic" in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def lgbm(features=ALL):
    return Model(BEST["model"], BEST["params"], IMPUTE, features)


def fit_all(train, rng):
    """Every regressor is trained once on the training years and then frozen."""
    t_ok = train[train.pce >= LOW_PCE]
    groups = train.groupby("doi").indices
    papers = np.array(list(groups))
    ens = [lgbm().fit(b, b.pce) for b in
           (train.iloc[np.concatenate([groups[p] for p in rng.choice(papers, len(papers), replace=True)])]
            for _ in range(N_ENS))]
    r_all, a1 = ridge(train)
    r_proc, a2 = ridge(train, PROC)
    return {"lgbm": lgbm().fit(train, train.pce), "lgbm_no_fail": lgbm().fit(t_ok, t_ok.pce), "ens": ens,
            "ridge": r_all, "lgbm_process": lgbm(PROC).fit(train, train.pce), "ridge_process": r_proc}, (a1, a2)


def row_preds(models, X):
    p = {k: m.predict(X) for k, m in models.items() if k != "ens"}
    p["ens_members"] = np.column_stack([m.predict(X) for m in models["ens"]])
    p["ens_mean"] = p["ens_members"].mean(axis=1)
    return p


def knn_score(keys, train):
    """Mean PCE of the K nearest training-period combinations by process distance. A candidate that already
    existed in the training years counts as its own neighbour (real past knowledge, see PLAN.md)."""
    known = train.dropna(subset=["key"]).groupby("key").pce.mean()
    pc, pk = parts(pd.Index(keys)), parts(known.index)
    span = {v: max(pk[v].astype(float).pipe(lambda s: s.max() - s.min()), 1) for v in ("temp", "time")}  # training range only
    D = np.zeros((len(pc), len(pk)))
    for v in ("solvent", "antisolvent", "additive", "solvent_annealing"):
        D += pc[v].to_numpy()[:, None] != pk[v].to_numpy()[None, :]
    D += np.abs(pc.dmso.map(DMSO_ORDER).to_numpy(float)[:, None] - pk.dmso.map(DMSO_ORDER).to_numpy(float)[None, :]) / 3
    for v in ("temp", "time"):
        D += np.abs(pc[v].to_numpy(float)[:, None] - pk[v].to_numpy(float)[None, :]) / span[v]
    ktie = tie_key(pk.index)
    return pd.Series([known.values[np.lexsort((ktie, D[i]))[:K_NN]].mean() for i in range(len(pc))], index=pc.index)


def combo_metrics(cand, scores, answers, tie):
    out = {}
    n, k = len(cand), len(answers)
    for m, s in scores.items():
        order = pd.DataFrame({"s": s.loc[cand.index].values, "t": tie}, index=cand.index) \
            .sort_values(["s", "t"], ascending=[False, True]).index
        hit = order.isin(answers)
        best = cand.loc[list(answers), "pce"].idxmax()
        out[m] = {"top10_share": hit[:k].sum() / k, "top20_recall": hit[:int(round(n * AUX_SHARE))].sum() / k,
                  "best_rank_pct": (order.get_loc(best) + 1) / n}
    return out


def evaluate_once(E_rows, pred_c, knn, seen, rep, rng_rand):
    """E_rows: evaluation rows of this (re)sample with columns key, pce, doi (copy id), orig_doi, ridx (row index into
    the prediction arrays). Re-derives candidates and answers, then scores every method."""
    g = E_rows.groupby("key")
    cand = pd.DataFrame({"n": g.size(), "papers": g.orig_doi.nunique(), "pce": g.pce.mean(),
                         "pce_paper": E_rows.groupby(["key", "doi"]).pce.mean().groupby(level="key").mean()})
    cand = cand[(cand.n >= E.MIN_DEVICES) & (cand.papers >= E.MIN_PAPERS)]
    k = int(round(len(cand) * SHARE))
    tie = tie_key(cand.index)
    rows = E_rows[E_rows.key.isin(cand.index)]
    scores = {m: pd.Series(pred_c[m][rows.ridx.values], index=rows.index).groupby(rows.key.values).mean()
              for m in REGRESSORS}
    members = pd.DataFrame(pred_c["ens_members"][rows.ridx.values], index=rows.index).groupby(rows.key.values).mean()
    scores["ens_bonus"] = members.mean(axis=1) + members.std(axis=1)
    scores["knn"] = knn
    scores["random"] = pd.Series(rng_rand.permutation(len(cand)).astype(float), index=cand.index)
    res = []
    for label, col, cset in [("main", "pce", cand), ("paper_mean", "pce_paper", cand),
                             ("unseen_only", "pce", cand[~cand.index.isin(seen)])]:
        kk = int(round(len(cset) * SHARE))
        if kk == 0:
            continue
        answers = set(cset[col].nlargest(kk).index)
        for m, v in combo_metrics(cset, scores, answers, tie_key(cset.index)).items():
            res.append({"rep": rep, "answers": label, "method": m, "n_candidates": len(cset), "n_answers": kk, **v})
    return res


def device_metrics(y, p, thr):
    hi = y >= HIGH
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p), "Spearman": spearmanr(y, p).statistic,
            "ROC-AUC": roc_auc_score(hi, p), "PR-AUC": average_precision_score(hi, p), "F1": f1_score(hi, p >= thr)}


def oof_threshold(train, make):
    p = np.empty(len(train))
    for a, b in GroupKFold(C.CV_FOLDS).split(train, groups=train["doi"]):
        p[b] = make(train.iloc[a]).predict(train.iloc[b])
    y = train.pce.values >= HIGH
    cands = np.unique(np.quantile(p, np.linspace(0.3, 0.98, 200)))
    return max(cands, key=lambda t: f1_score(y, p >= t))


def ci(v):
    v = np.asarray(v, float)
    return np.nanmean(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def main():
    smoke = "--smoke" in sys.argv
    train_max, eval_years, n_rep = (2015, (2016, 2017), 100) if smoke else (E.CUTOFF_YEAR, (2018, 2019), 1000)
    out = OUT / "smoke" if smoke else OUT
    (out / "figures").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ss_boot, ss_ens, ss_rand = np.random.SeedSequence(E.SEED).spawn(3)

    F = make_features(clean()[0])
    F["key"] = combo_key(F)
    F = F[F.year <= eval_years[1]].reset_index(drop=True)
    if smoke:
        assert F.year.max() <= 2017, "smoke run must not touch 2018+"
    train = F[F.year <= train_max].reset_index(drop=True)
    ev = F[F.year.between(*eval_years)].reset_index(drop=True)
    t0 = time.time()
    print(f"train <= {train_max}: {len(train)} rows / {train.doi.nunique()} papers; "
          f"evaluate {eval_years[0]}-{eval_years[1]}: {len(ev)} rows / {ev.doi.nunique()} papers")

    models, alphas = fit_all(train, np.random.default_rng(ss_ens))
    Xc = candidate_inputs(ev)  # pinned year / scan for ranking combinations
    fed = fed_values(Xc)
    assert fed == {v: [str(x)] for v, x in E.PREDICT_AT.items()}, fed
    pred_c = row_preds(models, Xc)
    keys = ev.key.dropna().unique()
    knn = knn_score(keys, train)
    seen = set(train.key.dropna())
    print(f"models fitted in {time.time() - t0:.0f}s; ridge alpha all/process = {alphas}; candidate inputs {fed}")

    base = ev.dropna(subset=["key"]).assign(orig_doi=lambda d: d.doi, ridx=lambda d: d.index)
    runs = evaluate_once(base, pred_c, knn, seen, -1, np.random.default_rng(ss_rand))
    groups = ev.groupby("doi").indices
    papers = np.array(list(groups))
    rng_b, rng_r = np.random.default_rng(ss_boot), np.random.default_rng(ss_rand)
    dev_boot = []
    pred_d = row_preds(models, ev)  # device-level: actual inputs
    thr = {m: oof_threshold(train, mk) for m, mk in
           [("lgbm", lambda d: lgbm().fit(d, d.pce)), ("lgbm_no_fail", lambda d: lgbm().fit(d[d.pce >= LOW_PCE], d[d.pce >= LOW_PCE].pce)),
            ("ridge", lambda d: ridge(d)[0]), ("lgbm_process", lambda d: lgbm(PROC).fit(d, d.pce)),
            ("ridge_process", lambda d: ridge(d, PROC)[0])]}
    thr["ens_mean"] = thr["lgbm"]  # ponytail: ensemble OOF would need 50 extra fits; reuse the single-model cut-off
    for rep in range(n_rep):
        drawn = rng_b.choice(papers, len(papers), replace=True)
        idx, copy = [], []
        seen_n = {}
        for p in drawn:
            seen_n[p] = seen_n.get(p, 0) + 1
            idx.append(groups[p])
            copy += [f"{p}#{seen_n[p]}"] * len(groups[p])
        idx = np.concatenate(idx)
        R = pd.DataFrame({"key": ev.key.values[idx], "pce": ev.pce.values[idx], "doi": copy,
                          "orig_doi": ev.doi.values[idx], "ridx": idx}).dropna(subset=["key"]).reset_index(drop=True)
        runs += evaluate_once(R, pred_c, knn, seen, rep, rng_r)
        y = ev.pce.values[idx]
        dev_boot.append({m: device_metrics(y, pred_d[m][idx], thr[m]) for m in REGRESSORS})
        if (rep + 1) % 100 == 0:
            print(f"  resample {rep + 1}/{n_rep}, {(time.time() - t0) / 60:.1f} min", flush=True)

    Rn = pd.DataFrame(runs)
    raw = out / f"bootstrap_exp2_{stamp}.csv"
    Rn.to_csv(raw, index=False)
    point = Rn[Rn.rep == -1].set_index(["answers", "method"])
    B = Rn[Rn.rep >= 0]
    rows = []
    for (ans, m), g in B.groupby(["answers", "method"]):
        mu, lo, hi = ci(g.top10_share)
        rows.append({"answers": ans, "method": NAME[m], "key": m, "point": point.top10_share.get((ans, m), np.nan),
                     "boot mean": mu, "2.5%": lo, "97.5%": hi, "top20 recall": g.top20_recall.mean(),
                     "best rank pct": g.best_rank_pct.mean()})
    S = pd.DataFrame(rows)
    W = B.pivot_table(index=["rep", "answers"], columns="method", values="top10_share")
    P = []
    for a, b in PAIRS:
        for ans, d in (W[a] - W[b]).groupby(level="answers"):
            mu, lo, hi = ci(d)
            P.append({"answers": ans, "comparison": f"{NAME[a]} − {NAME[b]}", "mean diff": mu, "2.5%": lo, "97.5%": hi,
                      "higher": (d > 0).mean(), "equal": (d == 0).mean(), "lower": (d < 0).mean(),
                      "judgement": "evidence insufficient" if lo <= 0 <= hi else ("higher" if lo > 0 else "lower")})
    P = pd.DataFrame(P)
    point_dev = {m: device_metrics(ev.pce.values, pred_d[m], thr[m]) for m in REGRESSORS}
    D = []
    for m in REGRESSORS:
        for k in point_dev[m]:
            mu, lo, hi = ci([b[m][k] for b in dev_boot])
            D.append({"method": NAME[m], "metric": k, "point": point_dev[m][k], "2.5%": lo, "97.5%": hi})
    D = pd.DataFrame(D)

    main_l = S[(S.answers == "main") & (S.key == "lgbm")].iloc[0]
    auc_l = D[(D.method == NAME["lgbm"]) & (D.metric == "ROC-AUC")].iloc[0]
    verdict = "SUCCESS" if main_l["2.5%"] > SHARE else "NOT MET"
    support = "met" if auc_l["2.5%"] > 0.5 else "not met"

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    t = S[S.answers == "main"].set_index("key").loc[[m for m in NAME if m != "random"] + ["random"]].iloc[::-1]
    ax[0].barh(t.method, t["boot mean"], xerr=[t["boot mean"] - t["2.5%"], t["97.5%"] - t["boot mean"]], capsize=3,
               color=["#4C72B0" if k == "lgbm" else "#C44E52" if k == "random" else "#A0B4D0" for k in t.index])
    ax[0].axvline(SHARE, color="#C44E52", ls="--")
    ax[0].set(xlabel="고효율 조합(상위 10%) 중 추천 상위 10% 안에 든 비율", xlim=(0, 1),
              title=f"(a) {eval_years[0]}~{eval_years[1]}년 조합 추천 (학습 ≤{train_max}, 논문 재표집 {n_rep}회)")
    d = D[D.metric.isin(["ROC-AUC", "Spearman", "R2"])].pivot(index="method", columns="metric", values="point")
    lo_ = D[D.metric.isin(["ROC-AUC", "Spearman", "R2"])].pivot(index="method", columns="metric", values="2.5%")
    hi_ = D[D.metric.isin(["ROC-AUC", "Spearman", "R2"])].pivot(index="method", columns="metric", values="97.5%")
    order = [NAME[m] for m in REGRESSORS]
    x = np.arange(len(order))
    for j, (mt, col) in enumerate([("R2", "#4C72B0"), ("Spearman", "#55A868"), ("ROC-AUC", "#DD8452")]):
        v = d.loc[order, mt].values
        ax[1].bar(x + (j - 1) * 0.27, v, 0.27, yerr=[v - lo_.loc[order, mt].values, hi_.loc[order, mt].values - v],
                  capsize=2, color=col, label={"R2": "R²", "Spearman": "순위 상관", "ROC-AUC": "고효율 판별 AUC"}[mt])
    ax[1].axhline(0.5, color="#DD8452", ls=":", lw=1)
    ax[1].set_xticks(x, [o.replace(", ", "\n") for o in order], fontsize=8)
    ax[1].set(ylim=(min(0, np.nanmin(lo_.values) - 0.05), 1), title="(b) 소자 단위 성능 (실제 입력값, 95% 구간)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "exp2_summary.png", dpi=120)
    plt.close(fig)

    (out / ("results_smoke.md" if smoke else "results_exp2.md")).write_text(
        f"# experiment 2 results (generated by experiments/exp2/exp2_future.py{' --smoke' if smoke else ''})\n\n"
        f"train <= {train_max}: {len(train)} rows / {train.doi.nunique()} papers; evaluate {eval_years[0]}-{eval_years[1]}: "
        f"{len(ev)} rows / {ev.doi.nunique()} papers; resamples {n_rep}; raw {raw.name}\n\n"
        f"hyperparameters {PAST.name}; ridge alpha (all / process-only) {alphas}; F1 thresholds { {k: round(float(v), 2) for k, v in thr.items()} }; "
        f"candidate inputs {fed}\n\n"
        f"**verdict (pre-registered): {verdict}** — main method top-10% share {main_l['boot mean']:.3f} "
        f"(95% {main_l['2.5%']:.3f}–{main_l['97.5%']:.3f}) vs random {SHARE}; supporting device ROC-AUC "
        f"{auc_l['point']:.3f} (95% {auc_l['2.5%']:.3f}–{auc_l['97.5%']:.3f}) -> {support}\n\n"
        f"## candidates / answers (original data)\n{md(Rn[(Rn.rep == -1) & (Rn.method == 'lgbm')][['answers', 'n_candidates', 'n_answers']], index=False)}\n\n"
        f"## top-10% share by method\n{md(S.drop(columns='key').round(3), index=False)}\n\n"
        f"## paired differences (same resamples)\n{md(P.round(3), index=False)}\n\n"
        f"## device-level metrics on evaluation rows (actual inputs)\n{md(D.round(3), index=False)}\n", encoding="utf-8")
    print(f"\nverdict: {verdict}  (LightGBM top-10% share {main_l['boot mean']:.2f} [{main_l['2.5%']:.2f}, {main_l['97.5%']:.2f}],"
          f" random {SHARE}); device AUC {auc_l['point']:.2f} [{auc_l['2.5%']:.2f}, {auc_l['97.5%']:.2f}] -> {support}")
    for _, r in S[S.answers == "main"].sort_values("boot mean", ascending=False).iterrows():
        print(f"  {r.method:24s} {r['boot mean']:.2f} [{r['2.5%']:.2f}, {r['97.5%']:.2f}]")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
