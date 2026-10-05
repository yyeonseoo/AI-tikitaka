"""Experiment 4: matched-input re-comparison on temporal backtests inside <= 2017 (plan: experiments/exp4/PLAN.md).
For each evaluation year Y in 2015-2017: train on papers before Y (hyperparameters chosen inside those years only),
score Y's process combinations, compare rank agreement with the actual combination efficiency.
Usage: python experiments/exp4/exp4_compare.py [R=1000]
Outputs: experiments/exp4/results_compare.md, bootstrap_compare_<timestamp>.csv, figures/compare_summary.png;
progress in experiments/exp4/progress.log (not committed)."""
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from experiments.exp1.exp1_bootstrap import tie_key
from experiments.exp1.exp1_bootstrap_baselines import parts, ridge
from experiments.exp1.exp1_hide_top import DMSO_ORDER, load_past
from models.v2.model_v2 import ALL, Model
from src.features import md

OUT = Path(__file__).resolve().parent
LOG = OUT / "progress.log"
YEARS = [2015, 2016, 2017]
PROC = C.PROCESS + C.CORRECTION
SHARE, LOW_PCE, N_ENS, K_NN, CFG_MIN = 0.10, 5.0, 10, 5, 8
CFG = ["arch", "etl", "htl", "backcontact"]
NAME = {"lgbm_full": "LightGBM (전체)", "lgbm_proc": "LightGBM (공정+보정)", "ridge_full": "선형 회귀 (전체)",
        "ridge_proc": "선형 회귀 (공정+보정)", "knn": "비슷한 조합 평균 (kNN)", "lgbm_lowcut": "LightGBM 저효율 제외 (전체)",
        "ens_mean": "LightGBM 10개 평균 (전체)", "ens_bonus": "LightGBM 10개 + 보너스 (전체)"}
REGS = ["lgbm_full", "lgbm_proc", "ridge_full", "ridge_proc", "lgbm_lowcut"]
PAIRS = [("lgbm_proc", "ridge_proc", "같은 공정 정보: 비선형 모델이 필요한가"),
         ("lgbm_proc", "knn", "같은 공정 정보: 모델이 비슷한 조합 평균보다 나은가"),
         ("lgbm_full", "lgbm_proc", "소자 조건 정보가 도움이 되는가 (LightGBM)"),
         ("ridge_full", "ridge_proc", "소자 조건 정보가 도움이 되는가 (선형 회귀)"),
         ("lgbm_full", "ridge_full", "같은 전체 정보: 비선형 모델이 필요한가"),
         ("lgbm_lowcut", "lgbm_full", "저효율 사례 제외가 도움이 되는가"),
         ("ens_mean", "lgbm_full", "모델 10개 평균이 도움이 되는가"),
         ("ens_bonus", "ens_mean", "불확실성 보너스가 도움이 되는가"),
         ("lgbm_full", "knn", "실용 기준선 (정보량 다름)")]
METRICS = ["spearman_dev", "spearman_paper", "hit_dev", "gain_dev"]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def log(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S} {text}\n")
    print(text, flush=True)


def tune_lgbm(train, feats):
    folds = list(GroupKFold(C.CV_FOLDS).split(train, groups=train["doi"]))
    best, best_mae = None, np.inf
    for impute in (True, False):
        for params in C.GRID["lgbm"]:
            err = [np.mean(np.abs(Model("lgbm", params, impute, feats).fit(train.iloc[a], train.pce.iloc[a])
                                  .predict(train.iloc[b]) - train.pce.iloc[b].values)) for a, b in folds]
            if np.mean(err) < best_mae:
                best, best_mae = (params, impute), np.mean(err)
    return best


def knn_score(keys, train):
    """Mean PCE of the K nearest training-year combinations (process distance, training-range scaling).
    A combination already reported in the training years counts as its own neighbour (real past record)."""
    known = train.dropna(subset=["key"]).groupby("key").pce.mean()
    pc, pk = parts(pd.Index(keys)), parts(known.index)
    D = np.zeros((len(pc), len(pk)))
    for v in ("solvent", "antisolvent", "additive", "solvent_annealing"):
        D += pc[v].to_numpy()[:, None] != pk[v].to_numpy()[None, :]
    D += np.abs(pc.dmso.map(DMSO_ORDER).to_numpy(float)[:, None] - pk.dmso.map(DMSO_ORDER).to_numpy(float)[None, :]) / 3
    for v in ("temp", "time"):
        x = pk[v].astype(float)
        D += np.abs(pc[v].to_numpy(float)[:, None] - x.to_numpy()[None, :]) / max(x.max() - x.min(), 1)
    ktie = tie_key(pk.index)
    return pd.Series([known.values[np.lexsort((ktie, D[i]))[:K_NN]].mean() for i in range(len(pc))], index=pc.index)


def fit_year(train, ev, rng):
    """Train every method on the training years (tuned inside them) and predict every evaluation row once."""
    X = ev.assign(year=int(train.year.max()), scan_direction="Reversed")  # correction variables pinned
    (pf, imf), (pp, imp) = tune_lgbm(train, ALL), tune_lgbm(train, PROC)
    rf, af = ridge(train, ALL)
    rp, ap = ridge(train, PROC)
    t = train[train.pce >= LOW_PCE]
    pred = {"lgbm_full": Model("lgbm", pf, imf, ALL).fit(train, train.pce).predict(X),
            "lgbm_proc": Model("lgbm", pp, imp, PROC).fit(train, train.pce).predict(X),
            "ridge_full": rf.predict(X), "ridge_proc": rp.predict(X),
            "lgbm_lowcut": Model("lgbm", pf, imf, ALL).fit(t, t.pce).predict(X)}
    groups = train.groupby("doi").indices
    papers = np.array(list(groups))
    for i in range(N_ENS):
        b = train.iloc[np.concatenate([groups[p] for p in rng.choice(papers, len(papers), replace=True)])]
        pred[f"ens{i}"] = Model("lgbm", pf, imf, ALL).fit(b, b.pce).predict(X)
    chosen = {"lgbm_full": {**pf, "impute": imf}, "lgbm_proc": {**pp, "impute": imp}, "ridge_full": af, "ridge_proc": ap}
    return pd.DataFrame(pred, index=ev.index), knn_score(ev.key.dropna().unique(), train), chosen


def eligible(R):
    """Candidate keys: >= 3 distinct original devices and >= 2 distinct original papers (copies counted once)."""
    u = R.drop_duplicates("rid")
    g = u.groupby("key").agg(n=("rid", "size"), p=("orig", "nunique"))
    return g[(g.n >= E.MIN_DEVICES) & (g.p >= E.MIN_PAPERS)].index


def combo_table(R, keys):
    """Per candidate: actual device mean, paper mean, and every method's score (mean of row predictions)."""
    S = R[R.key.isin(keys)]
    cols = REGS + [f"ens{i}" for i in range(N_ENS)]
    T = S.groupby("key")[cols + ["pce"]].mean()
    T["pce_paper"] = S.groupby(["key", "doi"]).pce.mean().groupby(level="key").mean()
    ens = T[[f"ens{i}" for i in range(N_ENS)]]
    T["ens_mean"], T["ens_bonus"] = ens.mean(axis=1), ens.mean(axis=1) + ens.std(axis=1)
    return T


def metrics(T, knn):
    T = T.assign(knn=knn.reindex(T.index).values)
    k = max(1, int(round(len(T) * SHARE)))
    tie = pd.Series(tie_key(T.index), index=T.index)
    ans = set(T.pce.nlargest(k).index)
    out = {}
    for m in NAME:
        order = pd.DataFrame({"s": T[m], "t": tie}).sort_values(["s", "t"], ascending=[False, True]).index
        top = order[:k]
        out[m] = {"spearman_dev": spearmanr(T[m], T.pce).statistic, "spearman_paper": spearmanr(T[m], T.pce_paper).statistic,
                  "hit_dev": top.isin(ans).sum() / k, "gain_dev": T.pce.loc[top].mean() - T.pce.mean()}
    return out, len(T)


def config_spearman(R, cfg_keys):
    """Fixed device configuration: rank combos inside each configuration with the configuration's own devices."""
    res = {m: [] for m in NAME if m != "knn"}
    weights = []
    for cfg, keys in cfg_keys.items():
        S = R[(R.cfg == cfg) & R.key.isin(keys)]
        T = combo_table(S, keys)
        if len(T) < CFG_MIN:
            continue
        weights.append(len(T))
        for m in res:
            res[m].append(spearmanr(T[m], T.pce).statistic)
    if not weights:
        return {}
    w = np.array(weights, float)
    return {m: float(np.nansum(np.array(v) * w) / w[~np.isnan(v)].sum()) for m, v in res.items()} | {"n_groups": len(w)}


def run_year(F, year, n_rep):
    ss_fit, ss_boot = np.random.SeedSequence([E.SEED, year]).spawn(2)
    train = F[F.year < year].reset_index(drop=True)
    ev = F[F.year == year].dropna(subset=["key"]).reset_index(drop=True)
    t0 = time.time()
    P, knn, chosen = fit_year(train, ev, np.random.default_rng(ss_fit))
    log(f"{year}: trained on {len(train)} rows ({train.doi.nunique()} papers) in {time.time() - t0:.0f}s; chosen {chosen}")
    base = pd.concat([ev[["key", "pce", "doi", "rid"]].assign(orig=ev.doi, cfg=ev[CFG].astype(object).fillna("missing").astype(str).agg(" | ".join, axis=1)), P], axis=1)
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
    groups = base.groupby("orig").indices
    papers = np.array(list(groups))
    rng = np.random.default_rng(ss_boot)
    for rep in range(n_rep):
        drawn = rng.choice(papers, len(papers), replace=True)
        seen = {}
        parts_ = []
        for p in drawn:
            seen[p] = seen.get(p, 0) + 1
            parts_.append(base.iloc[groups[p]].assign(doi=f"{p}#{seen[p]}"))
        record(rep, pd.concat(parts_, ignore_index=True))
        if (rep + 1) % 100 == 0:
            log(f"  {year}: resample {rep + 1}/{n_rep}, {(time.time() - t0) / 60:.1f} min")
    info = {"train_rows": len(train), "train_papers": train.doi.nunique(), "eval_rows": len(ev), "eval_papers": ev.doi.nunique(),
            "fixed_candidates": len(fixed), "config_groups": len(cfg_keys), "chosen": str(chosen)}
    return rows, info


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    F = load_past().reset_index(drop=True)
    assert F.year.max() <= 2017, "experiment 4 must not touch 2018+"
    F["rid"] = np.arange(len(F))
    t0 = time.time()
    log(f"start {stamp}: R={n_rep}, {len(F)} rows <= 2017")
    runs, info = [], {}
    for y in YEARS:
        r, inf = run_year(F, y, n_rep)
        runs += r
        info[y] = inf
    R = pd.DataFrame(runs)
    raw = OUT / f"bootstrap_compare_{stamp}.csv"
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

    W = pooled.pivot(index="rep", columns="method", values=allm)
    P = []
    for a, b, q in PAIRS:
        for mt in allm:
            d = (W[mt][a] - W[mt][b]).dropna()
            if d.empty:
                continue
            lo, hi = d.quantile(0.025), d.quantile(0.975)
            P.append({"question": q, "comparison": f"{NAME[a]} − {NAME[b]}", "metric": mt,
                      "point": point.loc[a, mt] - point.loc[b, mt], "2.5%": lo, "97.5%": hi,
                      "higher": (d > 0).mean(), "lower": (d < 0).mean(),
                      "judgement": "우열 판단 증거 부족" if lo <= 0 <= hi else ("앞이 높음" if lo > 0 else "뒤가 높음")})
    P = pd.DataFrame(P)
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
    fig.suptitle(f"실험 4: 전 해까지 학습 → 다음 해 평가 (2015·2016·2017, 연도별 논문 재표집 {n_rep}회)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "compare_summary.png", dpi=120)
    plt.close(fig)

    (OUT / "results_compare.md").write_text(
        "# experiment 4 results (generated by experiments/exp4/exp4_compare.py)\n\n"
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
