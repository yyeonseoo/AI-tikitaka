"""Reaction yield, plan 5 (reaction/README.md, "5차 계획"): sequential condition optimisation on Suzuki-Miyaura,
replayed on the dataset (yields are looked up, not measured).
Target = one (electrophile, nucleophile) pair with 384 conditions (ligand x base x solvent). Each run starts from 4 random
experiments, then 9 batches of 4 picked by a model (40 in total). Primary: warm start (the other 14 pairs as history),
greedy picks, regret@20, LightGBM - Bayesian linear, bootstrap over the 15 pairs.
Usage: python reaction/model5.py   (-> reaction/results_model5.md, reaction/figures/model5_curves.png)"""
import pickle
import sys
import time
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt, onehot, ridge_pairs  # noqa: E402
from model2 import desc_sm  # noqa: E402
from prepare import SM_SLOTS, load_sm  # noqa: E402

OUT, FIG = ROOT / "results_model5.md", ROOT / "figures"
CACHE5 = ROOT / "cache" / "model5"
PAIR, COND = ["electrophile", "nucleophile"], ["ligand", "base", "solvent"]
N0, BATCH, ROUNDS = 4, 4, 9
N_MAX = N0 + BATCH * ROUNDS
SEEDS, KAPPA, NEAR = range(10), 1.0, 5.0
REPORT_N = [8, 12, 20, 40]
LGBM = {"warm": dict(num_leaves=31, min_child_samples=10), "cold": dict(num_leaves=7, min_child_samples=3)}
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ---------------- surrogates: (X_train, y_train, X_apply) -> mean, std ----------------
def lgbm_ensemble(Xtr, ytr, Xap, scenario):
    P = np.stack([LGBMRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                random_state=s, verbose=-1, n_jobs=1, **LGBM[scenario]).fit(Xtr, ytr).predict(Xap)
                  for s in range(5)])
    return P.mean(0), P.std(0)


def bayes_linear(Xtr, ytr, Xap, scenario):
    return BayesianRidge().fit(Xtr, ytr).predict(Xap, return_std=True)


MODELS = {"LightGBM": lgbm_ensemble, "선형 모델": bayes_linear}


# ---------------- one run ----------------
def run(X, y, hist, target, model, scenario, kappa, seed, pair_i):
    """Indices of the 40 experiments in the order they were run."""
    threadpool_limits(1)  # one BLAS thread per worker: oversubscribed BayesianRidge was 20x slower
    rng = np.random.default_rng([seed, pair_i])
    done = list(rng.choice(target, N0, replace=False))
    for _ in range(ROUNDS):
        rest = np.setdiff1d(target, done)
        trn = np.concatenate([hist, done])
        mu, sd = MODELS[model](X[trn], y[trn], X[rest], scenario)
        score = mu + kappa * sd
        done += list(rest[np.lexsort((rng.random(len(rest)), -score))[:BATCH]])
    return np.array(done)


def traces(d, Xs, scenario, model, kappa):
    """{pair: array (seeds x 40) of yields in run order}, checkpointed per pair."""
    y = d.y.to_numpy(float)
    key = d[PAIR].agg(" + ".join, axis=1).to_numpy()
    out = {}
    for i, p in enumerate(pd.unique(key)):
        ck = CACHE5 / f"{scenario}_{model}_k{kappa:g}_{i}.pkl"
        if not ck.exists():
            target, hist = np.flatnonzero(key == p), (np.flatnonzero(key != p) if scenario == "warm" else np.array([], int))
            X = Xs[(scenario, model)]
            runs = Parallel(n_jobs=-1)(delayed(run)(X, y, hist, target, model, scenario, kappa, s, i) for s in SEEDS)
            CACHE5.mkdir(parents=True, exist_ok=True)
            ck.write_bytes(pickle.dumps(np.stack([y[r] for r in runs])))
        out[p] = pickle.loads(ck.read_bytes())
        print(f"  {scenario} {model} κ={kappa:g} pair {i}: done", flush=True)
    return out


def history_order(d):
    """Static order: mean yield of the same ligand/base/solvent over the other 14 pairs (fallback ligand/base)."""
    key = d[PAIR].agg(" + ".join, axis=1)
    out = {}
    for p in pd.unique(key):
        tr, ap = d[key != p], d[key == p]
        m3 = ap[COND].merge(tr.groupby(COND).y.mean().rename("m"), how="left", left_on=COND, right_index=True).m.to_numpy()
        m2 = ap[COND[:2]].merge(tr.groupby(COND[:2]).y.mean().rename("m"), how="left", left_on=COND[:2], right_index=True).m.to_numpy()
        s = np.where(np.isnan(m3), m2, m3)
        out[p] = ap.y.to_numpy()[np.lexsort((np.random.default_rng(0).random(len(s)), -s))][None, :N_MAX]
    return out


# ---------------- metrics ----------------
def trace_metrics(T, best):
    """Per pair (averaged over seeds): regret@n for REPORT_N, near-best found within 40, experiments to near-best (41 if not)."""
    cum = np.maximum.accumulate(T, axis=1)
    hit = cum >= best - NEAR
    first = np.where(hit.any(1), hit.argmax(1) + 1, N_MAX + 1)
    r = {f"regret@{n}": (best - cum[:, n - 1]).mean() for n in REPORT_N}
    return {**r, "40번 안 발견": hit[:, -1].mean(), "발견까지 실험 수": first.mean(), "curve": (best - cum).mean(0)}


def random_metrics(yp):
    """Exact expectations for experiments drawn at random without replacement."""
    ys, N, best = np.sort(yp), len(yp), yp.max()
    m = int((yp >= best - NEAR).sum())
    curve = [best - sum(comb(i - 1, n - 1) / comb(N, n) * ys[i - 1] for i in range(n, N + 1)) for n in range(1, N_MAX + 1)]
    surv = [comb(N - m, t) / comb(N, t) for t in range(N_MAX + 1)]  # P(not found after t)
    return {**{f"regret@{n}": curve[n - 1] for n in REPORT_N}, "40번 안 발견": 1 - surv[N_MAX],
            "발견까지 실험 수": sum(surv), "curve": np.array(curve)}


def table(d, runs):
    """Long table: one row per (method, pair)."""
    best = d.groupby(d[PAIR].agg(" + ".join, axis=1)).y.max()
    rows = []
    for p, b in best.items():
        yp = d.y[d[PAIR].agg(" + ".join, axis=1) == p].to_numpy(float)
        rows.append({"method": "무작위", "pair": p, **random_metrics(yp)})
        for name, tr in runs.items():
            rows.append({"method": name, "pair": p, **trace_metrics(tr[p], b)})
    return pd.DataFrame(rows)


def diff_ci(T, a, b, col):
    A, B = (T[T.method == m].set_index("pair")[col] for m in (a, b))
    return boot_ci((A - B).loc[A.index].to_numpy())


def judge(t):
    return "충족" if t[2] < 0 else "반대 방향" if t[1] > 0 else "우열 판단 증거 부족"


def summary(T, methods):
    cols = [f"regret@{n}" for n in REPORT_N] + ["40번 안 발견", "발견까지 실험 수"]
    rows = []
    for m in methods:
        t = T[T.method == m]
        r = {"방법": m}
        for c in cols:
            r[c] = f"{t[c].mean():.0%}" if c == "40번 안 발견" else fmt(boot_ci(t[c])) if c == "regret@20" else f"{t[c].mean():.2f}"
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    d = load_sm()
    F1 = desc_sm(d).to_numpy(float)
    Xs = {("warm", "LightGBM"): F1, ("cold", "LightGBM"): F1,
          ("warm", "선형 모델"): ridge_pairs(d, SM_SLOTS).to_numpy(float), ("cold", "선형 모델"): onehot(d, COND).to_numpy(float)}
    acq = {"탐욕": 0.0, "UCB": KAPPA}
    runs = {s: {f"{m} ({a})": traces(d, Xs, s, m, k) for m in MODELS for a, k in acq.items()} for s in ("warm", "cold")}
    runs["warm"]["과거 평균 순서"] = history_order(d)
    T = {s: table(d, r) for s, r in runs.items()}
    Tw, Tc = T["warm"], T["cold"]

    lg, li = "LightGBM (탐욕)", "선형 모델 (탐욕)"
    V = [("**주**", "과거 데이터 있음", Tw, lg, li), ("2차", "과거 데이터 있음", Tw, "LightGBM (UCB)", "선형 모델 (UCB)"),
         ("2차", "과거 데이터 없음", Tc, lg, li), ("2차", "과거 데이터 없음", Tc, "LightGBM (UCB)", "선형 모델 (UCB)")]
    for m in (lg, li, "LightGBM (UCB)", "선형 모델 (UCB)"):
        V += [("2차", "과거 데이터 있음", Tw, m, "무작위"), ("2차", "과거 데이터 있음", Tw, m, "과거 평균 순서")]
    V += [("2차", "과거 데이터 있음", Tw, "과거 평균 순서", "무작위")]
    for m in (lg, li, "LightGBM (UCB)", "선형 모델 (UCB)"):
        V.append(("2차", "과거 데이터 없음", Tc, m, "무작위"))
    vt = "| 구분 | 상황 | 비교 (regret@20) | 차이 (%p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|---|---|\n"
    for lab, name, Tt, a, b in V:
        t = diff_ci(Tt, a, b, "regret@20")
        res = f"**{judge(t)}**" if lab == "**주**" else judge(t)
        vt += f"| {lab} | {name} | {a} − {b} | {fmt(t, True)} | {res} |\n"

    mw = ["무작위", "과거 평균 순서", lg, li, "LightGBM (UCB)", "선형 모델 (UCB)"]
    mc = ["무작위", lg, li, "LightGBM (UCB)", "선형 모델 (UCB)"]
    per_pair = Tw[Tw.method.isin([lg, li, "과거 평균 순서"])].pivot(index="pair", columns="method", values="regret@20").round(2)
    per_pair.insert(0, "최고 수율", d.groupby(d[PAIR].agg(" + ".join, axis=1)).y.max().round(1))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = {"무작위": "#c3c2b7", "과거 평균 순서": "#9a9893", lg: "#2a78d6", li: "#eb6834",
              "LightGBM (UCB)": "#86b6ef", "선형 모델 (UCB)": "#f2a983"}
    n = np.arange(1, N_MAX + 1)
    for a, (title, Tt, ms) in zip(ax, (("(a) 과거 데이터 있음 (다른 14쌍)", Tw, mw), ("(b) 과거 데이터 없음", Tc, mc))):
        for m in ms:
            a.plot(n, np.mean(np.stack(Tt[Tt.method == m].curve), 0), lw=2, color=colors[m], label=m,
                   ls="--" if "UCB" in m else "-")
        a.axvline(20, color="#d6d4cc", lw=1)
        a.set(title=title, xlabel="실험 수", xlim=(1, N_MAX))
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e1e0d9")
    ax[0].set_ylabel("최고 수율과 지금까지 찾은 최고의 차이 (%p)")
    ax[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "model5_curves.png", dpi=110)
    plt.close(fig)

    OUT.write_text(
        "# 반응 수율 5차 결과: SM 순차 조건 최적화 (generated by reaction/model5.py)\n\n"
        f"목표 쌍 15개 × 시작 시드 10개. 쌍마다 조건 384개, 무작위 4개로 시작해 4개씩 9번 (총 {N_MAX}개). "
        f"구간은 15쌍 단위 부트스트랩 1,000회. '발견' = 최고와 {NEAR:g}%p 이내 조건 (못 찾으면 실험 수 {N_MAX + 1}로 셈). "
        f"실행 {(time.time() - t0) / 60:.0f}분 (이어서 실행하면 마지막 구간만 셈).\n\n"
        f"## 판정\n{vt}\n"
        f"## 과거 데이터 있음 (다른 14쌍 학습)\n{md(summary(Tw, mw), index=False)}\n\n"
        f"## 과거 데이터 없음\n{md(summary(Tc, mc), index=False)}\n\n"
        f"## 쌍별 regret@20 (과거 데이터 있음, 탐욕)\n{md(per_pair.reset_index(), index=False)}\n\n"
        "그림: [figures/model5_curves.png](figures/model5_curves.png)\n", encoding="utf-8")
    print(vt)


if __name__ == "__main__":
    main()
