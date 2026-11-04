"""Reaction yield, plan 11 (reaction/README.md, "11차 계획"): plan 5 sequential optimisation with a Gaussian-process
surrogate (standard BO), compared with the plan 5 LightGBM runs (read from cache, same starting 4 per seed).
GP cold: one-hot conditions, Matern 5/2 + noise, kernel refit each round. GP warm: standardised F1, kernel fitted once
per target pair on the 14 history pairs, then kept fixed. Primary: regret@20, LightGBM warm (greedy) - GP cold (UCB).
Usage: python reaction/model11.py   (-> reaction/results_model11.md)"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import fmt, onehot  # noqa: E402
from model2 import desc_sm  # noqa: E402
from model5 import (BATCH, CACHE5, COND, KAPPA, N0, PAIR, ROUNDS, SEEDS, diff_ci, judge, summary,  # noqa: E402
                    table)
from prepare import load_sm  # noqa: E402

OUT, CACHE = ROOT / "results_model11.md", ROOT / "cache" / "model11"
KFIT_N = 1500


def kernel():
    return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)


def gp(Xtr, ytr, k=None):
    """k=None: fit kernel hyperparameters; else keep k fixed."""
    m = GaussianProcessRegressor(k if k is not None else kernel(), normalize_y=True, optimizer=None if k is not None else "fmin_l_bfgs_b")
    return m.fit(Xtr, ytr)


def run(X, y, hist, target, kappa, seed, pair_i, k):
    """Same loop and random stream as model5.run, GP surrogate."""
    threadpool_limits(1)
    rng = np.random.default_rng([seed, pair_i])
    done = list(rng.choice(target, N0, replace=False))
    for _ in range(ROUNDS):
        rest = np.setdiff1d(target, done)
        trn = np.concatenate([hist, done]).astype(int)
        mu, sd = gp(X[trn], y[trn], k).predict(X[rest], return_std=True)
        done += list(rest[np.lexsort((rng.random(len(rest)), -(mu + kappa * sd)))[:BATCH]])
    return np.array(done)


def fit_kernel(X, y, hist, i):
    """Hyperparameters from a 1,500-row sample of the history (full 5,376 took > 30 min per pair)."""
    threadpool_limits(4)
    sub = np.random.default_rng(i).choice(hist, KFIT_N, replace=False)
    return gp(X[sub], y[sub]).kernel_


def traces(d, X, y, key, scenario, kappa, kernels):
    out = {}
    for i, p in enumerate(pd.unique(key)):
        ck = CACHE / f"{scenario}_GP_k{kappa:g}_{i}.pkl"
        if not ck.exists():
            target = np.flatnonzero(key == p)
            hist = np.flatnonzero(key != p) if scenario == "warm" else np.array([], int)
            runs = Parallel(n_jobs=len(SEEDS))(delayed(run)(X, y, hist, target, kappa, s, i, kernels.get(p)) for s in SEEDS)
            CACHE.mkdir(parents=True, exist_ok=True)
            ck.write_bytes(pickle.dumps(np.stack([y[r] for r in runs])))
        out[p] = pickle.loads(ck.read_bytes())
        print(f"  {scenario} GP κ={kappa:g} pair {i}: done", flush=True)
    return out


def lgbm_cached(key, scenario, kappa):
    return {p: pickle.loads((CACHE5 / f"{scenario}_LightGBM_k{kappa:g}_{i}.pkl").read_bytes()) for i, p in enumerate(pd.unique(key))}


def main():
    t0 = time.time()
    d = load_sm()
    y, key = d.y.to_numpy(float), d[PAIR].agg(" + ".join, axis=1).to_numpy()
    Xc = onehot(d, COND).to_numpy(float)
    Xw = StandardScaler().fit_transform(desc_sm(d).to_numpy(float))
    Xw = Xw[:, Xw.std(0) > 0]

    kpath = CACHE / "warm_kernels.pkl"
    if not kpath.exists():
        pairs = list(pd.unique(key))
        ks = Parallel(n_jobs=4)(delayed(fit_kernel)(Xw, y, np.flatnonzero(key != p), i) for i, p in enumerate(pairs))
        CACHE.mkdir(parents=True, exist_ok=True)
        kpath.write_bytes(pickle.dumps(dict(zip(pairs, ks))))
    kw = pickle.loads(kpath.read_bytes())
    print("  warm kernels:", {i: str(k) for i, k in enumerate(list(kw.values())[:2])}, flush=True)

    acq = {"탐욕": 0.0, "UCB": KAPPA}
    runs = {}
    for a, k in acq.items():
        runs[f"GP 과거 없음 ({a})"] = traces(d, Xc, y, key, "cold", k, {})
        runs[f"GP 과거 있음 ({a})"] = traces(d, Xw, y, key, "warm", k, kw)
        runs[f"LightGBM 과거 없음 ({a})"] = lgbm_cached(key, "cold", k)
        runs[f"LightGBM 과거 있음 ({a})"] = lgbm_cached(key, "warm", k)
    T = table(d, runs)

    lw = "LightGBM 과거 있음 (탐욕)"
    V = [("**주**", lw, "GP 과거 없음 (UCB)"),
         ("2차", lw, "GP 과거 있음 (탐욕)"), ("2차", "LightGBM 과거 있음 (UCB)", "GP 과거 있음 (UCB)"),
         ("참고", "LightGBM 과거 없음 (탐욕)", "GP 과거 없음 (탐욕)"), ("참고", "LightGBM 과거 없음 (UCB)", "GP 과거 없음 (UCB)")]
    V += [("2차", f"GP {s} ({a})", "무작위") for s in ("과거 없음", "과거 있음") for a in acq]
    vt = "| 구분 | 비교 | regret@20 차이 (%p, 음수 = 앞이 좋음) | 결과 | 발견까지 실험 수 차이 |\n|---|---|---|---|---|\n"
    for lab, a, b in V:
        t, e = diff_ci(T, a, b, "regret@20"), diff_ci(T, a, b, "발견까지 실험 수")
        vt += f"| {lab} | {a} − {b} | {fmt(t, True)} | {f'**{judge(t)}**' if lab == '**주**' else judge(t)} | {fmt(e, True)} |\n"

    methods = ["무작위"] + list(runs)
    OUT.write_text(
        "# 반응 수율 11차 결과: 가우시안 과정(표준 BO)과 비교 (generated by reaction/model11.py)\n\n"
        f"5차와 같은 설정(15쌍 × 시드 10개, 무작위 {N0}개 시작, {BATCH}개씩 {ROUNDS}번). LightGBM은 5차 저장 결과. "
        f"'발견' = 최고와 5%p 이내 (못 찾으면 41로 셈). 구간은 15쌍 부트스트랩. 실행 {(time.time() - t0) / 60:.0f}분 (이어서 실행하면 마지막 구간만).\n\n"
        f"## 판정\n{vt}\n## 방법별 (15쌍 평균)\n{md(summary(T, methods), index=False)}\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
