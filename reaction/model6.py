"""Reaction yield, plan 6 (reaction/README.md, "6차 계획"): one-shot condition recommendation on Suzuki-Miyaura.
For each of the 15 (electrophile, nucleophile) pairs, train on the other 14 and rank its 384 conditions with no experiment
in the target pair. Primary: regret@8, LightGBM·F1 - ridge·F1 (same molecular features), bootstrap over the 15 pairs.
Usage: python reaction/model6.py   (-> reaction/results_model6.md)"""
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt, ridge_pairs  # noqa: E402
from model2 import desc_sm  # noqa: E402
from model5 import NEAR, PAIR, bayes_linear, history_order, judge, lgbm_ensemble  # noqa: E402
from prepare import SM_SLOTS, load_sm  # noqa: E402

OUT = ROOT / "results_model6.md"
ALPHAS = [0.1, 1, 10, 100, 1000]
KS = [1, 4, 8, 20]
HIT_KS = [4, 8, 20]


def ridge_f1(X, y, key, hist, target):
    """Standardised ridge on F1; alpha by leave-one-pair-out MAE inside the 14 training pairs."""
    make = lambda a: make_pipeline(StandardScaler(), Ridge(a))
    hp = key[hist]
    cv = {a: np.mean([mean_absolute_error(y[hist][hp == g], make(a).fit(X[hist][hp != g], y[hist][hp != g]).predict(X[hist][hp == g]))
                      for g in pd.unique(hp)]) for a in ALPHAS}
    a = min(cv, key=cv.get)
    return np.clip(make(a).fit(X[hist], y[hist]).predict(X[target]), 0, 100), a


def predict_pair(XF, XR, y, key, p):
    threadpool_limits(1)
    hist, target = np.flatnonzero(key != p), np.flatnonzero(key == p)
    pr, a = ridge_f1(XF, y, key, hist, target)
    return p, a, {"선형 회귀 + 분자 특성": pr,
                  "선형 + 짝 상호작용": np.clip(bayes_linear(XR[hist], y[hist], XR[target], "warm")[0], 0, 100),
                  "LightGBM + 분자 특성": np.clip(lgbm_ensemble(XF[hist], y[hist], XF[target], "warm")[0], 0, 100)}


def ranked_metrics(yp, ranked):
    best = yp.max()
    cum = np.maximum.accumulate(ranked)
    return {**{f"regret@{k}": best - cum[k - 1] for k in KS}, **{f"hit@{k}": float(cum[k - 1] >= best - NEAR) for k in HIT_KS}}


def random_metrics(yp):
    ys, N, best = np.sort(yp), len(yp), yp.max()
    m = int((yp >= best - NEAR).sum())
    r = {f"regret@{k}": best - sum(comb(i - 1, k - 1) / comb(N, k) * ys[i - 1] for i in range(k, N + 1)) for k in KS}
    return {**r, **{f"hit@{k}": 1 - comb(N - m, k) / comb(N, k) for k in HIT_KS}}


def main():
    t0 = time.time()
    d = load_sm()
    y = d.y.to_numpy(float)
    key = d[PAIR].agg(" + ".join, axis=1).to_numpy()
    XF, XR = desc_sm(d).to_numpy(float), ridge_pairs(d, SM_SLOTS).to_numpy(float)
    res = Parallel(n_jobs=-1)(delayed(predict_pair)(XF, XR, y, key, p) for p in pd.unique(key))
    hist_order = history_order(d)
    tie = np.random.default_rng(0)
    rows, alphas = [], {}
    for p, a, P in res:
        alphas[p] = a
        yp = y[key == p]
        rows.append({"method": "무작위", "pair": p, **random_metrics(yp)})
        h = hist_order[p][0]
        rows.append({"method": "과거 평균 순서", "pair": p, **ranked_metrics(yp, h)})
        for m, q in P.items():
            order = np.lexsort((tie.random(len(q)), -q))
            rows.append({"method": m, "pair": p, **ranked_metrics(yp, yp[order]),
                         "spearman": spearmanr(q, yp)[0], "MAE": np.abs(q - yp).mean()})
    T = pd.DataFrame(rows)

    def diff(a, b, col="regret@8"):
        A, B = (T[T.method == m].set_index("pair")[col] for m in (a, b))
        return boot_ci((A - B.loc[A.index]).to_numpy())

    lg, lf = "LightGBM + 분자 특성", "선형 회귀 + 분자 특성"
    V = [("**주**", lg, lf)] + [("2차", lg, b) for b in ("선형 + 짝 상호작용", "과거 평균 순서")] + \
        [("2차", m, "무작위") for m in ("과거 평균 순서", "선형 + 짝 상호작용", lf, lg)]
    vt = "| 구분 | 비교 (regret@8) | 차이 (%p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|---|\n"
    for lab, a, b in V:
        t = diff(a, b)
        vt += f"| {lab} | {a} − {b} | {fmt(t, True)} | {f'**{judge(t)}**' if lab == '**주**' else judge(t)} |\n"

    order = ["무작위", "과거 평균 순서", "선형 + 짝 상호작용", lf, lg]
    S = []
    for m in order:
        t = T[T.method == m]
        r = {"방법": m}
        for k in KS:
            r[f"regret@{k}"] = fmt(boot_ci(t[f"regret@{k}"])) if k == 8 else f"{t[f'regret@{k}'].mean():.2f}"
        for k in HIT_KS:
            r[f"상위 {k}개 안 발견"] = f"{t[f'hit@{k}'].mean():.0%}"
        r["순위 상관"] = f"{t.spearman.mean():.2f}" if "spearman" in t and t.spearman.notna().any() else "–"
        r["MAE"] = f"{t.MAE.mean():.2f}" if "MAE" in t and t.MAE.notna().any() else "–"
        S.append(r)
    per_pair = T[T.method.isin(order[1:])].pivot(index="pair", columns="method", values="regret@8")[order[1:]].round(2)

    OUT.write_text(
        "# 반응 수율 6차 결과: SM 조건 추천 (generated by reaction/model6.py)\n\n"
        f"목표 쌍 15개, 쌍마다 나머지 14쌍으로 학습해 조건 384개에 순위. 구간은 15쌍 단위 부트스트랩 1,000회. "
        f"'발견' = 최고와 {NEAR:g}%p 이내 조건. 선형 회귀 + 분자 특성의 정규화 강도: {sorted(set(alphas.values()))}. "
        f"실행 {(time.time() - t0) / 60:.0f}분.\n\n"
        f"## 판정\n{vt}\n## 방법별 (15쌍 평균)\n{md(pd.DataFrame(S), index=False)}\n\n"
        f"## 쌍별 regret@8\n{md(per_pair.reset_index(), index=False)}\n", encoding="utf-8")
    print(vt)


if __name__ == "__main__":
    main()
