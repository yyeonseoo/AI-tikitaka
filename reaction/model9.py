"""Reaction yield, plan 9 (reaction/README.md, "9차 계획"): tuned ranking (K0t) vs tuned regression (R0t).
Both models pick one of the same 30 LightGBM settings by inner leave-one-group-out on the recommendation metric,
then refit with 5 seeds. SM: leave one electrophile out (primary, regret@8). BH: leave one arene out (regret@3).
Usage: python reaction/model9.py   (-> reaction/results_model9.md)"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRanker, LGBMRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt  # noqa: E402
from model2 import N_CFG, SEED, desc_bh, desc_sm, lgbm_space  # noqa: E402
from model5 import judge  # noqa: E402
from model7 import CFG, grades  # noqa: E402
from model8 import cluster_ci, run_bh, run_sm  # noqa: E402
from prepare import load_bh, load_sm  # noqa: E402

OUT = ROOT / "results_model9.md"
CACHE9 = ROOT / "cache" / "model9"
CFGS = [lgbm_space(np.random.default_rng([SEED, 1, i])) for i in range(N_CFG)]  # same candidates as plans 2-4
KINDS = ("reg", "rank")


def fit_score(kind, p, X, y, q, tr, te, seed):
    params = {**CFG, **p, "random_state": seed}
    if kind == "reg":
        return LGBMRegressor(**params).fit(X[tr], y[tr]).predict(X[te])
    o = tr[np.argsort(q[tr], kind="stable")]
    sizes = pd.Series(q[o]).value_counts(sort=False).reindex(pd.unique(q[o])).to_numpy()
    return LGBMRanker(**params).fit(X[o], grades(y[o], q[o]), group=sizes).predict(X[te])


def regret(y, s, q, k):
    """Mean over queries of (best yield - best yield among the top k by score); ties by score order then index."""
    out = []
    for g in pd.unique(q):
        m = q == g
        yg = y[m]
        out.append(yg.max() - yg[np.argsort(-s[m], kind="stable")][:k].max())
    return out


def inner_score(kind, i, X, y, q, hold, tr, k):
    threadpool_limits(1)
    r = []
    for h in pd.unique(hold[tr]):
        itr, ite = tr[hold[tr] != h], tr[hold[tr] == h]
        r += regret(y[ite], fit_score(kind, CFGS[i], X, y, q, itr, ite, 0), q[ite], k)
    return np.mean(r)


def outer(name, X, y, q, hold, h, k):
    """Chosen config per model (cached) and the 5-seed test scores."""
    ck = CACHE9 / f"{name}_{h}.pkl".replace("*", "x").replace("(", "").replace(")", "")
    tr, te = np.flatnonzero(hold != h), np.flatnonzero(hold == h)
    if not ck.exists():
        jobs = [(kind, i) for kind in KINDS for i in range(N_CFG)]
        sc = Parallel(n_jobs=-1)(delayed(inner_score)(kind, i, X, y, q, hold, tr, k) for kind, i in jobs)
        best = {kind: min(range(N_CFG), key=lambda i: sc[jobs.index((kind, i))]) for kind in KINDS}
        test = {kind: np.mean(Parallel(n_jobs=5)(delayed(fit_score)(kind, CFGS[best[kind]], X, y, q, tr, te, s) for s in range(5)), 0)
                for kind in KINDS}
        CACHE9.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(pickle.dumps((best, test)))
    best, test = pickle.loads(ck.read_bytes())
    print(f"  {name} {h}: chosen {best}", flush=True)
    return te, best, test


def main():
    t0 = time.time()
    # SM, leave one electrophile out
    d = load_sm()
    y = d.y.to_numpy(float)
    pair = d[["electrophile", "nucleophile"]].agg(" + ".join, axis=1).to_numpy()
    hold = d.electrophile.to_numpy()
    X = desc_sm(d).to_numpy(float)
    rows, chosen = [], []
    for h in pd.unique(hold):
        te, best, test = outer("SM", X, y, pair, hold, h, 8)
        chosen.append({"데이터": "SM", "시험": h, "R0t 설정 번호": best["reg"], "K0t 설정 번호": best["rank"]})
        for p in pd.unique(pair[te]):
            m = pair[te] == p
            rows.append({"unit": p, "cluster": h, **{f"{n} regret@8": regret(y[te][m], test[kind][m], pair[te][m], 8)[0]
                                                     for n, kind in (("R0t", "reg"), ("K0t", "rank"))}})
    Ts = pd.DataFrame(rows)
    fixed = run_sm("electrophile", False).set_index("unit")  # plan 8 fixed-setting models, same split
    Ts = Ts.join(fixed[["R0 regret@8", "K0 regret@8"]], on="unit")

    # BH, leave one arene out
    b = load_bh()
    b["additive"] = b.additive.fillna("없음").astype(str)
    yb = b.y.to_numpy(float)
    ctx = (b.aryl_halide + " | " + b.additive).to_numpy()
    arene = b.arene.to_numpy()
    Xb = desc_bh(b).to_numpy(float)
    rows = []
    for h in pd.unique(arene):
        te, best, test = outer("BH", Xb, yb, ctx, arene, h, 3)
        chosen.append({"데이터": "BH", "시험": h, "R0t 설정 번호": best["reg"], "K0t 설정 번호": best["rank"]})
        rows.append({"unit": h, **{f"{n} regret@3": np.mean(regret(yb[te], test[kind], ctx[te], 3)) for n, kind in (("R0t", "reg"), ("K0t", "rank"))}})
    Tb = pd.DataFrame(rows).join(run_bh().set_index("unit")[["R0 regret@3", "K0 regret@3"]], on="unit")

    def d_(T, a, b):
        return boot_ci((T[a] - T[b]).to_numpy())

    prim = d_(Ts, "K0t regret@8", "R0t regret@8")
    V = [("**주**", "SM 처음 보는 출발물질 regret@8, K0t − R0t (15쌍)", prim),
         ("민감도", "같은 비교, 출발물질 7종 단위", cluster_ci(Ts, "K0t regret@8", "R0t regret@8")),
         ("2차", "SM 튜닝 효과, K0t − K0(고정)", d_(Ts, "K0t regret@8", "K0 regret@8")),
         ("2차", "SM 튜닝 효과, R0t − R0(고정)", d_(Ts, "R0t regret@8", "R0 regret@8")),
         ("2차", "BH 처음 보는 고리 regret@3, K0t − R0t (고리 5개)", d_(Tb, "K0t regret@3", "R0t regret@3"))]
    vt = "| 구분 | 비교 | 차이 (%p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|---|\n"
    for lab, name, t in V:
        vt += f"| {lab} | {name} | {fmt(t, True)} | {f'**{judge(t)}**' if lab == '**주**' else judge(t)} |\n"
    means = pd.DataFrame([{"데이터": "SM regret@8", **{c.split()[0]: round(Ts[c].mean(), 2) for c in ("R0 regret@8", "R0t regret@8", "K0 regret@8", "K0t regret@8")}},
                          {"데이터": "BH regret@3", **{c.split()[0]: round(Tb[c].mean(), 2) for c in ("R0 regret@3", "R0t regret@3", "K0 regret@3", "K0t regret@3")}}])

    OUT.write_text(
        "# 반응 수율 9차 결과: 튜닝한 순위 학습 vs 튜닝한 기준선 (generated by reaction/model9.py)\n\n"
        "R0 / K0 = 8차 고정 설정(수율 회귀 / 순위 학습), R0t / K0t = 같은 후보 30개에서 안쪽 그룹 하나 빼기로 고른 설정. "
        f"실행 {(time.time() - t0) / 60:.0f}분 (이어서 실행하면 마지막 구간만 셈).\n\n"
        f"## 판정\n{vt}\n## 평균\n{md(means, index=False)}\n\n"
        f"## 고른 설정 (후보 번호)\n{md(pd.DataFrame(chosen), index=False)}\n\n"
        f"## SM 쌍별 regret@8\n{md(Ts.drop(columns='cluster').round(2), index=False)}\n\n"
        f"## BH 고리별 regret@3\n{md(Tb.round(2), index=False)}\n", encoding="utf-8")
    print(vt)


if __name__ == "__main__":
    main()
