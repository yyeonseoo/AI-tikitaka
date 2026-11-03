"""Reaction yield, plan 8 (reaction/README.md, "8차 계획"): confirm the ranking model (K0) against the regression
baseline (R0) on a harder split, and test an ensemble-agreement reliability warning.
A (primary): SM leave-one-electrophile-out, regret@8 per test pair, K0 - R0. BH leave-one-arene-out as a secondary check.
B: 7 K0 members trained on bootstrap samples of the training pairs; agreement = mean overlap of each member's top 8
with the consensus top 8. Checked on the electrophile split and on the plan 7 pair split.
Usage: python reaction/model8.py   (-> reaction/results_model8.md)"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt  # noqa: E402
from model2 import desc_bh, desc_sm  # noqa: E402
from model5 import judge  # noqa: E402
from model6 import ranked_metrics  # noqa: E402
from model7 import CFG, grades, scores  # noqa: E402
from prepare import load_bh, load_sm  # noqa: E402

OUT = ROOT / "results_model8.md"
N_MEMBERS, TOP = 7, 8


def member(X, y, q, tr, i):
    """K0 trained on a bootstrap sample of the training queries (each draw gets its own query id)."""
    rng = np.random.default_rng(i)
    groups = pd.unique(q[tr])
    rows, qid = [], []
    for j, g in enumerate(rng.choice(groups, len(groups), replace=True)):
        r = tr[q[tr] == g]
        rows.append(r)
        qid.append(np.full(len(r), j))
    rows, qid = np.concatenate(rows), np.concatenate(qid)
    sizes = pd.Series(qid).value_counts(sort=False).reindex(pd.unique(qid)).to_numpy()
    return LGBMRanker(random_state=i, **CFG).fit(X[rows], grades(y[rows], qid), group=sizes)


def fold(X, y, q, hold, h, warn):
    threadpool_limits(1)
    tr, te = np.flatnonzero(hold != h), np.flatnonzero(hold == h)
    out = {"R0": scores("reg", X[tr], y[tr], q[tr], X[te]), "K0": scores("rank", X[tr], y[tr], q[tr], X[te])}
    if warn:
        out["members"] = np.stack([member(X, y, q, tr, i).predict(X[te]) for i in range(N_MEMBERS)])
    return te, out


def agreement(M):
    cons = set(np.argsort(-M.mean(0))[:TOP])
    return np.mean([len(cons & set(np.argsort(-m)[:TOP])) / TOP for m in M])


def run_sm(hold_col, warn):
    d = load_sm()
    y = d.y.to_numpy(float)
    pair = d[["electrophile", "nucleophile"]].agg(" + ".join, axis=1).to_numpy()
    hold = pair if hold_col == "pair" else d[hold_col].to_numpy()
    X = desc_sm(d).to_numpy(float)
    res = Parallel(n_jobs=-1)(delayed(fold)(X, y, pair, hold, h, warn) for h in pd.unique(hold))
    tie = np.random.default_rng(0)
    rows = []
    for te, out in res:
        for p in pd.unique(pair[te]):
            k = pair[te] == p
            yp = y[te][k]
            r = {"unit": p, "cluster": hold[te][k][0]}
            for m in ("R0", "K0"):
                s = out[m][k]
                r.update({f"{m} {c}": v for c, v in ranked_metrics(yp, yp[np.lexsort((tie.random(len(s)), -s))]).items()})
            if warn:
                r["합의율"] = agreement(out["members"][:, k])
            rows.append(r)
    return pd.DataFrame(rows)


def run_bh():
    d = load_bh()
    d["additive"] = d.additive.fillna("없음").astype(str)
    y = d.y.to_numpy(float)
    ctx = (d.aryl_halide + " | " + d.additive).to_numpy()
    arene = d.arene.to_numpy()
    X = desc_bh(d).to_numpy(float)
    res = Parallel(n_jobs=-1)(delayed(fold)(X, y, ctx, arene, h, False) for h in pd.unique(arene))
    tie = np.random.default_rng(0)
    rows = []
    for te, out in res:
        r = {"unit": arene[te][0]}
        for m in ("R0", "K0"):
            reg = []
            for c in pd.unique(ctx[te]):
                k = ctx[te] == c
                yc, s = y[te][k], out[m][k]
                reg.append(yc.max() - yc[np.lexsort((tie.random(len(s)), -s))][:3].max())
            r[f"{m} regret@3"] = np.mean(reg)
        rows.append(r)
    return pd.DataFrame(rows)


def cluster_ci(T, col_a, col_b, b=1000):
    g = (T[col_a] - T[col_b]).groupby(T.cluster)
    sums, ns = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(7)
    s = []
    for _ in range(b):
        i = rng.integers(0, len(sums), len(sums))
        s.append(sums[i].sum() / ns[i].sum())
    return (sums.sum() / ns.sum(), np.quantile(s, 0.025), np.quantile(s, 0.975))


def warning_rows(T, name):
    r, a = T["K0 regret@8"].to_numpy(), T["합의율"].to_numpy()
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(r), len(r))
        boots.append(spearmanr(a[i], r[i])[0])
    warned = a < np.median(a)
    rho = spearmanr(a, r)[0]
    return {"분할": name, "합의율–regret 순위 상관": f"{rho:+.2f} [{np.nanquantile(boots, 0.025):+.2f}, {np.nanquantile(boots, 0.975):+.2f}]",
            "경고 단위 수": int(warned.sum()), "경고 단위 평균 regret@8": round(r[warned].mean(), 2),
            "경고 없는 단위 평균 regret@8": round(r[~warned].mean(), 2)}


def main():
    t0 = time.time()
    Te = run_sm("electrophile", True)
    Tp = run_sm("pair", True)
    Tb = run_bh()

    prim = boot_ci((Te["K0 regret@8"] - Te["R0 regret@8"]).to_numpy())
    clus = cluster_ci(Te, "K0 regret@8", "R0 regret@8")
    bh = boot_ci((Tb["K0 regret@3"] - Tb["R0 regret@3"]).to_numpy())
    vt = ("| 구분 | 비교 | 차이 (%p, 음수 = K0가 좋음) | 결과 |\n|---|---|---|---|\n"
          f"| **주** | SM 처음 보는 출발물질, regret@8, K0 − R0 (15쌍 부트스트랩) | {fmt(prim, True)} | **{judge(prim)}** |\n"
          f"| 민감도 | 같은 비교, 출발물질 7종 단위 부트스트랩 | {fmt(clus, True)} | {judge(clus)} |\n"
          f"| 2차 | BH 처음 보는 고리, regret@3, K0 − R0 | {fmt(bh, True)} | {judge(bh)} |\n")

    def summ(T):
        rows = []
        for m in ("R0", "K0"):
            rows.append({"방법": {"R0": "R0 수율 회귀 (기준선)", "K0": "K0 순위 학습"}[m],
                         **{c: f"{T[f'{m} {c}'].mean():.0%}" if c.startswith("hit") else f"{T[f'{m} {c}'].mean():.2f}"
                            for c in ("regret@1", "regret@4", "regret@8", "regret@20", "hit@4", "hit@8", "hit@20")}})
        return pd.DataFrame(rows)

    per = Te[["unit", "R0 regret@8", "K0 regret@8", "합의율"]].round(2)
    warn = pd.DataFrame([warning_rows(Te, "처음 보는 출발물질"), warning_rows(Tp, "처음 보는 쌍 (7차 분할)")])
    bh_tab = Tb.round(2)

    OUT.write_text(
        "# 반응 수율 8차 결과: 순위 학습 확인 + 신뢰도 경고 (generated by reaction/model8.py)\n\n"
        "K0 = 순위 학습, R0 = 수율 회귀 (7차와 같은 코드·설정). 합의율 = 복원 추출로 학습한 K0 구성원 7개의 상위 8개가 "
        f"합의 상위 8개와 겹치는 비율의 평균. 실행 {(time.time() - t0) / 60:.1f}분.\n\n"
        f"## 판정\n{vt}\n"
        f"## SM 처음 보는 출발물질 (15쌍 평균)\n{md(summ(Te), index=False)}\n\n"
        f"## 신뢰도 경고 (경고 = 합의율이 중앙값보다 낮음)\n{md(warn, index=False)}\n\n"
        f"## SM 처음 보는 출발물질, 쌍별\n{md(per, index=False)}\n\n"
        f"## BH 처음 보는 고리 (고리별)\n{md(bh_tab, index=False)}\n", encoding="utf-8")
    print(vt)
    print(warn.to_string())


if __name__ == "__main__":
    main()
