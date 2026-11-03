"""Reaction yield, plan 7 (reaction/README.md, "7차 계획"): recommendation engine = LambdaRank + history-prior feature,
plus a domain warning (distance of the test substrate to the training substrates). Ablation against the plan 6 LightGBM.
SM (primary): leave one (electrophile, nucleophile) pair out, rank its 384 conditions, regret@8.
BH (secondary): leave one aryl halide out, rank ligand x base inside each (substrate, additive) context, regret@3.
Usage: python reaction/model7.py   (-> reaction/results_model7.md)"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRanker, LGBMRegressor
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt, target_means  # noqa: E402
from model2 import desc_bh, desc_sm  # noqa: E402
from model5 import judge  # noqa: E402
from model6 import ranked_metrics  # noqa: E402
from prepare import load_bh, load_sm  # noqa: E402

OUT = ROOT / "results_model7.md"
CFG = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, verbose=-1, n_jobs=1)
VARIANTS = {"R0 (기준선: 수율 회귀)": ("reg", False), "R1 (회귀 + 과거 평균)": ("reg", True),
            "K0 (순위 학습)": ("rank", False), "K1 (엔진: 순위 학습 + 과거 평균)": ("rank", True)}
R0, R1, K0, K1 = VARIANTS
GRADES = 30


def grades(y, q):
    """Relevance 0..30 = round(30 * yield / best yield of its query); all 0 when the best is 0."""
    m = pd.Series(y).groupby(q).transform("max").to_numpy()
    return np.where(m > 0, np.rint(GRADES * y / np.where(m > 0, m, 1)), 0).astype(int)


def scores(kind, Xtr, ytr, qtr, Xte):
    """Mean score of 5 seeds. Ranker rows are sorted by query so each query is contiguous."""
    out = []
    for s in range(5):
        if kind == "reg":
            out.append(LGBMRegressor(random_state=s, **CFG).fit(Xtr, ytr).predict(Xte))
        else:
            o = np.argsort(qtr, kind="stable")
            sizes = pd.Series(qtr[o]).value_counts(sort=False).reindex(pd.unique(qtr[o])).to_numpy()
            out.append(LGBMRanker(random_state=s, **CFG).fit(Xtr[o], grades(ytr[o], qtr[o]), group=sizes).predict(Xte))
    return np.mean(out, 0)


def fold(X, d, y, hold, query, cond, sub, h):
    """Scores of every variant for the rows of held-out substrate h."""
    threadpool_limits(1)
    tr, te = np.flatnonzero(hold != h), np.flatnonzero(hold == h)
    prior_tr = target_means(d.iloc[tr], d.iloc[tr], cond, oof_groups=sub[tr])
    prior_te = target_means(d.iloc[tr], d.iloc[te], cond)
    out = {}
    for name, (kind, prior) in VARIANTS.items():
        Xtr = np.column_stack([X[tr], prior_tr]) if prior else X[tr]
        Xte = np.column_stack([X[te], prior_te]) if prior else X[te]
        out[name] = scores(kind, Xtr, y[tr], query[tr], Xte)
    return h, te, out


def distances(X, cols, hold):
    """Per held-out substrate: nearest standardised distance to the training substrates' descriptor vectors."""
    V = pd.DataFrame(X[:, cols]).groupby(hold).first()
    out = {}
    for h in V.index:
        tr = V.drop(index=h)
        mu, sd = tr.mean(), tr.std().replace(0, 1)
        out[h] = np.sqrt((((tr - mu) / sd - (V.loc[h] - mu) / sd) ** 2).sum(1)).min()
    return pd.Series(out)


def diff(T, a, b, col):
    A, B = (T[T.method == m].set_index("unit")[col] for m in (a, b))
    return boot_ci((A - B.loc[A.index]).to_numpy())


def run_sm():
    d = load_sm()
    y = d.y.to_numpy(float)
    pair = d[["electrophile", "nucleophile"]].agg(" + ".join, axis=1).to_numpy()
    D = desc_sm(d)
    X = D.to_numpy(float)
    res = Parallel(n_jobs=-1)(delayed(fold)(X, d, y, pair, pair, ["ligand", "base", "solvent"], pair, h) for h in pd.unique(pair))
    tie = np.random.default_rng(0)
    rows = []
    for h, te, out in res:
        for m, q in out.items():
            rows.append({"method": m, "unit": h, **ranked_metrics(y[te], y[te][np.lexsort((tie.random(len(q)), -q))])})
    cols = [i for i, c in enumerate(D.columns) if c.startswith(("electrophile:", "nucleophile:"))]
    return pd.DataFrame(rows), distances(X, cols, pair)


def run_bh():
    d = load_bh()
    d["additive"] = d.additive.fillna("없음").astype(str)
    y = d.y.to_numpy(float)
    sub = d.aryl_halide.to_numpy()
    ctx = (d.aryl_halide + " | " + d.additive).to_numpy()
    D = desc_bh(d)
    X = D.to_numpy(float)
    res = Parallel(n_jobs=-1)(delayed(fold)(X, d, y, sub, ctx, ["ligand", "base", "additive"], sub, h) for h in pd.unique(sub))
    tie = np.random.default_rng(0)
    rows = []
    for h, te, out in res:
        for m, q in out.items():
            r = []
            for c in pd.unique(ctx[te]):
                k = ctx[te] == c
                yc, qc = y[te][k], q[k]
                top = yc[np.lexsort((tie.random(len(qc)), -qc))][:3].max()
                r.append((yc.max() - top, float(top >= yc.max() - 5)))
            rows.append({"method": m, "unit": h, "regret@3": np.mean([a for a, _ in r]), "hit@3": np.mean([b for _, b in r])})
    cols = [i for i, c in enumerate(D.columns) if c.startswith("aryl_halide:")]
    return pd.DataFrame(rows), distances(X, cols, sub)


def main():
    t0 = time.time()
    Ts, dist_s = run_sm()
    Tb, dist_b = run_bh()

    V = [("**주**", "SM", Ts, K1, R0, "regret@8")] + [("2차", "SM", Ts, a, b, "regret@8") for a, b in ((K0, R0), (R1, R0), (K1, K0), (K1, R1))] + \
        [("2차", "BH", Tb, a, R0, "regret@3") for a in (K1, K0, R1)]
    vt = "| 구분 | 데이터 | 비교 | 차이 (%p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|---|---|\n"
    for lab, name, T, a, b, col in V:
        t = diff(T, a, b, col)
        vt += f"| {lab} | {name} {col} | {a} − {b} | {fmt(t, True)} | {f'**{judge(t)}**' if lab == '**주**' else judge(t)} |\n"

    def table(T, cols):
        rows = []
        for m in VARIANTS:
            t = T[T.method == m]
            rows.append({"방법": m, **{c: f"{t[c].mean():.0%}" if c.startswith("hit") else f"{t[c].mean():.2f}" for c in cols}})
        return pd.DataFrame(rows)

    sm_cols = ["regret@1", "regret@4", "regret@8", "regret@20", "hit@4", "hit@8", "hit@20"]
    per_pair = Ts.pivot(index="unit", columns="method", values="regret@8")[list(VARIANTS)].round(2)
    per_pair.insert(0, "거리", dist_s.round(2))
    warn = []
    for name, T, dist, col in (("SM (15쌍)", Ts, dist_s, "regret@8"), ("BH (15종)", Tb, dist_b, "regret@3")):
        r = T[T.method == K1].set_index("unit")[col]
        warn.append({"데이터": name, "거리와 엔진 regret의 순위 상관": round(spearmanr(dist.loc[r.index], r)[0], 2)})

    OUT.write_text(
        "# 반응 수율 7차 결과: 추천 엔진 개발 (generated by reaction/model7.py)\n\n"
        "변형 4개 모두 RDKit 분자 특성 F1, 고정 설정, 시드 5개 평균. SM은 출발물질 쌍 15개, BH는 출발물질 15종을 하나씩 빼고 학습. "
        f"구간은 쌍(종) 단위 부트스트랩 1,000회. hit = 최고와 5%p 이내 조건 포함. 실행 {(time.time() - t0) / 60:.1f}분.\n\n"
        f"## 판정\n{vt}\n"
        f"## SM (조건 384개)\n{md(table(Ts, sm_cols), index=False)}\n\n"
        f"## BH (상황마다 조건 11~12개)\n{md(table(Tb, ['regret@3', 'hit@3']), index=False)}\n\n"
        f"## 신뢰도 경고 확인 (기술 보고)\n{md(pd.DataFrame(warn), index=False)}\n\n"
        f"## SM 쌍별 regret@8 (거리 = 학습 쌍과의 최근접 표준화 거리)\n{md(per_pair.reset_index(), index=False)}\n", encoding="utf-8")
    print(vt)


if __name__ == "__main__":
    main()
