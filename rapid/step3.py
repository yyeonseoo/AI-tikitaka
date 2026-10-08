"""RAPID step 3 (plan and verdict criteria: rapid/README.md, "3단계 확정 사항"): condition recommendation.
A: known amine (per-amine 60/20/20 split). B: new amine (amine-level 5 outer folds; inside, 60/20 train/val amines),
k shots of the new amine added to training, with a stopping rule. C: zero-shot amine success rate from amine geometry.
Usage: python rapid/step3.py   (-> rapid/results_step3.md, rapid/figures/step3_curve.png)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import amine_table, data  # noqa: E402
from step1 import AMINE, SEED, md  # noqa: E402

OUT, FIG = ROOT / "results_step3.md", ROOT / "figures"
CONFIGS = [{"num_leaves": nl, "n_estimators": ne} for nl in (15, 31, 63) for ne in (200, 400)]
WEIGHTS = [1, 5, 10, 20]
KS = list(range(4, 33, 4))
REPS, TUNE_REPS, N_BOOT, TOP, STOP_RHO = 10, 2, 1000, 0.2, 0.9
GEO = ["eff_radius", "_feat_MinimalProjectionArea", "_feat_RotatableBondCount"]


def lgbm(cfg):
    return LGBMClassifier(learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=1, **cfg)


def prepare():
    d, feat = data()
    d = d[(d._rxn_M_inorganic > 0) & (d._rxn_M_organic > 0) & d["용매"].isin(["GBL", "DMF", "DMSO"])].reset_index(drop=True)
    desc = [f for f in feat if d.groupby(AMINE)[f].first().nunique() > 1]
    X = ["_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid", "amine_pb", "acid_pb", "_rxn_temperatureC_actual_bulk",
         "_rxn_reactiontimeS", "_rxn_stirrateRPM"] + desc
    return d, feat, X


def gain(y, p):
    """Recommendation gain: 4-point share in the top 20% predicted minus the share in all scored rows."""
    k = max(1, int(round(TOP * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    return y[top].mean() - y.mean()


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def per_amine(df, pcol="p"):
    """Per-amine gain and AUC on scored rows; amines without any 4-point row are skipped."""
    out = {}
    for g, s in df.groupby(AMINE):
        y, p = s.y.to_numpy(), s[pcol].to_numpy()
        if y.sum() > 0:
            out[g] = {"gain": gain(y, p), "auc": auc(y, p)}
    return pd.DataFrame(out).T


def boot(v, idx):
    v = np.asarray(v, float)
    b = np.array([np.nanmean(v[i]) for i in idx])
    return np.nanmean(v), np.nanquantile(b, 0.025), np.nanquantile(b, 0.975)


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


# ---------------- A: known amine ----------------
def part_a(d, X):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(d))
    pos = pd.Series(np.arange(len(d)), index=order).sort_index()  # random rank of each row
    fold = d.assign(r=pos).sort_values("r").groupby([AMINE, "y"]).cumcount().reindex(d.index) % 5
    role = np.where(fold < 3, "train", np.where(fold == 3, "val", "test"))
    tr, va, te = (d[role == r] for r in ("train", "val", "test"))

    def score(cfg):
        p = lgbm(cfg).fit(tr[X], tr.y).predict_proba(va[X])[:, 1]
        return per_amine(va.assign(p=p)).auc.mean()
    val = Parallel(n_jobs=-1)(delayed(score)(c) for c in CONFIGS)
    best = CONFIGS[int(np.nanargmax(val))]
    trva = pd.concat([tr, va])
    p = lgbm(best).fit(trva[X], trva.y).predict_proba(te[X])[:, 1]
    T = te.assign(p=p)
    return per_amine(T), per_amine(T[~T.model_chosen]), best, {"train": len(tr), "val": len(va), "test": len(te)}


# ---------------- B: new amine ----------------
def shots_runs(train, s, cfg, w, perm, ks, X):
    """Retrain with the first k rows of `perm` from new amine s (weight w); predict every row of s for each k."""
    preds = {}
    for k in ks:
        sh = s.iloc[perm[:k]]
        m = lgbm(cfg).fit(pd.concat([train[X], sh[X]]), np.concatenate([train.y, sh.y]),
                          sample_weight=np.concatenate([np.ones(len(train)), np.full(k, w)]))
        preds[k] = m.predict_proba(s[X])[:, 1]
    return preds


def kshot_only(s, perm, k, X):
    sh = s.iloc[perm[:k]]
    if sh.y.min() == sh.y.max():
        return np.full(len(s), (sh.y.sum() + 0.5) / (k + 1))
    cols = X[:8]  # reaction inputs only (amine descriptors are constant within one amine)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(sh[cols], sh.y)
    return m.predict_proba(s[cols])[:, 1]


def tune_b(train, val_amines, d, fold, X):
    def one(cfg, w):
        aucs = []
        for g in val_amines:
            s = d[d[AMINE] == g].reset_index(drop=True)
            if s.y.sum() == 0:
                continue
            for r in range(TUNE_REPS):
                perm = np.random.default_rng([SEED, 99, fold, r]).permutation(len(s))
                p = shots_runs(train, s, cfg, w, perm, [8], X)[8]
                rest = perm[8:]
                aucs.append(auc(s.y.to_numpy()[rest], p[rest]))
        return np.nanmean(aucs)
    grid = [(c, w) for c in CONFIGS for w in WEIGHTS]
    val = Parallel(n_jobs=-1)(delayed(one)(c, w) for c, w in grid)
    return grid[int(np.nanargmax(val))]


def test_amine(train, s, g, cfg, w, fold, X):
    """Every rep: nested shots k = 4..32 (same order), stopping rule, k-shot-only baseline."""
    rows = []
    y = s.y.to_numpy()
    p0 = lgbm(cfg).fit(train[X], train.y).predict_proba(s[X])[:, 1]
    ks = [k for k in KS if len(s) - k >= 10]
    for r in range(REPS):
        perm = np.random.default_rng([SEED, 7, fold, r, len(s)]).permutation(len(s))
        P = {0: p0, **shots_runs(train, s, cfg, w, perm, ks, X)}
        for k in [0] + ks:
            rest = perm[k:]
            rows.append({"amine": g, "rep": r, "k": k, "method": "retrain", "gain": gain(y[rest], P[k][rest]), "auc": auc(y[rest], P[k][rest])})
            if k > 0:
                q = kshot_only(s, perm, k, X)
                rows.append({"amine": g, "rep": r, "k": k, "method": "kshot", "gain": gain(y[rest], q[rest]), "auc": auc(y[rest], q[rest])})
        # stopping rule: first k where the ranking of not-yet-tested rows agrees with the previous model (rho >= STOP_RHO)
        stop = ks[-1]
        for a, b in zip([0] + ks[:-1], ks):
            rest = perm[b:]
            if spearmanr(P[a][rest], P[b][rest]).statistic >= STOP_RHO:
                stop = b
                break
        common = perm[ks[-1]:]
        rows.append({"amine": g, "rep": r, "k": stop, "method": "stop", "gain": np.nan,
                     "auc_stop": auc(y[common], P[stop][common]), "auc_last": auc(y[common], P[ks[-1]][common]), "k_last": ks[-1]})
    return rows


def part_b(d, X):
    amines = np.array(sorted(d[AMINE].unique()))
    rng = np.random.default_rng(SEED)
    outer = {g: i % 5 for i, g in enumerate(rng.permutation(amines))}
    rows, chosen = [], []
    for f in range(5):
        test_a = [g for g in amines if outer[g] == f]
        rest_a = rng.permutation([g for g in amines if outer[g] != f])
        n_val = int(round(len(rest_a) * 0.25))
        val_a, tr_a = rest_a[:n_val], rest_a[n_val:]
        cfg, w = tune_b(d[d[AMINE].isin(tr_a)], val_a, d, f, X)
        chosen.append({"fold": f, "test amines": len(test_a), "train amines": len(tr_a), "val amines": len(val_a), **cfg, "weight": w})
        train = d[d[AMINE].isin(np.concatenate([tr_a, val_a]))]
        res = Parallel(n_jobs=-1)(delayed(test_amine)(train, d[d[AMINE] == g].reset_index(drop=True), g, cfg, w, f, X)
                                  for g in test_a if d.loc[d[AMINE] == g, "y"].sum() > 0)
        rows += [x for r in res for x in r]
    return pd.DataFrame(rows), pd.DataFrame(chosen), outer


# ---------------- C: zero-shot amine success rate ----------------
def part_c(d, feat, outer):
    A = amine_table(d, feat)
    t = np.log((A.rate * A.n + 0.5) / ((1 - A.rate) * A.n + 0.5))  # smoothed logit of the 4-point share
    desc = [f for f in feat if A[f].nunique() > 1]
    sets = {"크기 1개 (유효 반지름)": ["eff_radius"], "기하 3개 (주)": GEO, f"물성 전체 ({len(desc)}개)": desc}
    pred = {}
    for name, cols in sets.items():
        p = pd.Series(index=A.index, dtype=float)
        for f in range(5):
            te = [g for g in A.index if outer[g] == f]
            tr = [g for g in A.index if outer[g] != f]
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100])).fit(A.loc[tr, cols], t[tr])  # LOO alpha
            p[te] = m.predict(A.loc[te, cols])
        pred[name] = p
    return A, pred


def main():
    FIG.mkdir(exist_ok=True)
    d, feat, XG = prepare()
    s = [f"# RAPID step 3 results (generated by rapid/step3.py)\n\n{len(d)} experiments after exclusions, "
         f"{d[AMINE].nunique()} amines, 4-point share {d.y.mean():.3f}. Gain = 4-point share in the top 20% recommended "
         f"minus the share in all scored rows (0 = random); cells: mean over amines [95% amine bootstrap, {N_BOOT}].\n"]
    rng = np.random.default_rng(SEED)

    # A
    RA, RA_s, best, sizes = part_a(d, X=XG)
    ia = [rng.integers(0, len(RA), len(RA)) for _ in range(N_BOOT)]
    ga, aa = boot(RA.gain, ia), boot(RA.auc, ia)
    ia_s = [rng.integers(0, len(RA_s), len(RA_s)) for _ in range(N_BOOT)]
    va = "충족" if ga[1] > 0 else "미달"
    s.append(f"## A. 이미 실험한 아민 (판정: **{va}**)\nsplit {sizes}, chosen {best}, amines scored {len(RA)}\n\n"
             f"- 추천 이득 {fmt(ga)}, AUC {fmt(aa)}\n"
             f"- 민감도 (원 연구팀 모델이 고른 실험 제외): 추천 이득 {fmt(boot(RA_s.gain, ia_s))}, AUC {fmt(boot(RA_s.auc, ia_s))}\n")
    sol = d.groupby(AMINE)["용매"].agg(lambda v: v.mode()[0])
    by_sol = RA.assign(sol=sol.reindex(RA.index)).groupby("sol")[["gain", "auc"]].agg(["mean", "size"]).round(3)
    s.append(f"용매별 (주 용매 기준): \n\n{md(by_sol)}\n")

    # B
    RB, chosen, outer = part_b(d, XG)
    per = RB[RB.method != "stop"].groupby(["amine", "k", "method"])[["gain", "auc"]].mean()
    amines_b = sorted(RB.amine.unique())
    ib = [rng.integers(0, len(amines_b), len(amines_b)) for _ in range(N_BOOT)]

    def vec(k, m, mt):
        return per.xs((k, m), level=["k", "method"])[mt].reindex(amines_b).to_numpy()
    curve = []
    for k in [0] + KS:
        for m in ("retrain", "kshot"):
            if k == 0 and m == "kshot":
                continue
            curve.append({"k": k, "방법": {"retrain": "재학습 (다른 아민 + k번)", "kshot": "새 아민 k번만"}[m],
                          "추천 이득": fmt(boot(vec(k, m, "gain"), ib)), "AUC": fmt(boot(vec(k, m, "auc"), ib))})
    g8 = boot(vec(8, "retrain", "gain"), ib)
    d8 = boot(vec(8, "retrain", "gain") - vec(8, "kshot", "gain"), ib)
    vb = "충족" if g8[1] > 0 and d8[1] > 0 else "미달"
    st = RB[RB.method == "stop"].groupby("amine")[["k", "auc_stop", "auc_last"]].mean().reindex(amines_b)
    drop = boot(st.auc_stop - st.auc_last, ib)
    kst = boot(st.k, ib)
    v2 = "충족" if drop[0] > -0.03 and kst[0] < 32 else "미달"
    s.append(f"## B. 처음 보는 아민 (판정: **{vb}**)\n{len(amines_b)} amines with any 4-point row; {REPS} random shot orders each.\n\n"
             f"- k = 8 재학습 추천 이득 {fmt(g8)}; 재학습 − 새 아민 8번만 {fmt(d8)}\n\n"
             f"{md(chosen, index=False)}\n\n{md(pd.DataFrame(curve), index=False)}\n\n"
             f"### ② 멈출 시점 (판정: **{v2}**)\n순위 상관 ≥ {STOP_RHO}이면 멈춤. 평균 실험 수 {kst[0]:.1f} [{kst[1]:.1f}, {kst[2]:.1f}], "
             f"멈춘 시점 AUC − 마지막 시점 AUC {fmt(drop)} (같은 남은 조건에서 비교)\n")

    # C
    A, pred = part_c(d, feat, outer)
    ic = [rng.integers(0, len(A), len(A)) for _ in range(N_BOOT)]
    rows = []
    for name, p in pred.items():
        r = spearmanr(p, A.rate).statistic
        b = [spearmanr(p.iloc[i], A.rate.iloc[i]).statistic for i in ic]
        rows.append({"입력": name, "순위 상관": round(r, 3), "2.5%": round(np.nanquantile(b, 0.025), 3), "97.5%": round(np.nanquantile(b, 0.975), 3)})
    C = pd.DataFrame(rows)
    vc = "충족" if C.iloc[1]["2.5%"] > 0 else "미달"
    s.append(f"## C. 실험 0번 아민 성공률 예측 (③, 판정: **{vc}**)\n아민 {len(A)}종, B와 같은 5묶음.\n\n{md(C, index=False)}\n")

    # figure: gain and AUC vs k
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, mt, lab in ((ax[0], "gain", "추천 이득 (상위 20% 4점 비율 − 전체)"), (ax[1], "auc", "아민 안 AUC")):
        for m, col, name in (("retrain", BLUE, "재학습 (다른 아민 + k번)"), ("kshot", ORANGE, "새 아민 k번만")):
            ks = ([0] if m == "retrain" else []) + KS
            v = [boot(vec(k, m, mt), ib) for k in ks]
            a.plot(ks, [x[0] for x in v], "-o", color=col, lw=2, ms=6, label=name)
            a.fill_between(ks, [x[1] for x in v], [x[2] for x in v], color=col, alpha=0.15, lw=0)
        a.axhline(0 if mt == "gain" else 0.5, color="#9a9893", ls="--", lw=1)
        a.set(xlabel="새 아민으로 먼저 해 본 실험 수 k", ylabel=lab, xticks=[0] + KS)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=8, frameon=False)
    fig.suptitle("처음 보는 아민: 먼저 해 본 실험 수에 따른 추천 성능 (선 = 평균, 띠 = 아민 부트스트랩 95% 구간)")
    fig.tight_layout()
    fig.savefig(FIG / "step3_curve.png", dpi=110)
    plt.close(fig)

    OUT.write_text("\n".join(s) + "\n그림: [figures/step3_curve.png](figures/step3_curve.png)\n", encoding="utf-8")
    print(f"A {va} | B {vb} | stop {v2} | C {vc}")
    print("\n".join(s[1:]))


if __name__ == "__main__":
    main()
