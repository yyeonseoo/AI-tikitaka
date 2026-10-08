"""RAPID step 2 (plan: rapid/README.md, "2단계 계획"): a new amine with only k experiments.
Leave one amine out; reveal k of its experiments; score the rest. Methods: cross-amine LightGBM as is, k-shot only,
cross-amine LightGBM recalibrated on the k shots (main), retrained with the k shots added. Amine-level bootstrap.
Usage: python rapid/step2.py   (-> rapid/results_step2.md)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from step1 import AMINE, RXN, SEED, load, md  # noqa: E402

OUT = ROOT / "results_step2.md"
KS, REPS, N_BOOT, ADD_WEIGHT, PRIOR = [4, 8, 16, 32], 20, 1000, 10.0, 1.0
RATIO = ["amine_pb_ratio", "acid_pb_ratio"]
METHODS = {"none": "보정 없음", "kshot": "새 아민 k번만", "recal": "보정 (주)", "addtrain": "추가 학습", "recal_unc": "보정, 불확실한 조건부터 (Q4)"}


def lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1, n_jobs=1)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recalibrate(z_k, y_k, z):
    """logit p = a + b z, fitted on the k shots with a Gaussian prior centred on (a, b) = (0, 1), i.e. 'trust the
    cross-amine model unless the shots say otherwise'. Works even when all k shots have the same outcome."""
    def loss(w):
        s = w[0] + w[1] * z_k
        return np.sum(np.logaddexp(0, s) - y_k * s) + PRIOR * (w[0] ** 2 + (w[1] - 1) ** 2) / 2
    a, b = minimize(loss, [0.0, 1.0]).x
    return 1 / (1 + np.exp(-(a + b * z)))


def kshot(X_k, y_k, X):
    """New-amine data only. One outcome seen -> its smoothed rate as a constant."""
    if y_k.min() == y_k.max():
        return np.full(len(X), (y_k.sum() + 0.5) / (len(y_k) + 1))
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X_k, y_k)
    return m.predict_proba(X)[:, 1]


def scores(y, p):
    return {"mcc": matthews_corrcoef(y, p >= 0.5), "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
            "brier": brier_score_loss(y, p)}


def one_amine(i, g, d, y, cols_full, cols_rxn):
    """Every method, every k and repeat, for held-out amine g. Returns per-repeat metric rows."""
    tr, te = np.where(d[AMINE].values != g)[0], np.where(d[AMINE].values == g)[0]
    Xtr, Xte, ytr, yte = d.iloc[tr][cols_full], d.iloc[te][cols_full], y[tr], y[te]
    base = lgbm().fit(Xtr, ytr)
    p0 = base.predict_proba(Xte)[:, 1]
    p0_rxn = lgbm().fit(d.iloc[tr][cols_rxn], ytr).predict_proba(d.iloc[te][cols_rxn])[:, 1]
    rows = [{"amine": g, "k": 0, "rep": 0, "method": "none", **scores(yte, p0)},
            {"amine": g, "k": 0, "rep": 0, "method": "none_no_ratio", **scores(yte, p0_rxn)}]
    small = RXN + RATIO  # k-shot model: reaction inputs only (amine descriptors are constant within one amine)
    for k in KS:
        unc = np.argsort(np.abs(p0 - 0.5), kind="stable")[:k]  # Q4: the k conditions the model is least sure about
        draws = [np.random.default_rng([SEED, i, k, r]).choice(len(te), k, replace=False) for r in range(REPS)]
        for r, shot in enumerate(draws + [unc]):
            rest = np.setdiff1d(np.arange(len(te)), shot)
            yk, yr = yte[shot], yte[rest]
            if r == REPS:  # uncertainty-chosen shots, recalibration only
                rows.append({"amine": g, "k": k, "rep": 0, "method": "recal_unc",
                             **scores(yr, recalibrate(logit(p0[shot]), yk, logit(p0[rest])))})
                continue
            add = lgbm().fit(pd.concat([Xtr, Xte.iloc[shot]]), np.concatenate([ytr, yk]),
                             sample_weight=np.concatenate([np.ones(len(tr)), np.full(k, ADD_WEIGHT)]))
            preds = {"none": p0[rest],
                     "kshot": kshot(d.iloc[te[shot]][small], yk, d.iloc[te[rest]][small]),
                     "recal": recalibrate(logit(p0[shot]), yk, logit(p0[rest])),
                     "addtrain": add.predict_proba(Xte.iloc[rest])[:, 1]}
            rows += [{"amine": g, "k": k, "rep": r, "method": m, **scores(yr, p)} for m, p in preds.items()]
    return rows


def amine_rate_model(d, y, feat):
    """Q1: amine descriptors -> amine success rate, leave one amine out (Ridge on standardized descriptors)."""
    A = d.groupby(AMINE)[feat].first()
    rate = pd.Series(y, index=d.index).groupby(d[AMINE]).mean().reindex(A.index)
    pred = pd.Series(index=A.index, dtype=float)
    for g in A.index:
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(A.drop(g), rate.drop(g))
        pred[g] = m.predict(A.loc[[g]])[0]
    return rate, pred


def boot_mean(v, idx):
    return np.array([np.nanmean(v[i]) for i in idx])


def main():
    d, feat, y = load()
    d["amine_pb_ratio"] = d._rxn_M_organic / d._rxn_M_inorganic.replace(0, np.nan)
    d["acid_pb_ratio"] = d._rxn_M_acid / d._rxn_M_inorganic.replace(0, np.nan)
    cols_full, cols_rxn = RXN + RATIO + feat, RXN + feat
    amines = sorted(d[AMINE].unique())
    res = Parallel(n_jobs=-1)(delayed(one_amine)(i, g, d, y, cols_full, cols_rxn) for i, g in enumerate(amines))
    R = pd.DataFrame([r for rows in res for r in rows])
    A = R.groupby(["amine", "k", "method"])[["mcc", "auc", "brier"]].mean()  # average the repeats per amine

    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(amines), len(amines)) for _ in range(N_BOOT)]

    def vec(k, m, metric):
        return A.xs((k, m), level=["k", "method"])[metric].reindex(amines).to_numpy()

    def ci(v):
        b = boot_mean(v, idx)
        return f"{np.nanmean(v):.3f} [{np.nanquantile(b, 0.025):.3f}, {np.nanquantile(b, 0.975):.3f}]"

    curve = []
    for k in KS:
        for m in ["none", "kshot", "recal", "addtrain", "recal_unc"]:
            curve.append({"k": k, "방법": METHODS[m], **{mt: ci(vec(k, m, mt)) for mt in ("mcc", "auc", "brier")}})
    curve = pd.DataFrame(curve)

    def diff(k, a, b, metric="mcc"):
        v = vec(k, a, metric) - vec(k, b, metric)
        bd = boot_mean(v, idx)
        lo, hi = np.nanquantile(bd, 0.025), np.nanquantile(bd, 0.975)
        return {"k": k, "비교": f"{METHODS[a]} − {METHODS[b]}", "지표": metric, "차이": round(np.nanmean(v), 3),
                "2.5%": round(lo, 3), "97.5%": round(hi, 3),
                "판단": "우열 판단 증거 부족" if lo <= 0 <= hi else ("앞이 높음" if lo > 0 else "뒤가 높음")}
    P = pd.DataFrame([diff(k, "recal", b, mt) for k in KS for b in ("none", "kshot", "addtrain") for mt in ("mcc", "auc")]
                     + [diff(k, "recal_unc", "recal") for k in KS])
    # post-hoc (added after the first run, not in the plan): retraining with the k shots was the strongest method
    PH = pd.DataFrame([diff(k, "addtrain", b, mt) for k in KS for b in ("kshot", "none") for mt in ("mcc", "auc")])
    main_rows = P[(P.k == 8) & (P.지표 == "mcc") & P.비교.str.contains("보정 없음|새 아민 k번만")]
    success = (main_rows["2.5%"] > 0).all()

    q2 = vec(0, "none", "auc") - vec(0, "none_no_ratio", "auc")
    b2 = boot_mean(q2, idx)
    rate, pred = amine_rate_model(d, y, feat)
    rho = spearmanr(pred, rate).statistic
    brho = [spearmanr(pred.iloc[i], rate.iloc[i]).statistic for i in idx]
    k0 = {mt: ci(vec(0, "none", mt)) for mt in ("mcc", "auc", "brier")}

    verdict = (f"주 판정 (k = 8, 보정이 보정 없음과 새 아민 k번만보다 MCC가 높은가): **{'성공' if success else '미달'}**")
    OUT.write_text(
        "# RAPID step 2 results (generated by rapid/step2.py)\n\n"
        f"{len(d)} experiments, {len(amines)} amines (MCC averaged over all amines; AUC over amines with both outcomes in the "
        f"scored rows). k shots drawn at random {REPS} times per amine and averaged; cells: mean over amines "
        f"[95% amine-bootstrap interval, {N_BOOT} resamples]. Retrain weight for the k shots: {ADD_WEIGHT}. "
        f"Recalibration prior strength: {PRIOR}. Experiments flagged as model-chosen by the original team "
        f"(_raw_model_predicted > 0): {(d._raw_model_predicted > 0).mean():.1%}.\n\n"
        f"## verdict\n{verdict}\n\n{md(main_rows, index=False)}\n\n"
        f"## k = 0 (no experiments with the new amine)\n- cross-amine LightGBM: MCC {k0['mcc']}, AUC {k0['auc']}, Brier {k0['brier']}\n"
        f"- Q2, adding molar ratios (AUC difference): {np.nanmean(q2):+.3f} [{np.nanquantile(b2, 0.025):+.3f}, {np.nanquantile(b2, 0.975):+.3f}]\n"
        f"- Q1, amine descriptors -> amine success rate (Spearman over {len(amines)} amines): {rho:.3f} "
        f"[{np.nanquantile(brho, 0.025):.3f}, {np.nanquantile(brho, 0.975):.3f}]\n\n"
        f"## methods by k\n{md(curve, index=False)}\n\n## paired comparisons (planned)\n{md(P, index=False)}\n\n"
        "## post-hoc comparisons (added after the first run; exploratory, not part of the verdict)\n"
        f"{md(PH, index=False)}\n", encoding="utf-8")
    print(verdict)
    print(md(curve[curve.k == 8], index=False))
    print(md(main_rows, index=False))


if __name__ == "__main__":
    main()
