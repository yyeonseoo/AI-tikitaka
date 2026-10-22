"""Reaction yield, plan 2 (reaction/README.md, "2차 계획"): knowledge features, leave-one-group-out tuning,
Suzuki confirmation. Model families are fixed in advance; inner cross-validation only picks each model's settings.
Outer: 5 group folds (same seed and fold function as model.py). Inner: leave-one-group-out over the training groups
(5-fold by row for the random split). Reagent SMILES for Suzuki: reaction/reagents_sm.csv; knowledge: reaction/knowledge.csv.
Usage: python reaction/model2.py [--mlp]   (-> reaction/results_model2.md, reaction/figures/model2_*.png)
Each finished outer fold is saved to reaction/cache/model2/; rerunning skips finished folds, so the run can be stopped anytime."""
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt, folds, onehot, per_group_mae, ridge_pairs, target_means  # noqa: E402
from prepare import BH_SLOTS, SM_SLOTS, descriptors, load_bh, load_sm  # noqa: E402

OUT, FIG, CACHE = ROOT / "results_model2.md", ROOT / "figures", ROOT / "cache" / "model2"
SEED, N_CFG = 0, 30
WITH_MLP = "--mlp" in sys.argv  # the reference neural network runs only when asked (slowest model)
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- knowledge features (values from reaction/knowledge.csv) ----------------
LG_RANK = {"Cl": 1, "Br": 2, "OTf": 3, "I": 4}
BDE = {"Cl": 97.6, "Br": 84.0, "I": 67.0}
HAMMETT = {"*c1ccc(C(F)(F)F)cc1": 0.54, "*c1ccc(CC)cc1": -0.15, "*c1ccc(OC)cc1": -0.27}
PYRIDINE = {"*c1ccccn1": (1, 1), "*c1cccnc1": (1, 0)}  # (is pyridine, nitrogen next to the reactive carbon)
SM_LG = {"1a, 6-Cl-Q": "Cl", "1b, 6-Br-Q": "Br", "1c, 6-OTf-Q": "OTf", "1d, 6-I-Q": "I"}  # 1e-1g react with indazole bromide
BORON = {"2a, Boronic Acid": "BOH2", "2b, Boronic Ester": "BPin", "2c, Trifluoroborate": "BF3K",
         "1e, 6-BOH2-Q": "BOH2", "1f, 6-BPin-Q": "BPin", "1g, 6-BF3K-Q": "BF3K"}
LIGAND = {  # denticity, biaryl, P-substituent, Tolman cone angle
    "P(tBu)3": (1, 0, "알킬", 182), "P(Ph)3": (1, 0, "아릴", 145), "AmPhos": (1, 0, "혼합", np.nan), "P(Cy)3": (1, 0, "알킬", 170),
    "P(o-Tol)3": (1, 0, "아릴", 194), "CataCXium A": (1, 0, "알킬", np.nan), "SPhos": (1, 1, "알킬", np.nan),
    "dtbpf": (2, 0, "알킬", np.nan), "XPhos": (1, 1, "알킬", np.nan), "dppf": (2, 0, "아릴", np.nan),
    "Xantphos": (2, 0, "아릴", np.nan), "없음": (0, 0, "없음", np.nan)}
BASE_PKA = {"NaOH": 15.7, "KOH": 15.7, "LiOtBu": 17.0, "K3PO4": 12.3, "Et3N": 10.75, "NaHCO3": 6.35, "CsF": 3.2, "없음": np.nan}
SOLVENT_EPS = {"MeCN": 37.5, "DMF": 36.7, "MeOH": 32.7, "THF": 7.6, "THF_V2": 7.6, "MeOH/H2O_V2 9:1": 36.0}


def knowledge_bh(d):
    lg = d.halogen
    K = pd.DataFrame({"K:lg_rank": lg.map(LG_RANK), "K:bde": lg.map(BDE), "K:hammett": d.arene.map(HAMMETT),
                      "K:is_pyridine": d.arene.map(lambda a: PYRIDINE.get(a, (0, 0))[0]),
                      "K:n_adjacent": d.arene.map(lambda a: PYRIDINE.get(a, (0, 0))[1])}, index=d.index)
    return K.astype(float)


def knowledge_sm(d):
    lg = d.electrophile.map(SM_LG).fillna("Br")  # role-swapped reactions: the reactive halide is the indazole bromide
    lig = d.ligand.map(LIGAND)
    K = pd.DataFrame({"K:lg_rank": lg.map(LG_RANK), "K:bde": lg.map(BDE), "K:role_swapped": (~d.electrophile.isin(SM_LG)).astype(int),
                      "K:denticity": lig.str[0], "K:biaryl": lig.str[1], "K:cone": lig.str[3],
                      "K:base_pka": d.base.map(BASE_PKA), "K:base_none": (d.base == "없음").astype(int),
                      "K:solvent_eps": d.solvent.map(SOLVENT_EPS), "K:solvent_v2": d.solvent.str.contains("V2").astype(int)}, index=d.index)
    K = K.join(pd.get_dummies(d.electrophile.map(BORON).fillna(d.nucleophile.map(BORON)), prefix="K:boron", dtype=float))
    K = K.join(pd.get_dummies(lig.str[2], prefix="K:p_sub", dtype=float))
    return K.astype(float)


# ---------------- descriptor features ----------------
def desc_bh(d):
    return pd.concat([descriptors(d[s]).loc[d[s]].set_axis(d.index).add_prefix(f"{s}:") for s in BH_SLOTS], axis=1)


def desc_sm(d):
    smi = pd.read_csv(ROOT / "reagents_sm.csv", keep_default_na=False)
    parts = []
    for s in SM_SLOTS:
        m = smi[smi.slot == s].set_index("name").smiles
        col = d[s].map(m)
        D = descriptors(col[col != ""])
        X = D.reindex(col.to_numpy()).set_axis(d.index).fillna(0.0).add_prefix(f"{s}:")
        X[f"{s}:none"] = (col == "").astype(float).to_numpy()
        parts.append(X)
    return pd.concat(parts, axis=1)


# ---------------- models ----------------
def lgbm_space(rng):
    return {"num_leaves": int(rng.choice([7, 15, 31, 63])), "n_estimators": int(rng.choice([200, 400, 800])),
            "learning_rate": float(rng.choice([0.02, 0.05, 0.1])), "min_child_samples": int(rng.choice([5, 10, 20, 40])),
            "colsample_bytree": float(rng.choice([0.5, 0.8, 1.0])), "reg_lambda": float(rng.choice([0, 1, 10]))}


def mlp_space(rng):
    sizes = [(64,), (128,), (128, 64), (256, 128)]
    return {"hidden_layer_sizes": sizes[int(rng.integers(len(sizes)))], "alpha": float(rng.choice([1e-4, 1e-3, 1e-2, 1e-1])),
            "learning_rate_init": float(rng.choice([1e-3, 3e-3]))}


def make_model(kind, p, mono=None):
    if kind == "lgbm":
        extra = {"monotone_constraints": mono, "monotone_constraints_method": "advanced"} if mono is not None else {}
        return LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1, **p, **extra)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPRegressor(max_iter=500, early_stopping=True, random_state=SEED, **p))


def fit_predict(kind, p, Xtr, ytr, Xap, mono=None):
    return np.clip(make_model(kind, p, mono).fit(Xtr, ytr).predict(Xap), 0, 100)


def inner_mae(kind, p, X, y, splits, mono):
    return np.mean([mean_absolute_error(y[b], fit_predict(kind, p, X[a], y[a], X[b], mono)) for a, b in splits])


# ---------------- one split ----------------
def run_split(name, d, slots, group, D, K):
    rng = np.random.default_rng(SEED)
    cfg_l = [lgbm_space(np.random.default_rng([SEED, 1, i])) for i in range(N_CFG)]
    cfg_m = [mlp_space(np.random.default_rng([SEED, 2, i])) for i in range(N_CFG)]
    F1, F1K = D, D.join(K)
    mono = [1 if c == "K:lg_rank" else 0 for c in F1K.columns]
    models = {"LGBM·F1 (주)": ("lgbm", F1, cfg_l, None), "LGBM·F1+K": ("lgbm", F1K, cfg_l, None),
              "LGBM·F1+K+단조": ("lgbm", F1K, cfg_l, mono)}
    if WITH_MLP:
        models["MLP·F1+K (참고)"] = ("mlp", F1K, cfg_m, None)
    y_all = d.y.to_numpy(float)
    pred, chosen = {}, []
    for f, tr, va, te in folds(d, group, rng):
        # checkpoint per outer fold, updated after every model, so a rerun computes only what is missing
        ck = CACHE / f"{name.split(' (')[0].replace(' ', '_')}_fold{f}.pkl"
        saved, row = pickle.loads(ck.read_bytes()) if ck.exists() else ({}, {"split": name, "fold": f})

        def save():
            CACHE.mkdir(parents=True, exist_ok=True)
            ck.write_bytes(pickle.dumps((saved, row)))
        trn = np.concatenate([tr, va])  # outer training = train + validation groups of model.py
        ytr = y_all[trn]
        if group:
            inner = list(LeaveOneGroupOut().split(trn, groups=d[group].to_numpy()[trn]))
        else:
            inner = list(KFold(5, shuffle=True, random_state=SEED).split(trn))
        if "최선 단순 (안쪽 검증 선택)" not in saved:  # simple baselines, chosen by the same inner CV
            simple = {}
            if group:
                # same other conditions; when a whole arene is held out, its substrates are unseen too (halogen pooled)
                keys = [s for s in slots if s != group and not (group == "arene" and s == "aryl_halide")]
                dt = d.iloc[trn]
                cv = np.mean([mean_absolute_error(dt.y.iloc[b], target_means(dt.iloc[a], dt.iloc[b], keys)) for a, b in inner])
                simple["조건 평균"] = (cv, target_means(dt, d.iloc[te], keys))
            XR = ridge_pairs(d, slots).to_numpy(float)
            alphas = {a: np.mean([mean_absolute_error(ytr[b], Ridge(a).fit(XR[trn][a_], ytr[a_]).predict(XR[trn][b]))
                                  for a_, b in inner]) for a in [0.1, 1, 10, 100]}
            a = min(alphas, key=alphas.get)
            simple["선형 + 짝 상호작용"] = (alphas[a], np.clip(Ridge(a).fit(XR[trn], ytr).predict(XR[te]), 0, 100))
            best_s = min(simple, key=lambda k: simple[k][0])
            saved.update({k: v for k, (_, v) in simple.items()})
            saved["최선 단순 (안쪽 검증 선택)"] = simple[best_s][1]
            row["최선 단순"] = best_s
            save()
        for mname, (kind, X, cfgs, mo) in models.items():
            if mname in saved:
                continue
            Xa = X.to_numpy(float)
            scores = Parallel(n_jobs=-1)(delayed(inner_mae)(kind, p, Xa[trn], ytr, inner, mo) for p in cfgs)
            p = cfgs[int(np.argmin(scores))]
            saved[mname] = fit_predict(kind, p, Xa[trn], ytr, Xa[te], mo)
            row[mname] = round(min(scores), 2)
            save()
        keep = [k for k in saved if k in models or "(" not in k or "안쪽 검증" in k]  # drop MLP results when it is switched off
        for k in keep:
            pred.setdefault(k, np.full(len(d), np.nan))[te] = saved[k]
        chosen.append({k: v for k, v in row.items() if k in models or k in ("split", "fold", "최선 단순")})
        print(f"  {name} fold {f}: done", flush=True)
    return pred, pd.DataFrame(chosen)


def within_group(d, p, group):
    """Per test group: Spearman between predicted and actual yield across its conditions, and top-1 actual yield."""
    rows = []
    for g, idx in d.groupby(group).indices.items():
        y, q = d.y.to_numpy()[idx], p[idx]
        rows.append({"group": g, "rho": spearmanr(q, y).statistic, "top1": y[np.argmax(q)], "best": y.max(), "mean": y.mean()})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    bh, sm = load_bh(), load_sm()
    Dbh, Kbh = desc_bh(bh), knowledge_bh(bh)
    Dsm, Ksm = desc_sm(sm), knowledge_sm(sm)
    splits = [("SM 처음 보는 리간드 (주 확인)", sm, SM_SLOTS, "ligand", Dsm, Ksm),
              ("SM 처음 보는 반응 자리 (보조)", sm, SM_SLOTS, "electrophile", Dsm, Ksm),
              ("BH 처음 보는 기질 (재분석)", bh, BH_SLOTS, "aryl_halide", Dbh, Kbh),
              ("SM 무작위 (참고)", sm, SM_SLOTS, None, Dsm, Ksm),
              ("BH 무작위 (참고)", bh, BH_SLOTS, None, Dbh, Kbh)]
    text, chosen_all, res = [], [], {}
    for name, d, slots, group, D, K in splits:
        pred, chosen = run_split(name, d, slots, group, D, K)
        res[name] = (d, group, pred)
        chosen_all.append(chosen)
        rows = []
        for k, p in pred.items():
            r = {"방법": k, "MAE (%p)": fmt(boot_ci(per_group_mae(d, p, group))), "R²": round(r2_score(d.y, p), 3)}
            if group:
                W = within_group(d, p, group)
                r["그룹 안 순위 상관"] = round(W.rho.mean(), 3)
                r["1순위 추천 실제 수율"] = f"{W.top1.mean():.1f}% (최고 {W.best.mean():.1f}%, 평균 {W['mean'].mean():.1f}%)"
            rows.append(r)
        text.append(f"## {name}\n그룹: {group or '반응'}; 구간은 {'그룹' if group else '반응'} 단위 부트스트랩\n\n{md(pd.DataFrame(rows), index=False)}\n")
        print(name, f"{(time.time() - t0) / 60:.1f} min", flush=True)

    def judge(t):
        return "충족" if t[2] < 0 else ("우열 판단 증거 부족" if t[1] <= 0 <= t[2] else "반대 방향")

    def cmp(name, a, b):
        d, g, P = res[name]
        return boot_ci(per_group_mae(d, P[a], g) - per_group_mae(d, P[b], g))
    V = [("확인 1", "SM 처음 보는 리간드 (주 확인)", "LGBM·F1 (주)", "최선 단순 (안쪽 검증 선택)"),
         ("확인 2", "SM 처음 보는 리간드 (주 확인)", "LGBM·F1+K", "LGBM·F1 (주)"),
         ("보조", "SM 처음 보는 반응 자리 (보조)", "LGBM·F1 (주)", "최선 단순 (안쪽 검증 선택)"),
         ("보조", "SM 처음 보는 반응 자리 (보조)", "LGBM·F1+K", "LGBM·F1 (주)"),
         ("재분석", "BH 처음 보는 기질 (재분석)", "LGBM·F1 (주)", "최선 단순 (안쪽 검증 선택)"),
         ("재분석", "BH 처음 보는 기질 (재분석)", "LGBM·F1+K", "LGBM·F1 (주)"),
         ("참고", "SM 처음 보는 리간드 (주 확인)", "LGBM·F1+K+단조", "LGBM·F1+K"),
         ("참고", "SM 처음 보는 반응 자리 (보조)", "LGBM·F1+K+단조", "LGBM·F1+K")]
    vt = "## 판정\n| 구분 | 상황 | 비교 | MAE 차이 (%p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|---|---|\n"
    for lab, sp, a, b in V:
        t = cmp(sp, a, b)
        vt += f"| {lab} | {sp} | {a} − {b} | {fmt(t, True)} | **{judge(t)}** |\n"

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, sp in zip(ax, [s[0] for s in splits[:3]]):
        d, g, P = res[sp]
        keys = [k for k in ["조건 평균", "선형 + 짝 상호작용", "LGBM·F1 (주)", "LGBM·F1+K", "LGBM·F1+K+단조", "MLP·F1+K (참고)"] if k in P]
        v = [boot_ci(per_group_mae(d, P[k], g)) for k in keys]
        m = np.array([x[0] for x in v])
        a.barh(keys[::-1], m[::-1], xerr=[(m - [x[1] for x in v])[::-1], ([x[2] for x in v] - m)[::-1]],
               color=[ORANGE if "MLP" in k else BLUE if "LGBM" in k else GRAY for k in keys][::-1], height=0.6, capsize=3)
        a.set(title=sp, xlabel="MAE (%p, 작을수록 좋음)")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("2차: 처음 보는 시약·기질의 수율 예측 오차 (회색 = 단순, 파랑 = LightGBM, 주황 = 신경망; 선 = 그룹 단위 95% 구간)")
    fig.tight_layout()
    fig.savefig(FIG / "model2_summary.png", dpi=110)
    plt.close(fig)

    C = pd.concat(chosen_all)
    OUT.write_text("# 반응 수율 2차 결과 (generated by reaction/model2.py)\n\n"
                   f"모델 종류는 미리 고정했고, 안쪽 검증(학습 그룹 하나 빼기)으로 모델마다 설정 후보 {N_CFG}개 중 하나를 골랐습니다. "
                   f"F1 = RDKit 분자 특성, K = 화학 지식 특성 (knowledge.csv). 실행 {(time.time() - t0) / 60:.0f}분.\n\n"
                   + vt + "\n" + "\n".join(text)
                   + f"\n## 묶음마다 안쪽 검증 MAE와 고른 단순 기준선\n{md(C, index=False)}\n\n그림: [figures/model2_summary.png](figures/model2_summary.png)\n",
                   encoding="utf-8")
    print(vt)


if __name__ == "__main__":
    main()
