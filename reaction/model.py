"""Reaction yield: steps 3-4 of the plan in reaction/README.md (baselines, tuning, test evaluation, verdicts).
Pipeline per split and outer fold: groups -> 5 outer folds (test 20%); remaining groups -> train 75% / validation 25%
(60/20/20 overall). Features are built from the training rows only (target encodings are out-of-group inside training).
Every model is tuned on validation MAE (random search), refit on train + validation, then scored once on test.
Usage: python reaction/model.py   (-> reaction/results_model.md, reaction/figures/model_*.png)"""
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from prepare import BH_SLOTS, SM_SLOTS, descriptors, load_bh, load_sm  # noqa: E402

OUT, FIG = ROOT / "results_model.md", ROOT / "figures"
SEED, N_CFG, N_BOOT = 0, 20, 1000
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ---------------- splits ----------------
def folds(d, group, rng):
    """Outer 5 folds by group (or by row when group is None); inside, 25% of the remaining groups for validation."""
    keys = d.index.to_numpy() if group is None else d[group].to_numpy()
    uniq = rng.permutation(pd.unique(keys))
    outer = {g: i % 5 for i, g in enumerate(uniq)}
    for f in range(5):
        test_g = [g for g in uniq if outer[g] == f]
        rest = rng.permutation([g for g in uniq if outer[g] != f])
        n_val = int(round(len(rest) * 0.25))
        val_g, tr_g = rest[:n_val], rest[n_val:]
        yield f, np.flatnonzero(np.isin(keys, tr_g)), np.flatnonzero(np.isin(keys, val_g)), np.flatnonzero(np.isin(keys, test_g))


# ---------------- features ----------------
def onehot(d, slots):
    return pd.get_dummies(d[slots].astype(str), dtype=float)


def desc_block(d, slots):
    parts = []
    for s in slots:
        D = descriptors(d[s])
        parts.append(D.loc[d[s]].set_axis(d.index).add_prefix(f"{s}:"))
    return pd.concat(parts, axis=1)


def target_means(tr, ap, keys, oof_groups=None):
    """Mean training yield for rows sharing `keys`; for training rows, computed without their own group (no leakage)."""
    gm = tr.y.mean()
    if oof_groups is None:
        return ap[keys].merge(tr.groupby(keys).y.mean().rename("m"), how="left", left_on=keys, right_index=True).m.fillna(gm).to_numpy()
    out = np.empty(len(ap))
    for g in pd.unique(oof_groups):
        rest, me = tr[oof_groups != g], (oof_groups == g)
        out[me] = ap.loc[me, keys].merge(rest.groupby(keys).y.mean().rename("m"), how="left", left_on=keys, right_index=True).m.fillna(rest.y.mean()).to_numpy()
    return out


def f2_block(d, tr_idx, ap_idx, group, slots, oof):
    """Engineered features (BH): halogen rank, arene one-hot, condition/reagent/pair mean yields from training rows."""
    tr, ap = d.iloc[tr_idx], d.iloc[ap_idx]
    keysets = [["ligand", "base", "additive"], ["ligand"], ["base"], ["additive"], ["arene"], ["halogen"],
               ["halogen", "ligand", "base", "additive"]] + [list(p) for p in combinations(["ligand", "base", "additive", "arene", "halogen"], 2)]
    og = None
    if oof:
        og = (tr[group].to_numpy() if group else np.arange(len(tr)) % 5)
    cols = {"te:" + "×".join(k): target_means(tr, ap, k, og) for k in keysets}
    F = pd.DataFrame(cols, index=ap.index)
    F["halogen_rank"] = ap.halogen_rank.to_numpy()
    return F.join(pd.get_dummies(ap.arene, prefix="arene", dtype=float).reindex(columns=[f"arene_{a}" for a in sorted(d.arene.unique())], fill_value=0.0))


def build(d, fs, slots, group, tr_idx, ap_idx, is_train):
    if fs == "F0":
        return onehot(d, slots).iloc[ap_idx]
    X = desc_block(d, slots).iloc[ap_idx]
    if fs == "F2":
        X = X.join(f2_block(d, tr_idx, ap_idx, group, slots, oof=is_train))
    return X


# ---------------- models ----------------
def model_space(rng):
    def lgbm():
        return {"num_leaves": int(rng.choice([7, 15, 31, 63])), "n_estimators": int(rng.choice([200, 400, 800])),
                "learning_rate": float(rng.choice([0.02, 0.05, 0.1])), "min_child_samples": int(rng.choice([5, 10, 20, 40])),
                "colsample_bytree": float(rng.choice([0.5, 0.8, 1.0])), "reg_lambda": float(rng.choice([0, 1, 10]))}

    def rf():
        return {"n_estimators": 300, "max_features": float(rng.choice([0.2, 0.4, 0.7, 1.0])),
                "min_samples_leaf": int(rng.choice([1, 2, 5, 10])), "max_depth": [None, 10, 20][int(rng.integers(3))]}

    def mlp():
        sizes = [(64,), (128,), (128, 64), (256, 128)]
        return {"hidden_layer_sizes": sizes[int(rng.integers(len(sizes)))],
                "alpha": float(rng.choice([1e-4, 1e-3, 1e-2, 1e-1])), "learning_rate_init": float(rng.choice([1e-3, 3e-3]))}
    return {"LightGBM": (lgbm, lambda p: LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1, **p)),
            "랜덤 포레스트": (rf, lambda p: RandomForestRegressor(random_state=SEED, n_jobs=1, **p)),
            "신경망 (MLP)": (mlp, lambda p: make_pipeline(StandardScaler(), MLPRegressor(max_iter=500, early_stopping=True, random_state=SEED, **p)))}


def ridge_pairs(d, slots):
    X = onehot(d, slots)
    for a, b in combinations(slots, 2):
        X = X.join(pd.get_dummies(d[a].astype(str) + "×" + d[b].astype(str), prefix=f"{a}×{b}", dtype=float))
    return X


def condition_average(d, slots, group, tr_idx, ap_idx):
    """Simple baseline: mean yield of training rows that share every slot except the held-out one."""
    keys = [s for s in slots if s != group] if group else slots
    return target_means(d.iloc[tr_idx], d.iloc[ap_idx], keys)


def fit_predict(make, p, X_tr, y_tr, X_ap):
    # plain arrays: SMILES-based column names contain characters LightGBM rejects
    return np.clip(make(p).fit(np.asarray(X_tr, float), np.asarray(y_tr, float)).predict(np.asarray(X_ap, float)), 0, 100)


# ---------------- one split ----------------
def run_split(name, d, slots, group, feature_sets):
    rng = np.random.default_rng(SEED)
    space = model_space(np.random.default_rng(SEED + 1))
    configs = {m: [gen() for _ in range(N_CFG)] for m, (gen, _) in space.items()}
    pred = {}  # method -> test predictions (aligned with d)
    chosen_rows = []
    for f, tr, va, te in folds(d, group, rng):
        trva = np.concatenate([tr, va])
        ytr, yva = d.y.to_numpy()[tr], d.y.to_numpy()[va]
        # simple baselines (tuned on validation where they have a setting)
        simple = {}
        if group:
            simple["조건 평균"] = (mean_absolute_error(yva, condition_average(d, slots, group, tr, va)),
                                condition_average(d, slots, group, trva, te))
        for nm, Xall in (("선형", onehot(d, slots)), ("선형 + 짝 상호작용", ridge_pairs(d, slots))):
            best = min([0.1, 1, 10, 100], key=lambda a: mean_absolute_error(yva, Ridge(a).fit(Xall.iloc[tr], ytr).predict(Xall.iloc[va])))
            simple[nm] = (mean_absolute_error(yva, Ridge(best).fit(Xall.iloc[tr], ytr).predict(Xall.iloc[va])),
                          np.clip(Ridge(best).fit(Xall.iloc[trva], d.y.iloc[trva]).predict(Xall.iloc[te]), 0, 100))
        # ML: model x feature set, random search on validation
        ml = {}
        for fs in feature_sets:
            X_tr, X_va = build(d, fs, slots, group, tr, tr, True), build(d, fs, slots, group, tr, va, False)
            X_trva, X_te = build(d, fs, slots, group, trva, trva, True), build(d, fs, slots, group, trva, te, False)
            for m, (_, make) in space.items():
                maes = Parallel(n_jobs=-1)(delayed(lambda p: mean_absolute_error(yva, fit_predict(make, p, X_tr, ytr, X_va)))(p)
                                           for p in configs[m])
                p = configs[m][int(np.argmin(maes))]
                ml[f"{m} · {fs}"] = (min(maes), fit_predict(make, p, X_trva, d.y.iloc[trva], X_te))
        for k, (v, p) in {**simple, **ml}.items():
            pred.setdefault(k, np.full(len(d), np.nan))[te] = p
        best_s = min(simple, key=lambda k: simple[k][0])
        best_m = min(ml, key=lambda k: ml[k][0])
        pred.setdefault("최선 단순 (검증 선택)", np.full(len(d), np.nan))[te] = simple[best_s][1]
        pred.setdefault("최선 머신러닝 (검증 선택)", np.full(len(d), np.nan))[te] = ml[best_m][1]
        chosen_rows.append({"split": name, "fold": f, "test groups": len(pd.unique(d.iloc[te][group])) if group else len(te),
                            "최선 단순": best_s, "최선 머신러닝": best_m})
    return pred, pd.DataFrame(chosen_rows)


# ---------------- evaluation ----------------
def per_group_mae(d, p, group):
    if group is None:
        return pd.Series(np.abs(p - d.y.to_numpy()))
    return pd.Series(np.abs(p - d.y.to_numpy())).groupby(d[group].to_numpy()).mean()


def boot_ci(v, b=N_BOOT):
    v = np.asarray(v, float)
    rng = np.random.default_rng(SEED + 7)
    s = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(b)])
    return v.mean(), np.quantile(s, 0.025), np.quantile(s, 0.975)


def fmt(t, sign=False):
    f = "{:+.2f}" if sign else "{:.2f}"
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


def recommend(d, p, group):
    """Per held-out group (substrate): actual yield of the top-1 prediction, best possible, mean; top-10 hit of actual top 10%."""
    rows = []
    for g, idx in d.groupby(group).indices.items():
        y, q = d.y.to_numpy()[idx], p[idx]
        order = np.argsort(-q, kind="stable")
        top10 = set(np.argsort(-y)[:max(1, int(round(0.1 * len(y))))])
        rows.append({"top1": y[order[0]], "best": y.max(), "mean": y.mean(), "hit10": np.mean([i in top10 for i in order[:10]])})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    bh, sm = load_bh(), load_sm()
    splits = [("BH 처음 보는 기질 (주)", bh, BH_SLOTS, "aryl_halide", ["F0", "F1", "F2"]),
              ("BH 처음 보는 첨가제", bh, BH_SLOTS, "additive", ["F0", "F1", "F2"]),
              ("SM 처음 보는 리간드", sm, SM_SLOTS, "ligand", ["F0"]),
              ("BH 무작위", bh, BH_SLOTS, None, ["F0", "F1", "F2"]),
              ("SM 무작위", sm, SM_SLOTS, None, ["F0"])]
    text, chosen_all, results = [], [], {}
    for name, d, slots, group, fsets in splits:
        pred, chosen = run_split(name, d, slots, group, fsets)
        chosen_all.append(chosen)
        results[name] = (d, group, pred)
        rows = []
        for k, p in pred.items():
            rows.append({"방법": k, "MAE (%p)": fmt(boot_ci(per_group_mae(d, p, group))), "R²": round(r2_score(d.y, p), 3)})
        text.append(f"## {name}\n그룹: {group or '반응 (무작위)'}, MAE 구간은 {'그룹' if group else '반응'} 단위 부트스트랩 {N_BOOT}회\n\n"
                    + md(pd.DataFrame(rows), index=False) + "\n")
        print(name, f"{(time.time() - t0) / 60:.1f} min", flush=True)

    # verdicts
    d, g, P = results["BH 처음 보는 기질 (주)"]
    v_main = boot_ci(per_group_mae(d, P["최선 머신러닝 (검증 선택)"], g) - per_group_mae(d, P["최선 단순 (검증 선택)"], g))
    v_f2 = boot_ci(per_group_mae(d, P["LightGBM · F2"], g) - per_group_mae(d, P["LightGBM · F1"], g))
    d2, g2, P2 = results["BH 처음 보는 첨가제"]
    v_f1 = boot_ci(per_group_mae(d2, P2["LightGBM · F1"], g2) - per_group_mae(d2, P2["LightGBM · F0"], g2))

    def judge(t):
        return "충족" if t[2] < 0 else ("우열 판단 증거 부족" if t[1] <= 0 <= t[2] else "반대 방향")
    verdict = ("## 판정\n| 판정 | 차이 (MAE %p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|\n"
               f"| 주: 처음 보는 기질, 최선 머신러닝 − 최선 단순 | {fmt(v_main, True)} | **{judge(v_main)}** |\n"
               f"| 분자 특성: 처음 보는 첨가제, LightGBM F1 − F0 | {fmt(v_f1, True)} | **{judge(v_f1)}** |\n"
               f"| 합성 특성: 처음 보는 기질, LightGBM F2 − F1 | {fmt(v_f2, True)} | **{judge(v_f2)}** |\n")

    # recommendation (unseen substrate)
    rec_rows = []
    for k in ["조건 평균", "선형 + 짝 상호작용", "LightGBM · F0", "LightGBM · F2", "최선 단순 (검증 선택)", "최선 머신러닝 (검증 선택)"]:
        R = recommend(d, P[k], g)
        rec_rows.append({"방법": k, "1순위 추천의 실제 수율": f"{R.top1.mean():.1f}%", "가능한 최고": f"{R.best.mean():.1f}%",
                         "기질 평균": f"{R['mean'].mean():.1f}%", "상위 10개 중 실제 상위 10%": f"{R.hit10.mean():.2f}"})
    rec = "## 추천 지표 (처음 보는 기질 15종 평균, 무작위면 상위 10% 비율 0.10)\n" + md(pd.DataFrame(rec_rows), index=False) + "\n"

    # figure: per-method MAE on the main split
    keys = [k for k in P if "검증 선택" not in k]
    vals = {k: boot_ci(per_group_mae(d, P[k], g)) for k in keys}
    order = sorted(keys, key=lambda k: vals[k][0])
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(order))[::-1]
    m = np.array([vals[k][0] for k in order])
    lo = np.array([vals[k][1] for k in order])
    hi = np.array([vals[k][2] for k in order])
    ax.barh(y, m, xerr=[m - lo, hi - m], color=[GRAY if "·" not in k else BLUE for k in order], height=0.6, capsize=3)
    ax.set_yticks(y, order)
    ax.set(xlabel="MAE (%p, 작을수록 좋음; 선 = 기질 단위 95% 구간)", title="처음 보는 기질(BH)의 수율 예측 오차 (회색 = 단순 기준선, 파랑 = 머신러닝)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "model_substrate.png", dpi=110)
    plt.close(fig)

    C = pd.concat(chosen_all)
    OUT.write_text("# 반응 수율 모델 결과 (generated by reaction/model.py)\n\n"
                   f"모델마다 설정 후보 {N_CFG}개를 검증 MAE로 골랐고, 학습+검증으로 다시 학습해 시험에서 한 번 채점했습니다. "
                   f"F0 = 시약 이름(원-핫), F1 = RDKit 분자 특성, F2 = F1 + 합성 특성. 실행 {(time.time() - t0) / 60:.0f}분.\n\n"
                   + verdict + "\n" + rec + "\n" + "\n".join(text)
                   + f"\n## 묶음마다 검증으로 고른 방법\n{md(C, index=False)}\n\n그림: [figures/model_substrate.png](figures/model_substrate.png)\n",
                   encoding="utf-8")
    print(verdict)
    print(rec)


if __name__ == "__main__":
    main()
