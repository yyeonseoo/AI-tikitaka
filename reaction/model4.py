"""Reaction yield, plan 4 (reaction/README.md, "4차 계획"): condition recommendation on Buchwald-Hartwig.
A context is (substrate, additive); the conditions to choose from are ligand x base (11-12 per context).
Primary: regret@3 on unseen substrates, LightGBM·F1 - chemist-rule ridge, bootstrap over the 15 substrates.
LightGBM·F1 and condition-average predictions are read from the plan 2/3 checkpoints; the chemist-rule ridge,
the halogen-wise condition average and LightGBM on random fingerprints (3 seeds) are new.
Usage: python reaction/model4.py   (-> reaction/results_model4.md, reaction/figures/model4_topk.png)"""
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt, folds, per_group_mae, target_means  # noqa: E402
from model2 import CACHE, N_CFG, SEED, desc_bh, fit_predict, inner_mae, lgbm_space  # noqa: E402
from prepare import BH_SLOTS, load_bh, load_sm  # noqa: E402

OUT, FIG = ROOT / "results_model4.md", ROOT / "figures"
CACHE4 = ROOT / "cache" / "model4"
SPLITS = {"aryl_halide": ("처음 보는 출발물질", "BH_처음_보는_기질"), "arene": ("처음 보는 고리", "BH_처음_보는_고리")}
COND = ["ligand", "base", "additive"]
KS = [1, 2, 3, 5]
ALPHAS = [0.1, 1, 10, 100, 1000]
FP_SEEDS = [1, 2, 3]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ---------------- recommenders ----------------
def rule_features(d):
    """Chemist-rule one-hot: main effects, condition pairs, halogen x condition, arene x condition (no halogen x arene)."""
    cols = {s: d[s].astype(str) for s in COND + ["halogen", "arene"]}
    parts = [pd.get_dummies(pd.DataFrame(cols), dtype=float)]
    pairs = [("ligand", "base"), ("ligand", "additive"), ("base", "additive")] + \
            [(a, c) for a in ("halogen", "arene") for c in COND]
    for a, b in pairs:
        parts.append(pd.get_dummies(cols[a] + "×" + cols[b], prefix=f"{a}×{b}", dtype=float))
    return pd.concat(parts, axis=1).to_numpy(float)


def chemist_rule(X, y, trn, te, inner):
    cv = {a: np.mean([mean_absolute_error(y[trn][b], Ridge(a).fit(X[trn][i], y[trn][i]).predict(X[trn][b])) for i, b in inner])
          for a in ALPHAS}
    a = min(cv, key=cv.get)
    return np.clip(Ridge(a).fit(X[trn], y[trn]).predict(X[te]), 0, 100), a


def halogen_condition_mean(tr, ap):
    keys = ["halogen"] + COND
    m = ap[keys].merge(tr.groupby(keys).y.mean().rename("m"), how="left", left_on=keys, right_index=True).m.to_numpy()
    return np.where(np.isnan(m), target_means(tr, ap, COND), m)


def random_fingerprints(d, D, seed):
    """Each reagent gets a fixed random 0/1 vector as long as its slot's descriptor block."""
    rng = np.random.default_rng([seed, 4])
    parts = []
    for s in BH_SLOTS:
        n = sum(c.startswith(f"{s}:") for c in D.columns)
        code = {r: rng.integers(0, 2, n) for r in pd.unique(d[s])}
        parts.append(np.stack([code[r] for r in d[s]]))
    return np.hstack(parts).astype(float)


def lgbm_tuned(X, y, trn, te, inner):
    cfgs = [lgbm_space(np.random.default_rng([SEED, 1, i])) for i in range(N_CFG)]  # same candidates as plan 2
    scores = Parallel(n_jobs=-1)(delayed(inner_mae)("lgbm", p, X[trn], y[trn], inner, None) for p in cfgs)
    return fit_predict("lgbm", cfgs[int(np.argmin(scores))], X[trn], y[trn], X[te])


def predictions(d, group):
    """Out-of-fold predictions of every recommender on the same outer folds as plans 2/3 (checkpointed per fold)."""
    name, cache_name = SPLITS[group]
    y = d.y.to_numpy(float)
    XR, D = rule_features(d), desc_bh(d)
    XF = {s: random_fingerprints(d, D, s) for s in FP_SEEDS}
    P, alphas = {}, []
    for f, tr, va, te in folds(d, group, np.random.default_rng(SEED)):
        old, _ = pickle.loads((CACHE / f"{cache_name}_fold{f}.pkl").read_bytes())
        ck = CACHE4 / f"{cache_name}_fold{f}.pkl"
        new = pickle.loads(ck.read_bytes()) if ck.exists() else {}
        trn = np.concatenate([tr, va])
        inner = list(LeaveOneGroupOut().split(trn, groups=d[group].to_numpy()[trn]))
        if "화학자 규칙 모델" not in new:
            new["화학자 규칙 모델"], new["alpha"] = chemist_rule(XR, y, trn, te, inner)
            new["할로젠별 조건 평균"] = halogen_condition_mean(d.iloc[trn], d.iloc[te])
        for s in FP_SEEDS:
            k = f"LightGBM + 무작위 지문 (시드 {s})"
            if k not in new:
                new[k] = lgbm_tuned(XF[s], y, trn, te, inner)
                CACHE4.mkdir(parents=True, exist_ok=True)
                ck.write_bytes(pickle.dumps(new))
        CACHE4.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(pickle.dumps(new))
        cols = {"조건 평균": old["조건 평균"], "할로젠별 조건 평균": new["할로젠별 조건 평균"],
                "화학자 규칙 모델": new["화학자 규칙 모델"], "LightGBM + 분자 특성": old["LGBM·F1 (주)"]}
        cols.update({f"LightGBM + 무작위 지문 (시드 {s})": new[f"LightGBM + 무작위 지문 (시드 {s})"] for s in FP_SEEDS})
        for k, v in cols.items():
            P.setdefault(k, np.full(len(d), np.nan))[te] = v
        alphas.append(new["alpha"])
        print(f"  {name} fold {f}: done", flush=True)
    return P, alphas


# ---------------- metrics per context ----------------
def model_metrics(y, q, rng, kmax):
    order = np.lexsort((rng.random(len(y)), -q))  # by prediction, ties in random order
    ys, best = y[order], y.max()
    r = {f"regret@{k}": best - ys[:k].max() for k in KS}
    r.update({f"hit@{k}": float((ys[:k] == best).any()) for k in range(1, kmax + 1)})
    r["near"] = float(np.argmax(ys >= best - 5) + 1)
    r["rho"] = spearmanr(q, y).statistic if np.ptp(q) > 0 else np.nan
    r["mae"] = np.abs(q - y).mean()
    r["ties"] = len(q) - len(np.unique(q))
    return r


def random_metrics(y, kmax):
    """Exact expectations for k conditions drawn at random from the context."""
    n, ys, best = len(y), np.sort(y), y.max()
    m, m5 = int((y == best).sum()), int((y >= best - 5).sum())
    r = {}
    for k in KS:
        e_max = sum(comb(i - 1, k - 1) * ys[i - 1] for i in range(k, n + 1)) / comb(n, k)
        r[f"regret@{k}"] = best - e_max
    r.update({f"hit@{k}": 1 - comb(n - m, min(k, n)) / comb(n, min(k, n)) for k in range(1, kmax + 1)})
    r["near"] = (n + 1) / (m5 + 1)
    r.update(rho=np.nan, mae=np.nan, ties=0)
    return r


def context_table(d, P, kmax=12):
    rng = np.random.default_rng(SEED)
    rows = []
    for (sub, add), idx in d.groupby(["aryl_halide", "additive"]).indices.items():
        y = d.y.to_numpy(float)[idx]
        base = {"substrate": sub, "additive": add, "n": len(idx)}
        rows.append({**base, "method": "무작위", **random_metrics(y, kmax)})
        for k, p in P.items():
            rows.append({**base, "method": k, **model_metrics(y, p[idx], rng, kmax)})
    T = pd.DataFrame(rows)
    fp = T[T.method.str.startswith("LightGBM + 무작위 지문")]
    avg = fp.groupby(["substrate", "additive", "n"], as_index=False).mean(numeric_only=True)
    return pd.concat([T, avg.assign(method="LightGBM + 무작위 지문 (3개 평균)")], ignore_index=True)


def per_substrate(T, method, col):
    return T[T.method == method].groupby("substrate")[col].mean()


def diff_ci(T, a, b, col):
    return boot_ci(per_substrate(T, a, col) - per_substrate(T, b, col))


def two_stage_ci(T, a, b, col, n_boot=1000):
    """Resample substrates, then contexts (additives) within each drawn substrate."""
    da = T[T.method == a].set_index(["substrate", "additive"])[col]
    db = T[T.method == b].set_index(["substrate", "additive"])[col]
    diff = (da - db).groupby(level=0).apply(lambda s: s.to_numpy())
    rng = np.random.default_rng(SEED + 7)
    subs = diff.index.to_numpy()
    stats = []
    for _ in range(n_boot):
        draw = rng.choice(subs, len(subs))
        stats.append(np.mean([rng.choice(diff[s], len(diff[s])).mean() for s in draw]))
    return np.mean([v.mean() for v in diff]), np.quantile(stats, 0.025), np.quantile(stats, 0.975)


def judge(t):
    return "충족" if t[2] < 0 else ("우열 판단 증거 부족" if t[1] <= 0 <= t[2] else "반대 방향")


ORDER = ["무작위", "조건 평균", "할로젠별 조건 평균", "화학자 규칙 모델", "LightGBM + 분자 특성",
         "LightGBM + 무작위 지문 (3개 평균)"]


def summary(T):
    rows = []
    for m in ORDER + [f"LightGBM + 무작위 지문 (시드 {s})" for s in FP_SEEDS]:
        t = T[T.method == m]
        r = {"방법": m, "regret@3 (%p)": fmt(boot_ci(per_substrate(T, m, "regret@3")))}
        r.update({f"regret@{k}": round(t[f"regret@{k}"].mean(), 2) for k in KS if k != 3})
        r.update({f"적중@{k}": f"{t[f'hit@{k}'].mean():.0%}" for k in KS})
        r["5%p 이내까지 실험 수"] = round(t.near.mean(), 2)
        r["순위 상관"] = "–" if m == "무작위" else round(t.rho.mean(), 3)
        r["MAE"] = "–" if m == "무작위" else round(t.mae.mean(), 2)
        rows.append(r)
    return pd.DataFrame(rows)


def sm_ligand_without_none():
    sm = load_sm()
    P = {}
    for f, tr, va, te in folds(sm, "ligand", np.random.default_rng(SEED)):
        saved, _ = pickle.loads((CACHE / f"SM_처음_보는_리간드_fold{f}.pkl").read_bytes())
        for k in ("최선 단순 (안쪽 검증 선택)", "LGBM·F1 (주)"):
            P.setdefault(k, np.full(len(sm), np.nan))[te] = saved[k]
    rows = []
    for label, keep in (("전체 12종", sm.ligand != ""), ("리간드 없음 제외 11종", sm.ligand != "없음")):
        s = sm[keep]
        g = {k: per_group_mae(s, v[keep.to_numpy()], "ligand") for k, v in P.items()}
        rows.append({"평가 범위": label, "최선 단순": fmt(boot_ci(g["최선 단순 (안쪽 검증 선택)"])),
                     "LightGBM + 분자 특성": fmt(boot_ci(g["LGBM·F1 (주)"])),
                     "차이 (LightGBM − 단순)": fmt(boot_ci(g["LGBM·F1 (주)"] - g["최선 단순 (안쪽 검증 선택)"]), True)})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    d = load_bh()
    tables, alphas = {}, {}
    for group in SPLITS:
        P, alphas[group] = predictions(d, group)
        tables[group] = context_table(d, P)
    Ts, Ta = tables["aryl_halide"], tables["arene"]
    rule, lgbm = "화학자 규칙 모델", "LightGBM + 분자 특성"

    V = [("**주**", "처음 보는 출발물질", Ts, lgbm, rule)]
    for a, b in ((rule, "무작위"), (lgbm, "무작위"), (lgbm, "조건 평균"), (rule, "할로젠별 조건 평균"),
                 ("할로젠별 조건 평균", "조건 평균"), ("LightGBM + 무작위 지문 (3개 평균)", lgbm)):
        V.append(("2차", "처음 보는 출발물질", Ts, a, b))
    V += [("2차", "처음 보는 고리", Ta, lgbm, rule), ("2차", "처음 보는 고리", Ta, lgbm, "무작위"),
          ("2차", "처음 보는 고리", Ta, rule, "무작위"), ("2차", "처음 보는 고리", Ta, "LightGBM + 무작위 지문 (3개 평균)", lgbm)]
    vt = "| 구분 | 분할 | 비교 (regret@3) | 차이 (%p, 음수 = 앞이 좋음) | 결과 |\n|---|---|---|---|---|\n"
    for lab, name, T, a, b in V:
        t = diff_ci(T, a, b, "regret@3")
        res = f"**{judge(t)}**" if lab == "**주**" else judge(t)
        vt += f"| {lab} | {name} | {a} − {b} | {fmt(t, True)} | {res} |\n"

    full = Ts[Ts.n == 12]
    sens = pd.DataFrame([
        {"분석": "기본 (출발물질 단위 부트스트랩)", "LightGBM − 화학자 규칙, regret@3": fmt(diff_ci(Ts, lgbm, rule, "regret@3"), True)},
        {"분석": "2단계 부트스트랩 (출발물질 → 첨가제 상황)", "LightGBM − 화학자 규칙, regret@3": fmt(two_stage_ci(Ts, lgbm, rule, "regret@3"), True)},
        {"분석": "조건 11개 상황 5개 제외", "LightGBM − 화학자 규칙, regret@3": fmt(diff_ci(full, lgbm, rule, "regret@3"), True)}])
    fp = pd.DataFrame([{"시드": s, **{name: round(per_substrate(T, f"LightGBM + 무작위 지문 (시드 {s})", "regret@3").mean(), 2)
                                    for name, T in (("처음 보는 출발물질", Ts), ("처음 보는 고리", Ta))}} for s in FP_SEEDS])
    near = {m: Ts[Ts.method == m].near.mean() for m in ORDER}
    saving = " · ".join(f"{m} {near[m]:.1f}번 ({1 - near[m] / near['무작위']:+.0%})" for m in ORDER)
    ties = {m: int(Ts[Ts.method == m].ties.sum()) for m in ORDER[1:]}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = {"무작위": "#c3c2b7", "조건 평균": "#9a9893", "할로젠별 조건 평균": "#86b6ef", "화학자 규칙 모델": "#eb6834",
              "LightGBM + 분자 특성": "#2a78d6", "LightGBM + 무작위 지문 (3개 평균)": "#1baf7a"}
    for a, (name, T) in zip(ax, (("(a) 처음 보는 출발물질", Ts), ("(b) 처음 보는 고리", Ta))):
        for m in ORDER:
            t = T[T.method == m]
            a.plot(range(1, 13), [t[f"hit@{k}"].mean() for k in range(1, 13)], marker="o", ms=4, lw=2, color=colors[m], label=m)
        a.set(title=name, xlabel="추천 상위 k개 조건만 실험", xticks=range(1, 13), ylim=(0, 1.02))
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e1e0d9")
    ax[0].set_ylabel("최고 조건이 포함될 확률")
    ax[1].legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "model4_topk.png", dpi=110)
    plt.close(fig)

    OUT.write_text(
        "# 반응 수율 4차 결과: 조건 추천 (generated by reaction/model4.py)\n\n"
        "상황 = (출발물질, 첨가제) 330개, 고르는 조건 = 리간드 × 염기 11~12개. 구간은 출발물질 15종 단위 부트스트랩 1,000회. "
        f"화학자 규칙 모델의 정규화 강도(묶음별): 출발물질 {alphas['aryl_halide']}, 고리 {alphas['arene']}. 실행 {(time.time() - t0) / 60:.0f}분.\n\n"
        f"## 판정\n{vt}\n"
        f"## 처음 보는 출발물질\n{md(summary(Ts), index=False)}\n\n"
        f"최고와 5%p 이내 조건을 찾을 때까지 실험 수 (무작위 대비): {saving}\n\n"
        f"예측값 동점 조건 수 합계: {ties}\n\n"
        f"## 처음 보는 고리\n{md(summary(Ta), index=False)}\n\n"
        f"## 민감도: 주 판정\n{md(sens, index=False)}\n\n"
        f"## 무작위 지문 시드별 regret@3 평균\n{md(fp, index=False)}\n\n"
        f"## 민감도: SM 처음 보는 리간드에서 '리간드 없음' 제외 (MAE, 평가에서만 제외)\n{md(sm_ligand_without_none(), index=False)}\n\n"
        "그림: [figures/model4_topk.png](figures/model4_topk.png)\n", encoding="utf-8")
    print(vt)


if __name__ == "__main__":
    main()
