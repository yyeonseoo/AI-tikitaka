"""Plain performance check of the PCE model inside <= 2017 data (2018+ stays untouched for experiment 2).
Train = train-role papers <= 2017, test = test-role papers <= 2017 (splits/doi_split.csv), hyperparameters from
best_params_past.json. Reports regression metrics, a 'high efficiency (>= 15%)' classification view
(accuracy / precision / recall / F1 / ROC-AUC / PR-AUC, threshold chosen on train out-of-fold predictions),
paper-bootstrap 95% intervals, SHAP importance (LightGBM's built-in pred_contrib), and a summary of experiment 1.
Usage: python models/v2/evaluate_past.py   (-> models/v2/evaluation_past.md, models/v2/figures/eval_*.png)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import v2 as C
from experiments.exp1.exp1_hide_top import BEST, load_past
from experiments.exp1.exp1_bootstrap_baselines import ridge
from models.v2.model_v2 import ALL, Model
from src.data import split_from_file
from src.features import md

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
EXP1 = ROOT / "experiments" / "exp1"
HIGH = 15.0  # "high efficiency" cut-off for the classification view
N_BOOT = 1000
if "Malgun Gothic" in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams["font.family"] = "Malgun Gothic"  # Korean labels on Windows
plt.rcParams["axes.unicode_minus"] = False
NAME = {"solvent": "용매", "dmso_frac": "DMSO 비율", "antisolvent": "반용매", "anneal_temp": "어닐링 온도",
        "anneal_time": "어닐링 시간", "additive": "첨가제", "solvent_annealing": "용매 어닐링", "arch": "소자 구조",
        "backcontact": "후면 전극", "flexible": "유연 기판", "etl": "전자수송층", "htl": "정공수송층",
        "year": "출판 연도", "scan_direction": "측정 스캔 방향"}
GROUP_COLOR = {**{v: "#4C72B0" for v in C.PROCESS}, **{v: "#DD8452" for v in C.CONDITION},
               **{v: "#8C8C8C" for v in C.CORRECTION}}


def lgbm():
    return Model(BEST["model"], BEST["params"], BEST["missing"] == "median+flag", ALL)


def oof(train, fit):
    p = np.empty(len(train))
    for a, b in GroupKFold(C.CV_FOLDS).split(train, groups=train["doi"]):
        p[b] = fit(train.iloc[a]).predict(train.iloc[b])
    return p


def best_threshold(score, y_high):
    """Score cut-off that maximizes F1 on train out-of-fold predictions (test is never looked at)."""
    cands = np.unique(np.quantile(score, np.linspace(0.3, 0.98, 200)))
    return max(cands, key=lambda t: f1_score(y_high, score >= t))


def metrics(y, p, t):
    hi, ph = y >= HIGH, p >= t
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(np.mean((y - p) ** 2)), "R2": r2_score(y, p),
            "Spearman": spearmanr(y, p).statistic, "ROC-AUC": roc_auc_score(hi, p), "PR-AUC": average_precision_score(hi, p),
            "accuracy": accuracy_score(hi, ph), "precision": precision_score(hi, ph, zero_division=0),
            "recall": recall_score(hi, ph), "F1": f1_score(hi, ph)}


def paper_bootstrap(te, preds, thresholds, rng):
    groups = te.groupby("doi").indices
    papers = np.array(list(groups))
    rows = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[d] for d in rng.choice(papers, len(papers), replace=True)])
        y = te.pce.values[idx]
        rows.append({name: metrics(y, p[idx], thresholds[name]) for name, p in preds.items()})
    return rows


def shap_values(model, X):
    """Per-feature SHAP from LightGBM's pred_contrib; missing-value flag columns are added to their variable."""
    Xt = model.prep.transform(X)
    contrib = model.m.predict(Xt, pred_contrib=True)[:, :-1]
    S = pd.DataFrame(contrib, columns=Xt.columns, index=X.index)
    for c in [c for c in S if c.endswith("_missing")]:
        S[c.removesuffix("_missing")] += S.pop(c)
    return S[ALL]


def fig_overview(te, preds, thresholds, M, CI):
    fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))
    y, p = te.pce.values, preds["LightGBM"]
    a = ax[0, 0]
    a.scatter(y, p, s=6, alpha=0.3, color="#4C72B0")
    a.plot([0, 25], [0, 25], "k--", lw=1)
    a.axhline(thresholds["LightGBM"], color="#C44E52", lw=0.8, ls=":")
    a.axvline(HIGH, color="#C44E52", lw=0.8, ls=":")
    a.set(xlim=(0, 25), ylim=(0, 25), xlabel="실제 효율 PCE (%)", ylabel="예측 효율 (%)",
          title=f"(a) 예측 vs 실제  R² {M.loc['LightGBM', 'R2']:.2f}, MAE {M.loc['LightGBM', 'MAE']:.2f}%p")
    a = ax[0, 1]
    names = ["평균값 찍기", "선형 회귀(Ridge)", "LightGBM"]
    keys = ["train 평균", "Ridge", "LightGBM"]
    xs = np.arange(3)
    for j, (m, col) in enumerate([("R2", "#4C72B0"), ("Spearman", "#55A868"), ("ROC-AUC", "#DD8452")]):
        vals = [M.loc[k, m] for k in keys]
        lo = [vals[i] - CI[k][m][0] for i, k in enumerate(keys)]
        hi = [CI[k][m][1] - vals[i] for i, k in enumerate(keys)]
        a.bar(xs + (j - 1) * 0.26, vals, 0.26, yerr=[lo, hi], capsize=3, color=col,
              label={"R2": "R² (설명력)", "Spearman": "순위 상관", "ROC-AUC": "ROC-AUC (고효율 판별)"}[m])
    a.axhline(0.5, color="#DD8452", lw=0.8, ls=":")
    a.set_xticks(xs, names)
    a.set(ylim=(0, 1), title="(b) 모델별 성능 (막대 위 선 = 95% 구간)")
    a.legend(fontsize=8, loc="upper left")
    hi_y = y >= HIGH
    a = ax[1, 0]
    for k, col in [("LightGBM", "#4C72B0"), ("Ridge", "#55A868")]:
        fpr, tpr, _ = roc_curve(hi_y, preds[k])
        a.plot(fpr, tpr, color=col, label=f"{k} (AUC {M.loc[k, 'ROC-AUC']:.2f})")
    a.plot([0, 1], [0, 1], "k--", lw=1, label="무작위 (AUC 0.50)")
    a.set(xlabel="고효율이 아닌데 고효율로 판정한 비율", ylabel="고효율을 맞게 찾은 비율",
          title=f"(c) 고효율(≥{HIGH:g}%) 판별 ROC 곡선")
    a.legend(fontsize=8)
    a = ax[1, 1]
    cm = confusion_matrix(hi_y, preds["LightGBM"] >= thresholds["LightGBM"])
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum():.0%})", ha="center", va="center", fontsize=11,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1], ["예측: 보통", "예측: 고효율"])
    a.set_yticks([0, 1], ["실제: 보통", "실제: 고효율"])
    a.set_title(f"(d) LightGBM 판정 결과 (정확도 {M.loc['LightGBM', 'accuracy']:.0%}, F1 {M.loc['LightGBM', 'F1']:.2f})")
    fig.suptitle("2017년까지 데이터: 학습에 안 쓴 논문 test 1,440행 평가", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "eval_overview.png", dpi=120)
    plt.close(fig)


def fig_shap(S, X):
    imp = S.abs().mean().sort_values()
    fig = plt.figure(figsize=(13, 9))
    a = fig.add_subplot(1, 2, 1)
    a.barh([NAME[c] for c in imp.index], imp.values, color=[GROUP_COLOR[c] for c in imp.index])
    a.set(xlabel="평균 |SHAP| (예측 효율을 평균에서 몇 %p 움직이는가)", title="(a) 변수 중요도 (SHAP)")
    a.legend(handles=[Patch(color=col, label=g) for g, col in
                      [("공정 (추천 대상)", "#4C72B0"), ("소자 조건", "#DD8452"), ("보정 (연도·측정)", "#8C8C8C")]],
             loc="lower right", fontsize=9)
    top = imp.index[::-1][:6]
    for i, c in enumerate(top):
        a = fig.add_subplot(3, 4, [3, 4, 7, 8, 11, 12][i])
        if c in C.NUMERIC:
            a.scatter(X[c], S[c], s=5, alpha=0.3, color=GROUP_COLOR[c])
            a.set_xlabel(NAME[c], fontsize=8)
        else:
            v = X[c].fillna("기록없음").astype(str)
            keep = v.value_counts().index[:5]
            d = pd.DataFrame({"v": v.where(v.isin(keep), "기타"), "s": S[c]}).groupby("v").s.mean().sort_values()
            a.barh([k[:16] for k in d.index], d.values, color=GROUP_COLOR[c])
            a.tick_params(axis="y", labelsize=7)
        a.axhline(0, color="k", lw=0.5) if c in C.NUMERIC else a.axvline(0, color="k", lw=0.5)
        a.set_title(f"{NAME[c]}: 값에 따른 효과 (%p)", fontsize=9)
        a.tick_params(labelsize=7)
    fig.suptitle("무엇이 예측 효율을 올리고 내리는가 (test 데이터, LightGBM)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "eval_shap.png", dpi=120)
    plt.close(fig)
    return imp[::-1]


def fig_exp1():
    """Experiment 1 summary from the saved bootstrap CSVs (no retraining)."""
    runs = [pd.read_csv(p) for p in sorted(EXP1.glob("bootstrap_*.csv"))]
    if not runs:
        return None
    R = pd.concat(runs)
    R = R[(R.rep >= 0) & (R.setting == "D1: hide answer papers")].drop_duplicates(["rep", "method"])
    label = {"v2": "LightGBM (기본)", "B drop PCE<5": "LightGBM, 실패 소자 제외", "ensemble k=1": "LightGBM 10개 + 불확실성",
             "ensemble k=0": "LightGBM 10개 평균", "Ridge": "선형 회귀(Ridge)", "A2 within-paper ranker": "논문 안 순위 학습",
             "kNN k=5": "비슷한 조합 평균"}
    g = R[R.method.isin(label)].groupby("method").top10_share
    t = pd.DataFrame({"mean": g.mean(), "lo": g.quantile(0.025), "hi": g.quantile(0.975)}).sort_values("mean")
    fig, a = plt.subplots(figsize=(8, 4.5))
    a.barh([label[m] for m in t.index], t["mean"], xerr=[t["mean"] - t.lo, t.hi - t["mean"]], capsize=3,
           color=["#4C72B0" if m == "v2" else "#A0B4D0" for m in t.index])
    a.axvline(0.10, color="#C44E52", ls="--", label="무작위로 고를 때 (10%)")
    a.set(xlabel="숨긴 고효율 조합 중 추천 상위 10% 안에 들어온 비율", xlim=(0, 0.7),
          title="실험 1: 숨긴 고효율 조합을 얼마나 찾아내나 (논문 재표집 200회, 선 = 95% 구간)")
    a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "eval_exp1_summary.png", dpi=120)
    plt.close(fig)
    return t


def main():
    F = load_past()
    tr_i, te_i = split_from_file(F["doi"])
    tr, te = F.iloc[tr_i].reset_index(drop=True), F.iloc[te_i].reset_index(drop=True)
    print(f"train {len(tr)} rows / {tr.doi.nunique()} papers, test {len(te)} rows / {te.doi.nunique()} papers (<= 2017)")

    fits = {"LightGBM": lambda d: lgbm().fit(d, d.pce), "Ridge": lambda d: ridge(d)[0]}
    thresholds, preds = {}, {}
    for name, fit in fits.items():
        thresholds[name] = best_threshold(oof(tr, fit), tr.pce.values >= HIGH)
        preds[name] = fit(tr).predict(te)
    thresholds["train 평균"] = tr.pce.mean()
    preds["train 평균"] = np.full(len(te), tr.pce.mean()) + np.random.default_rng(C.SEED).normal(0, 1e-6, len(te))
    M = pd.DataFrame({k: metrics(te.pce.values, p, thresholds[k]) for k, p in preds.items()}).T
    boot = paper_bootstrap(te, preds, thresholds, np.random.default_rng(C.SEED))
    CI = {k: {m: np.percentile([b[k][m] for b in boot], [2.5, 97.5]) for m in M.columns} for k in preds}
    d_r2 = np.array([b["LightGBM"]["R2"] - b["Ridge"]["R2"] for b in boot])
    d_auc = np.array([b["LightGBM"]["ROC-AUC"] - b["Ridge"]["ROC-AUC"] for b in boot])

    lg = te.assign(pred=preds["LightGBM"])
    top10 = lg[lg.pred >= lg.pred.quantile(0.9)]
    lgbm_model = lgbm().fit(tr, tr.pce)
    S = shap_values(lgbm_model, te)

    fig_overview(te, preds, thresholds, M, CI)
    imp = fig_shap(S, te)
    t1 = fig_exp1()

    tab = M.astype(object)
    for k in tab.index:
        for m in tab.columns:
            tab.loc[k, m] = f"{M.loc[k, m]:.3f} ({CI[k][m][0]:.3f}–{CI[k][m][1]:.3f})"
    (OUT / "evaluation_past.md").write_text(
        "# PCE model evaluation inside <= 2017 (generated by models/v2/evaluate_past.py)\n\n"
        f"train {len(tr)} rows / {tr.doi.nunique()} papers, test {len(te)} rows / {te.doi.nunique()} papers; "
        f"high efficiency = PCE >= {HIGH:g}% (test base rate {np.mean(te.pce >= HIGH):.1%}); "
        f"thresholds from train OOF F1: {', '.join(f'{k} {v:.2f}' for k, v in thresholds.items())}\n\n"
        f"## metrics on test (95% paper-bootstrap interval, {N_BOOT} resamples)\n{md(tab)}\n\n"
        f"always-'not high' accuracy: {np.mean(te.pce < HIGH):.3f}; random F1 at base rate: "
        f"{np.mean(te.pce >= HIGH):.3f}\n\n"
        f"LightGBM - Ridge (paired, same resamples): R2 {d_r2.mean():+.3f} ({np.percentile(d_r2, 2.5):+.3f} to "
        f"{np.percentile(d_r2, 97.5):+.3f}), ROC-AUC {d_auc.mean():+.3f} ({np.percentile(d_auc, 2.5):+.3f} to "
        f"{np.percentile(d_auc, 97.5):+.3f})\n\n"
        f"top 10% predicted (LightGBM): {len(top10)} rows, actual mean {top10.pce.mean():.1f}% vs all {te.pce.mean():.1f}%, "
        f"share >= {HIGH:g}%: {np.mean(top10.pce >= HIGH):.0%} vs {np.mean(te.pce >= HIGH):.0%}\n\n"
        f"## SHAP mean |value| (%p)\n{md(imp.round(3).to_frame('mean |SHAP|'))}\n\n"
        + (f"## experiment 1 summary (D1, top-10% share)\n{md(t1.round(3))}\n" if t1 is not None else ""),
        encoding="utf-8")

    print(md(M.round(3)))
    print(f"base rate >= {HIGH:g}%: {np.mean(te.pce >= HIGH):.1%}; always-'not high' accuracy {np.mean(te.pce < HIGH):.1%}")
    print(f"LightGBM - Ridge: R2 {d_r2.mean():+.3f} ({np.percentile(d_r2, 2.5):+.3f}..{np.percentile(d_r2, 97.5):+.3f}),"
          f" AUC {d_auc.mean():+.3f} ({np.percentile(d_auc, 2.5):+.3f}..{np.percentile(d_auc, 97.5):+.3f})")
    print("LightGBM CI: " + ", ".join(f"{m} {CI['LightGBM'][m][0]:.2f}-{CI['LightGBM'][m][1]:.2f}"
                                       for m in ["R2", "ROC-AUC", "F1", "accuracy"]))
    print(f"top10% predicted actual >= {HIGH:g}%: {np.mean(top10.pce >= HIGH):.0%} (all {np.mean(te.pce >= HIGH):.0%})")
    print("SHAP top: " + ", ".join(f"{NAME[k]} {v:.2f}" for k, v in imp.head(8).items()))
    print("figures: models/v2/figures/eval_overview.png, eval_shap.png, eval_exp1_summary.png")


if __name__ == "__main__":
    main()
