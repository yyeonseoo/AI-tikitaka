"""Post-hoc analysis (not in any plan) of the reaction-yield models.
A. Is the LightGBM advantage on unseen substrates / arenes from the model or from the molecular features?
   Ridge regression on the same molecular features (alpha chosen by the same inner leave-one-group-out), same outer folds;
   LightGBM and simple-baseline predictions are read from the checkpoints of model2.py / model3.py.
B. SHAP (LightGBM pred_contrib): which reagent slot and which molecular properties drive the predictions.
Usage: python reaction/analysis.py   (-> reaction/results_analysis.md, reaction/figures/analysis_shap.png)"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eda import md  # noqa: E402
from model import boot_ci, fmt, folds, per_group_mae  # noqa: E402
from model2 import CACHE, desc_bh, desc_sm, knowledge_bh, knowledge_sm  # noqa: E402
from prepare import load_bh, load_sm  # noqa: E402

OUT, FIG = ROOT / "results_analysis.md", ROOT / "figures"
SEED = 0
ALPHAS = [0.1, 1, 10, 100, 1000]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SLOT_KO = {"ligand": "리간드", "base": "염기", "additive": "첨가제", "aryl_halide": "출발물질", "electrophile": "친전자체",
           "nucleophile": "친핵체", "solvent": "용매", "K": "화학 지식"}


def ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha))


def part_a(name, cache_name, d, group, X):
    """Ridge on molecular features vs the checkpointed LightGBM and best simple baseline."""
    Xa, y = X.to_numpy(float), d.y.to_numpy(float)
    p_ridge = np.full(len(d), np.nan)
    saved_all = {}
    for f, tr, va, te in folds(d, group, np.random.default_rng(SEED)):
        trn = np.concatenate([tr, va])
        inner = list(LeaveOneGroupOut().split(trn, groups=d[group].to_numpy()[trn]))
        cv = {a: np.mean([mean_absolute_error(y[trn][b], ridge(a).fit(Xa[trn][i], y[trn][i]).predict(Xa[trn][b])) for i, b in inner])
              for a in ALPHAS}
        a = min(cv, key=cv.get)
        p_ridge[te] = np.clip(ridge(a).fit(Xa[trn], y[trn]).predict(Xa[te]), 0, 100)
        saved, _ = pickle.loads((CACHE / f"{cache_name}_fold{f}.pkl").read_bytes())
        for k in ("최선 단순 (안쪽 검증 선택)", "LGBM·F1 (주)"):
            saved_all.setdefault(k, np.full(len(d), np.nan))[te] = saved[k]
    P = {"최선 단순 (분자 특성 없음)": saved_all["최선 단순 (안쪽 검증 선택)"], "선형 회귀 + 분자 특성": p_ridge,
         "LightGBM + 분자 특성": saved_all["LGBM·F1 (주)"]}
    unit = "aryl_halide" if group == "arene" else group
    rows = [{"상황": name, "방법": k, "MAE (%p)": fmt(boot_ci(per_group_mae(d, p, unit)))} for k, p in P.items()]
    diffs = [{"상황": name, "비교": f"{a} − {b}", "MAE 차이": fmt(boot_ci(per_group_mae(d, P[a], unit) - per_group_mae(d, P[b], unit)), True)}
             for a, b in (("선형 회귀 + 분자 특성", "최선 단순 (분자 특성 없음)"), ("LightGBM + 분자 특성", "선형 회귀 + 분자 특성"))]
    return rows, diffs


def shap_slots(d, X):
    """LightGBM on all rows; mean |SHAP| per feature, summed per reagent slot."""
    m = LGBMRegressor(num_leaves=31, n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1).fit(X.to_numpy(float), d.y)
    S = pd.DataFrame(m.booster_.predict(X.to_numpy(float), pred_contrib=True)[:, :-1], columns=X.columns)
    imp = S.abs().mean()
    slot = imp.groupby([c.split(":")[0] for c in imp.index]).sum().sort_values(ascending=False)
    return imp.sort_values(ascending=False), slot, S


def pretty(c):
    s, f = c.split(":", 1)
    return f"{SLOT_KO.get(s, s)}: {f}"


def main():
    FIG.mkdir(exist_ok=True)
    bh, sm = load_bh(), load_sm()
    Dbh, Kbh, Dsm, Ksm = desc_bh(bh), knowledge_bh(bh), desc_sm(sm), knowledge_sm(sm)

    rows, diffs = [], []
    for name, cache, d, g, X in (("BH 처음 보는 출발물질", "BH_처음_보는_기질", bh, "aryl_halide", Dbh),
                                 ("BH 처음 보는 고리", "BH_처음_보는_고리", bh, "arene", Dbh),
                                 ("SM 처음 보는 리간드", "SM_처음_보는_리간드", sm, "ligand", Dsm)):
        r, df = part_a(name, cache, d, g, X)
        rows += r
        diffs += df
        print(name, "done", flush=True)

    imp_bh, slot_bh, S_bh = shap_slots(bh, Dbh.join(Kbh))
    imp_sm, slot_sm, _ = shap_slots(sm, Dsm.join(Ksm))
    sub = [c for c in imp_bh.index if c.startswith(("aryl_halide:", "K:"))][:10]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, slot, title, col in ((ax[0], slot_bh, "(a) BH: 시약 칸별 영향 (평균 |SHAP| 합)", BLUE),
                                (ax[1], slot_sm, "(b) SM: 시약 칸별 영향 (평균 |SHAP| 합)", ORANGE)):
        t = slot[::-1]
        a.barh([SLOT_KO.get(i, i) for i in t.index], t.values, color=col, height=0.6)
        a.set(title=title, xlabel="수율 예측에 준 평균 영향 (%p)")
    t = imp_bh[sub][::-1]
    ax[2].barh([pretty(c) for c in t.index], t.values, color=GRAY, height=0.6)
    ax[2].set(title="(c) BH: 출발물질 특성과 화학 지식 중 영향 상위 10개", xlabel="평균 |SHAP| (%p)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "analysis_shap.png", dpi=110)
    plt.close(fig)

    top = lambda imp: pd.DataFrame({"특성": [pretty(c) for c in imp.index[:15]], "평균 |SHAP| (%p)": imp.values[:15].round(2)})
    OUT.write_text(
        "# 반응 수율 사후 분석 (generated by reaction/analysis.py)\n\n계획에 없던 분석입니다. 판정에 쓰지 않습니다.\n\n"
        "## A. 이긴 이유는 모델인가, 분자 특성인가\n같은 분할, 같은 안쪽 검증(그룹 하나 빼기)으로 선형 회귀(능형)에 분자 특성을 넣어 비교. "
        "LightGBM과 단순 방법의 예측은 2·3차 실행 저장본을 그대로 씀.\n\n"
        f"{md(pd.DataFrame(rows), index=False)}\n\n{md(pd.DataFrame(diffs), index=False)}\n\n"
        "## B. SHAP: 무엇이 예측을 움직이나 (전체 데이터로 학습한 LightGBM, 분자 특성 + 화학 지식)\n"
        "SHAP = 반응마다 예측 수율을 특성별 기여로 나눈 값(%p). 칸별 값은 그 칸 특성들의 평균 |SHAP| 합.\n\n"
        f"### BH 칸별\n{md(slot_bh.round(2).rename(index=SLOT_KO).to_frame('평균 |SHAP| 합'))}\n\n"
        f"### SM 칸별\n{md(slot_sm.round(2).rename(index=SLOT_KO).to_frame('평균 |SHAP| 합'))}\n\n"
        f"### BH 특성 상위 15\n{md(top(imp_bh), index=False)}\n\n### SM 특성 상위 15\n{md(top(imp_sm), index=False)}\n\n"
        "그림: [figures/analysis_shap.png](figures/analysis_shap.png)\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
