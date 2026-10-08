"""RAPID EDA for the step-3 plan (condition recommendation; see rapid/README.md).
Solvent comes from the provided Excel (data/RAPID_*.xlsx, same experiments, joined by experiment ID).
Usage: python rapid/eda.py   (-> rapid/results_eda.md, rapid/figures/eda_*.png)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from step1 import AMINE, load, md  # noqa: E402

OUT, FIG = ROOT / "results_eda.md", ROOT / "figures"
CONC = {"_rxn_M_inorganic": "PbI₂ (M)", "_rxn_M_organic": "아민 (M)", "_rxn_M_acid": "포름산 (M)"}
PROC = {"_rxn_temperatureC_actual_bulk": "온도 (°C)", "_rxn_reactiontimeS": "반응 시간 (s)", "_rxn_stirrateRPM": "교반 (rpm)",
        "_rxn_mixingtime1S": "혼합 시간 1 (s)", "_rxn_mixingtime2S": "혼합 시간 2 (s)"}
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def data():
    d, feat, y = load()
    x = pd.read_excel(next((ROOT / "data").glob("RAPID*.xlsx")), "실험별 물성", header=4)
    d = d.merge(x[["실험 ID", "용매", "아민(SMILES)"]], left_on="name", right_on="실험 ID", how="left", validate="1:1")
    assert d["용매"].notna().all()
    d["y"] = y
    d["amine_pb"] = d._rxn_M_organic / d._rxn_M_inorganic.replace(0, np.nan)
    d["acid_pb"] = d._rxn_M_acid / d._rxn_M_inorganic.replace(0, np.nan)
    d["model_chosen"] = d._raw_model_predicted > 0
    return d.copy(), feat


def within_amine_diff(d, flag):
    """Success-rate difference (flag - not flag) inside amines that have both groups, averaged over those amines."""
    g = d.groupby([AMINE, d[flag].astype(int)]).y.mean().unstack().dropna()
    return (g[1] - g[0]).mean(), len(g)


def amine_table(d, feat):
    A = d.groupby(AMINE).agg(smiles=("아민(SMILES)", "first"), solvent=("용매", lambda s: s.mode()[0]),
                             n=("y", "size"), rate=("y", "mean"), **{f: (f, "first") for f in feat})
    A["eff_radius"] = (3 * A["_feat_VanderWaalsVolume"] / (4 * np.pi)) ** (1 / 3)  # Å, sphere of equal volume
    A["n_ammonium"] = A.smiles.str.count(r"NH3\+|NH2\+|NH\+")
    return A


def descriptor_groups(A, feat, thr=0.95):
    """Greedy grouping of the 67 amine descriptors by |Spearman| >= thr across the 45 amines."""
    X = A[feat].loc[:, A[feat].nunique() > 1]
    C = X.rank().corr().abs()
    left, groups = list(C.columns), []
    while left:
        f = left[0]
        g = [c for c in left if C.loc[f, c] >= thr]
        groups.append(g)
        left = [c for c in left if c not in g]
    return len(feat) - X.shape[1], groups


def main():
    FIG.mkdir(exist_ok=True)
    d, feat = data()
    A = amine_table(d, feat)
    s = []

    # 0. overview
    s.append(f"## 0. 개요\n{len(d)}건 (점수 0 제외), 아민 {d[AMINE].nunique()}종, 4점 비율 {d.y.mean():.3f}. "
             f"실험실: {d._raw_lab.value_counts().to_dict()}\n")

    # 1. solvent x concentration coverage
    rows = []
    for sol, g in d.groupby("용매"):
        r = {"용매": sol, "실험": len(g), "아민": g[AMINE].nunique(), "성공 있는 아민": (g.groupby(AMINE).y.max() > 0).sum(),
             "4점 비율": round(g.y.mean(), 3)}
        for c, name in CONC.items():
            q = g[c].quantile([0.05, 0.5, 0.95]).round(2).tolist()
            r[name + " 5/50/95%"] = "/".join(map(str, q))
        rows.append(r)
    multi = (d[d["용매"] != "미기록"].groupby(AMINE)["용매"].nunique() > 1).sum()  # unrecorded solvent is not a second solvent
    s.append(f"## 1. 용매별 농도 범위\n{md(pd.DataFrame(rows), index=False)}\n\n용매를 두 가지 이상 쓴 아민 (미기록 제외): {multi}종\n")

    # success vs ratio (pooled and within amine)
    rows = []
    for col, name in (("amine_pb", "아민/PbI₂"), ("acid_pb", "포름산/PbI₂")):
        b = pd.qcut(d[col], 5, duplicates="drop")
        pooled = d.groupby(b, observed=True).y.mean()
        within = d.assign(q=d.groupby(AMINE)[col].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False)))
        w = within.groupby(["q", AMINE]).y.mean().groupby(level="q").mean()
        rows.append({"비율": name, "구간(전체 5분위)": " | ".join(str(i) for i in pooled.index.astype(str)),
                     "4점 비율(전체)": " | ".join(f"{v:.2f}" for v in pooled.values),
                     "아민 안 5분위별 4점 비율(아민 평균)": " | ".join(f"{v:.2f}" for v in w.values)})
    s.append(f"### 몰비와 성공률\n{md(pd.DataFrame(rows), index=False)}\n")

    # 2. process variables
    rows = []
    for c, name in PROC.items():
        vc = d[c].round(0).value_counts()
        rows.append({"변수": name, "값 종류": len(vc), "최빈값 비율": round(vc.iloc[0] / len(d), 3),
                     "값별 실험 수": str(vc.head(5).to_dict())})
    t_hi, n_t = within_amine_diff(d.assign(hot=d._rxn_temperatureC_actual_bulk >= 90), "hot")
    s.append(f"## 2. 공정 변수\n{md(pd.DataFrame(rows), index=False)}\n\n"
             f"온도 90°C 이상 − 미만의 4점 비율 차이 (두 구간이 모두 있는 아민 {n_t}종 평균): {t_hi:+.3f}\n")

    # 3. odd rows
    odd = {"PbI₂ = 0": d._rxn_M_inorganic == 0, "아민 = 0": d._rxn_M_organic == 0, "포름산 = 0": d._rxn_M_acid == 0,
           "용매 미기록": d["용매"] == "미기록"}
    rows = [{"조건": k, "실험 수": int(m.sum()), "점수 분포": str(d.loc[m, "_out_crystalscore"].value_counts().sort_index().to_dict())}
            for k, m in odd.items()]
    s.append(f"## 3. 이상한 실험\n{md(pd.DataFrame(rows), index=False)}\n")

    # 4. descriptors
    n_const, groups = descriptor_groups(A, feat)
    geo = {"eff_radius": "유효 반지름 (Å)", "_feat_MinimalProjectionArea": "최소 단면적 (Å²)", "_feat_VanderWaalsVolume": "부피 (Å³)",
           "n_ammonium": "암모늄기 수", "_feat_RotatableBondCount": "회전 결합 수", "_feat_AromaticRingCount": "방향족 고리 수",
           "_feat_donorcount": "수소결합 주개 수"}
    rows = []
    for c, name in geo.items():
        r = spearmanr(A[c], A.rate)
        rows.append({"변수": name, "범위": f"{A[c].min():.1f}~{A[c].max():.1f}", "아민 성공률과 순위 상관": round(r.statistic, 3),
                     "p": round(r.pvalue, 4)})
    thr = []
    for c, cut, name in (("eff_radius", 2.6, "유효 반지름 ≤ 2.6 Å"),):  # 40 Å² cross-section rule: every amine here is <= 34 Å²
        m = A[c] <= cut
        thr.append({"기준": name, "해당 아민": int(m.sum()), "평균 성공률(해당)": round(A.rate[m].mean(), 3),
                    "평균 성공률(나머지)": round(A.rate[~m].mean(), 3)})
    s.append(f"## 4. 아민 물성\n아민 안에서 변하지 않는 값이라, 아민 45종 수준에서 봅니다(탐색용, 다중 비교 보정 없음).\n\n"
             f"- 67개 중 아민 간에 값이 같은(변별력 없는) 물성: {n_const}개\n"
             f"- 나머지를 |순위 상관| ≥ 0.95로 묶으면 {len(groups)}개 묶음 (가장 큰 묶음 {max(map(len, groups))}개)\n\n"
             f"### 기하 변수와 아민 성공률\n{md(pd.DataFrame(rows), index=False)}\n\n### 크기 기준\n{md(pd.DataFrame(thr), index=False)}\n")

    # 5. model-chosen experiments
    diff, n_m = within_amine_diff(d, "model_chosen")
    s.append(f"## 5. 원 연구팀 모델이 고른 실험\n전체 {d.model_chosen.mean():.1%}. 4점 비율: 모델 선택 {d[d.model_chosen].y.mean():.3f}, "
             f"그 외 {d[~d.model_chosen].y.mean():.3f}. 두 종류가 모두 있는 아민 {n_m}종 안에서의 평균 차이: {diff:+.3f}\n")

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, sol in zip(ax, ["GBL", "DMF", "DMSO"]):
        g = d[d["용매"] == sol]
        a.scatter(g._rxn_M_inorganic[g.y == 0], g._rxn_M_organic[g.y == 0], s=3, c="#BBBBBB", alpha=0.4, label="1~3점")
        a.scatter(g._rxn_M_inorganic[g.y == 1], g._rxn_M_organic[g.y == 1], s=5, c="#C44E52", alpha=0.7, label="4점")
        a.set(title=f"{sol} ({len(g)}건, 4점 {g.y.mean():.0%})", xlabel="PbI₂ 농도 (M)", ylabel="아민 농도 (M)")
        a.legend(fontsize=8, markerscale=3)
    fig.suptitle("용매별 농도 조합과 결과 (점 하나 = 실험 하나)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_concentration.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, (c, name, cut) in zip(ax, [("eff_radius", "유효 반지름 (Å)", 2.6), ("_feat_MinimalProjectionArea", "최소 단면적 (Å²)", 40),
                                      ("_feat_RotatableBondCount", "회전 결합 수", None)]):
        a.scatter(A[c], A.rate, c=["#4C72B0" if n > 1 else "#DD8452" for n in A.n_ammonium], s=30)
        if cut:
            a.axvline(cut, color="#C44E52", ls="--", lw=1)
        a.set(xlabel=name, ylabel="아민별 4점 비율", title=f"{name} (순위 상관 {spearmanr(A[c], A.rate).statistic:+.2f})")
    fig.suptitle("아민 45종의 생김새와 성공률 (파랑 = 암모늄기 2개 이상, 주황 = 1개; 빨간 점선 = 문헌 기준)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_amine_geometry.png", dpi=110)
    plt.close(fig)

    OUT.write_text("# RAPID EDA (generated by rapid/eda.py)\n\n"
                   "그림: [농도 조합](figures/eda_concentration.png), [아민 생김새](figures/eda_amine_geometry.png)\n\n"
                   + "\n".join(s), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
