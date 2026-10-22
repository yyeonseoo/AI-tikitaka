"""Reaction-yield EDA (step 2 of the plan in reaction/README.md).
How yields are distributed, how much each reagent slot explains on its own, and how much the pairwise combinations add.
Usage: python reaction/eda.py   (-> reaction/results_eda.md, reaction/figures/eda_summary.png)"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prepare import BH_SLOTS, SM_SLOTS, load_bh, load_sm  # noqa: E402

OUT, FIG = ROOT / "results_eda.md", ROOT / "figures"
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9a9893", "#52514e"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def md(t, index=True):
    t = t.reset_index() if index else t
    rows = [list(map(str, t.columns))] + [list(map(str, r)) for r in t.values.tolist()]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def explained(d, keys):
    """Share of yield variance explained by the group means of `keys` (in-sample; descriptive only)."""
    m = d.groupby(keys).y.transform("mean")
    return 1 - ((d.y - m) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()


def additive_fit(d, slots, pairs=False):
    """In-sample R² of an additive model: main effects (and all pairwise effects) fitted by least squares on one-hot."""
    X = pd.get_dummies(d[slots].astype(str))
    if pairs:
        for a, b in combinations(slots, 2):
            X = X.join(pd.get_dummies(d[a].astype(str) + "×" + d[b].astype(str), prefix=f"{a}×{b}"))
    A = np.column_stack([np.ones(len(d)), X.to_numpy(float)])
    coef, *_ = np.linalg.lstsq(A, d.y.to_numpy(), rcond=None)
    p = A @ coef
    return 1 - ((d.y - p) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()


def section(name, d, slots):
    y = d.y
    s = [f"## {name}: {len(d)}건\n",
         f"수율 평균 {y.mean():.1f}%, 중앙값 {y.median():.1f}%, 1% 미만 {np.mean(y < 1):.0%}, 80% 이상 {np.mean(y >= 80):.0%}\n"]
    one = pd.Series({c: explained(d, [c]) for c in slots}).sort_values(ascending=False)
    pair = pd.Series({f"{a} × {b}": explained(d, [a, b]) for a, b in combinations(slots, 2)}).sort_values(ascending=False)
    main, withpairs = additive_fit(d, slots), additive_fit(d, slots, pairs=True)
    s.append("### 칸 하나가 설명하는 수율 차이 (그 칸 시약별 평균만으로, 학습 데이터 안에서의 설명력)\n"
             + md(one.round(3).to_frame("설명 비율")) + "\n")
    s.append("### 두 칸 조합이 설명하는 수율 차이 (상위 5개)\n" + md(pair.head(5).round(3).to_frame("설명 비율")) + "\n")
    s.append(f"### 효과를 더하는 모델 vs 짝 효과까지 넣은 모델 (학습 데이터 안)\n- 시약 효과만 더함: {main:.3f}\n"
             f"- 시약 짝 효과까지: {withpairs:.3f}\n- 짝 효과가 추가로 설명하는 비율: {withpairs - main:.3f}\n")
    return s, one


def main():
    FIG.mkdir(exist_ok=True)
    bh, sm = load_bh(), load_sm()
    s1, one_bh = section("Buchwald–Hartwig", bh, BH_SLOTS)
    s2, _ = section("Suzuki–Miyaura", sm, SM_SLOTS)

    sub = bh.groupby(["arene", "halogen"]).y.agg(["mean", "max"]).round(1).unstack("halogen")
    sub.columns = [f"{h} {k}" for k, h in sub.columns]
    hal = bh.groupby("halogen").y.mean().round(1)
    best = bh.loc[bh.groupby("aryl_halide").y.idxmax()]
    same_best = best.groupby(["ligand", "base"]).size().sort_values(ascending=False)
    lig_base = bh.pivot_table(index="ligand", columns="base", values="y", aggfunc="mean").round(1)
    lig_base.index = [f"리간드 {i + 1}" for i in range(len(lig_base))]
    lig_base.columns = [f"염기 {i + 1}" for i in range(lig_base.shape[1])]
    s1 += [f"### 기질 (방향족 고리 × 할로젠): 평균 / 최고 수율\n{md(sub)}\n",
           f"할로젠별 평균 수율: {hal.to_dict()} (반응성 순서 Cl < Br < I와 일치하는지 확인)\n",
           f"### 기질별 최고 수율 조건의 (리간드, 염기)\n기질 15종 중 같은 (리간드, 염기)가 최고였던 횟수: {same_best.values.tolist()} "
           "(한두 조합에 몰리면 '조건 평균' 기준선이 강할 것)\n",
           f"### 리간드 × 염기 평균 수율\n{md(lig_base)}\n"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for a, (d, name, col) in zip(ax[0], ((bh, "Buchwald–Hartwig", BLUE), (sm, "Suzuki–Miyaura", ORANGE))):
        a.hist(d.y, bins=40, color=col)
        a.set(title=f"(a) {name} 수율 분포 ({len(d)}건)" if col == BLUE else f"(b) {name} 수율 분포 ({len(d)}건)", xlabel="수율 (%)", ylabel="반응 수")
    t = one_bh[::-1]
    ax[1, 0].barh(t.index, t.values, color=GRAY, height=0.6)
    for i, v in enumerate(t.values):
        ax[1, 0].text(v, i, f" {v:.2f}", va="center", fontsize=9, color=INK)
    ax[1, 0].set(title="(c) BH: 칸 하나로 설명되는 수율 차이 비율", xlim=(0, 1))
    im = ax[1, 1].imshow(lig_base.to_numpy(), cmap="Blues")
    ax[1, 1].set_xticks(range(lig_base.shape[1]), lig_base.columns)
    ax[1, 1].set_yticks(range(len(lig_base)), lig_base.index)
    for i in range(lig_base.shape[0]):
        for j in range(lig_base.shape[1]):
            v = lig_base.iat[i, j]
            ax[1, 1].text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > lig_base.to_numpy().mean() else INK)
    ax[1, 1].set(title="(d) BH: 리간드 × 염기 평균 수율 (%)")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.8)
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_summary.png", dpi=110)
    plt.close(fig)

    OUT.write_text("# 반응 수율 EDA (generated by reaction/eda.py)\n\n그림: [figures/eda_summary.png](figures/eda_summary.png). "
                   "설명 비율은 학습 데이터 안에서 계산한 값이라 예측 성능이 아닙니다.\n\n" + "\n".join(s1 + s2), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
