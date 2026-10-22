"""Presentation figures for the reaction-yield study (reaction/README.md).
Numbers for the result charts are copied from results_model2.md, results_model3.md and results_analysis.md;
the SHAP beeswarms are recomputed (LightGBM on all data, same settings as analysis.py).
Usage: python reaction/figures_deck.py   (-> reaction/figures/deck/*.png)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analysis import shap_slots  # noqa: E402
from model2 import desc_bh, desc_sm, knowledge_bh, knowledge_sm  # noqa: E402
from prepare import load_bh, load_sm  # noqa: E402

OUT = ROOT / "figures" / "deck"
BG, INK, INK2, MUTED, GRID = "#fafaf7", "#1b2430", "#52514e", "#898781", "#e1e0d9"
SIMPLE, RIDGE, ML = "#9a9893", "#86b6ef", "#2a78d6"
ORANGE = "#eb6834"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 15,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left"})

BH_LIGAND = {"CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC": "tBuBrettPhos",
             "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2": "tBuXPhos",
             "CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2": "XPhos"}
BH_BASE = {"CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG", "CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et", "CN1CCCN2C1=NCCC2": "MTBD"}
ARENE = {"*c1ccc(OC)cc1": "4-메톡시페닐", "*c1ccc(CC)cc1": "4-에틸페닐", "*c1ccc(C(F)(F)F)cc1": "4-CF₃-페닐",
         "*c1cccnc1": "3-피리딜", "*c1ccccn1": "2-피리딜"}
# what the top SHAP features mean (fingerprint bits mapped to fragments with RDKit bitInfo)
LABEL = {"aryl_halide:fp49": "출발물질 · 고리의 C–Cl / C–Br", "aryl_halide:fp46": "출발물질 · 벤젠의 C–Cl",
         "aryl_halide:MolWt": "출발물질 · 분자량 (I > Br > Cl)", "aryl_halide:fp2": "출발물질 · 메톡시(OMe)",
         "aryl_halide:fp1": "출발물질 · CF₃ 고리 / 3-Cl-피리딘", "aryl_halide:fp63": "출발물질 · 4번 치환기 / 피리딘 N",
         "aryl_halide:fp7": "출발물질 · 에틸 / 2-Cl-피리딘",
         "base:MolWt": "염기 · 분자량 (염기 종류)", "ligand:fp2": "리간드 · P-사이클로헥실 (XPhos)",
         "ligand:MolLogP": "리간드 · 소수성 (리간드 종류)", "ligand:MolWt": "리간드 · 분자량",
         "additive:fp6": "첨가제 · 아이소옥사졸 C=N–O", "additive:MolLogP": "첨가제 · 소수성",
         "additive:fp63": "첨가제 · 4번 자리 치환 / 벤조 고리", "additive:fp35": "첨가제 · 5번 자리 치환 고리",
         "nucleophile:TPSA": "친핵체 · 극성 표면적 (붕소 종류)", "electrophile:MolWt": "친전자체 · 분자량 (I > OTf > Br > Cl)",
         "solvent:MolWt": "용매 · 분자량 (용매 종류)", "electrophile:fp12": "친전자체 · C–I",
         "solvent:TPSA": "용매 · 극성 표면적", "ligand:fp40": "리간드 · 두 자리 골격 (Xantphos, dppf)",
         "nucleophile:MolLogP": "친핵체 · 소수성", "K:base_pka": "화학 지식 · 염기 pKa", "electrophile:fp5": "친전자체 · C–Br / OTf",
         "ligand:fp6": "리간드 · P-사이클로헥실 (SPhos, XPhos)", "K:cone": "화학 지식 · 리간드 원뿔각"}


def style(ax, grid="x"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_substrates(bh):
    t = bh.groupby([bh.arene.map(ARENE), "halogen"]).y.mean().unstack()[["Cl", "Br", "I"]].reindex(list(ARENE.values()))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(t.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef4fc", "#2a78d6", "#0d366b"]),
              vmin=0, vmax=65, aspect="auto")
    for (i, j), v in np.ndenumerate(t.values):
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=17, fontweight="bold", color="white" if v > 30 else INK)
    ax.set(xticks=range(3), xticklabels=["C–Cl", "C–Br", "C–I"], yticks=range(5), yticklabels=t.index)
    ax.tick_params(length=0, labelsize=16)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "substrates")


def fig_interaction():
    # in-sample R², results_eda.md
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(2)
    ax.bar(x - 0.19, [71, 51], 0.36, color=SIMPLE, label="시약 효과를 더하기만")
    ax.bar(x + 0.19, [90, 78], 0.36, color=ML, label="시약 짝 효과까지")
    for i, (a, b) in enumerate([(71, 90), (51, 78)]):
        ax.text(i - 0.19, a + 1.5, f"{a}%", ha="center", fontsize=16, color=INK2)
        ax.text(i + 0.19, b + 1.5, f"{b}%", ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.set(xticks=x, xticklabels=["Buchwald–Hartwig", "Suzuki–Miyaura"], ylim=(0, 100), ylabel="설명되는 수율 차이 (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=14)
    style(ax, "y")
    save(fig, "interaction")


def fig_main():
    # LightGBM·F1 vs best simple baseline, MAE with 95% CI: results_model2.md, results_model3.md
    rows = [("BH 무작위", 6.77, (6.56, 6.97), 3.95, (3.83, 4.08)),
            ("SM 무작위", 10.79, (10.57, 11.03), 6.80, (6.61, 7.00)),
            ("BH 처음 보는 출발물질", 18.05, (14.27, 21.89), 11.82, (8.97, 15.36)),
            ("BH 처음 보는 고리", 19.35, (15.59, 23.22), 15.21, (12.40, 18.46)),
            ("SM 처음 보는 반응 자리", 18.70, (15.37, 22.18), 18.54, (16.36, 21.20)),
            ("SM 처음 보는 리간드", 15.04, (12.85, 17.61), 18.22, (14.72, 21.95))]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, (name, s, sci, m, mci) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([s, m], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        for v, ci, c, dy in ((s, sci, SIMPLE, 0.12), (m, mci, ML, -0.12)):
            ax.plot(ci, [y + dy] * 2, color=c, lw=2, alpha=0.6, zorder=2)
            ax.scatter(v, y + dy, s=150, color=c, edgecolor=BG, lw=2, zorder=3)
    ax.axhline(3.5, color="#c3c2b7", lw=1, ls=(0, (4, 4)))
    ax.text(25.5, 4.5, "학습 때 본 물질", ha="right", va="center", color=MUTED, fontsize=14)
    ax.text(25.5, 1.5, "처음 보는 물질", ha="right", va="center", color=MUTED, fontsize=14)
    ax.set(yticks=range(len(rows)), yticklabels=[r[0] for r in rows][::-1], xlim=(0, 26),
           xlabel="예측 오차 MAE (%p, 낮을수록 좋음) · 선 = 95% 신뢰구간")
    ax.scatter([], [], s=150, color=SIMPLE, label="단순 방법 (조건 평균 등)")
    ax.scatter([], [], s=150, color=ML, label="LightGBM + 분자 특성")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, 1.12), fontsize=14)
    ax.tick_params(axis="y", labelsize=15)
    style(ax)
    save(fig, "main_result")


def fig_arene():
    # results_model3.md, per-arene MAE
    t = [("2-피리딜", 24.1, 19.3), ("3-피리딜", 19.7, 18.1), ("4-CF₃-페닐", 14.1, 8.8),
         ("4-메톡시페닐", 18.4, 14.8), ("4-에틸페닐", 20.4, 15.1)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (name, s, m) in enumerate(t[::-1]):
        ax.annotate("", xy=(m + 0.35, i), xytext=(s - 0.35, i), arrowprops=dict(arrowstyle="-|>", color="#86b6ef", lw=2.5))
        ax.scatter(s, i, s=150, color=SIMPLE, edgecolor=BG, lw=2, zorder=3)
        ax.scatter(m, i, s=150, color=ML, edgecolor=BG, lw=2, zorder=3)
        ax.text(m - 0.6, i + 0.28, f"{m:.1f}", ha="center", fontsize=13, color=ML, fontweight="bold")
        ax.text(s + 0.6, i + 0.28, f"{s:.1f}", ha="center", fontsize=13, color=INK2)
    ax.set(yticks=range(5), yticklabels=[r[0] for r in t[::-1]], xlim=(6, 27), ylim=(-0.6, 4.7),
           xlabel="예측 오차 MAE (%p) · 그 고리를 통째로 처음 볼 때")
    ax.scatter([], [], s=150, color=SIMPLE, label="단순 방법")
    ax.scatter([], [], s=150, color=ML, label="LightGBM + 분자 특성")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.45, 1.13), fontsize=13)
    ax.tick_params(axis="y", labelsize=15)
    style(ax)
    save(fig, "arene")


def fig_ablation():
    # results_analysis.md part A
    t = [("BH 처음 보는\n출발물질", 18.05, 15.83, 11.82), ("BH 처음 보는\n고리", 19.35, 17.44, 15.21),
         ("SM 처음 보는\n리간드", 15.04, 21.54, 18.22)]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    for k, (lab, c) in enumerate((("단순 방법 (분자 정보 없음)", SIMPLE), ("선형 회귀 + 분자 특성", RIDGE), ("LightGBM + 분자 특성", ML))):
        v = [r[k + 1] for r in t]
        ax.bar(x + (k - 1) * 0.27, v, 0.25, color=c, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1) * 0.27, vi + 0.4, f"{vi:.1f}", ha="center", fontsize=13, color=INK2)
    ax.set(xticks=x, xticklabels=[r[0] for r in t], ylim=(0, 25), ylabel="예측 오차 MAE (%p)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=13)
    style(ax, "y")
    save(fig, "ablation")


def beeswarm(d, X, name, n=10):
    imp, _, S = shap_slots(d, X)
    top = list(imp.index[:n])
    fig, ax = plt.subplots(figsize=(10, 0.62 * n + 1.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#2a78d6", "#d6d4cc", "#e34948"])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d), min(len(d), 2000), replace=False)
    for i, c in enumerate(top[::-1]):
        s, v = S[c].to_numpy()[idx], X[c].to_numpy(float)[idx]
        lo, hi = np.nanmin(v), np.nanmax(v)
        z = (v - lo) / (hi - lo) if hi > lo else np.full(len(v), 0.5)
        # jitter by local density so the swarm spreads where points pile up
        h, e = np.histogram(s, bins=60)
        dens = h[np.clip(np.digitize(s, e) - 1, 0, 59)] / h.max()
        yj = i + rng.uniform(-1, 1, len(s)) * 0.36 * dens
        ax.scatter(s, yj, c=z, cmap=cmap, vmin=0, vmax=1, s=7, alpha=0.75, lw=0, rasterized=True)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set(yticks=range(n), yticklabels=[LABEL.get(c, c) for c in top[::-1]],
           xlabel="SHAP 값: 이 특성 때문에 예측 수율이 오르내린 정도 (%p)")
    ax.tick_params(axis="y", labelsize=14)
    style(ax)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), ax=ax, pad=0.01, aspect=30, ticks=[0, 1])
    cb.ax.set_yticklabels(["낮음\n(없음)", "높음\n(있음)"], fontsize=12)
    cb.outline.set_visible(False)
    cb.set_label("특성 값", fontsize=13, color=INK2)
    save(fig, f"shap_{name}")


def hbars(ax, t, title, hi=None, lo=None):
    t = t.sort_values()
    cols = [ML if k in (hi or []) else ORANGE if k in (lo or []) else SIMPLE for k in t.index]
    ax.barh(range(len(t)), t.values, color=cols, height=0.62)
    for i, v in enumerate(t.values):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=13, color=INK2)
    ax.set(yticks=range(len(t)), yticklabels=t.index, title=title, xlim=(0, max(t.values) * 1.2))
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks([])
    style(ax)
    ax.grid(False)


def fig_chem_bh(bh):
    lig = bh.groupby(bh.ligand.map(BH_LIGAND).fillna("AdBrettPhos")).y.mean()
    base = bh.groupby(bh.base.map(BH_BASE)).y.mean()
    hal = bh.groupby("halogen").y.mean().rename({"Cl": "C–Cl", "Br": "C–Br", "I": "C–I"})
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1.2, 1]})
    hbars(ax[0], hal, "반응 자리 할로젠", lo=["C–Cl"])
    hbars(ax[1], lig, "리간드", lo=["XPhos"])
    hbars(ax[2], base, "염기", hi=["MTBD"])
    fig.tight_layout(w_pad=3)
    save(fig, "chem_bh")


def fig_chem_sm(sm):
    lg = sm[sm.electrophile.str.match(r"1[a-d]")].groupby("electrophile").y.mean()
    lg.index = lg.index.str.extract(r"6-(\w+)-Q")[0].map({"Cl": "C–Cl", "Br": "C–Br", "OTf": "C–OTf", "I": "C–I"}).to_numpy()
    nu = sm[sm.nucleophile.str.match(r"2[a-c]")].groupby("nucleophile").y.mean()
    nu.index = ["보론산 B(OH)₂", "피나콜 에스터 Bpin", "BF₃K 염"]
    lig = sm.groupby("ligand").y.mean()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.1, 1.2]})
    hbars(ax[0], lg, "반응 자리 (친전자체)", lo=["C–Cl"])
    hbars(ax[1], nu, "붕소 종류 (친핵체)", lo=["BF₃K 염"])
    hbars(ax[2], lig, "리간드", hi=["P(Ph)3"], lo=["Xantphos"])
    fig.tight_layout(w_pad=3)
    save(fig, "chem_sm")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bh, sm = load_bh(), load_sm()
    fig_substrates(bh)
    fig_interaction()
    fig_main()
    fig_arene()
    fig_ablation()
    fig_chem_bh(bh)
    fig_chem_sm(sm)
    beeswarm(bh, desc_bh(bh).join(knowledge_bh(bh)), "bh")
    beeswarm(sm, desc_sm(sm).join(knowledge_sm(sm)), "sm")
    print("saved", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
