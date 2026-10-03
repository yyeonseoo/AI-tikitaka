"""Experiment 3, phase 2 (post-hoc, second use of 2018-2019; plan: experiments/exp3/PLAN_phase2.md, committed first).
Train <= 2017 once, score 2018-2019 combinations, rank agreement (Spearman) with the actual combination mean PCE.
Resamples are drawn exactly as in experiments/exp2/exp2_future.py (same seed stream, same paper order), so the 1,000
resamples are the same as experiment 2's; this is checked against the experiment 2 raw file.
Usage: python experiments/exp3/exp3_phase2.py
Outputs: experiments/exp3/results_phase2.md, bootstrap_phase2_<timestamp>.csv, figures/phase2_summary.png"""
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from experiments.exp1.exp1_bootstrap import tie_key
from experiments.exp1.exp1_bootstrap_baselines import ridge
from experiments.exp1.exp1_hide_top import candidate_inputs, fed_values
from experiments.exp2.exp2_future import PROC, knn_score, lgbm
from models.v2.model_v2 import clean, combo_key, make_features
from src.features import md

OUT = Path(__file__).resolve().parent
EXP2 = ROOT / "experiments" / "exp2"
N_REP, SHARE, MIN_UNSEEN = 1000, 0.10, 10
METHODS = ["lgbm", "ridge", "lgbm_process", "knn"]
NAME = {"lgbm": "LightGBM", "ridge": "선형 회귀", "lgbm_process": "LightGBM, 공정만", "knn": "비슷한 조합 평균 (kNN)"}
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def metrics(cand, scores, tie):
    """Spearman with the actual combination mean, and the top-10% gain (%p) over the candidates' mean."""
    k = max(1, int(round(len(cand) * SHARE)))
    out = {}
    for m, s in scores.items():
        s = s.loc[cand.index]
        order = pd.DataFrame({"s": s.values, "t": tie}, index=cand.index).sort_values(["s", "t"], ascending=[False, True]).index
        out[m] = {"spearman": s.corr(cand.pce, method="spearman"), "gain": cand.pce.loc[order[:k]].mean() - cand.pce.mean()}
    return out


def evaluate(R, pred, knn, seen, rep):
    g = R.groupby("key")
    cand = pd.DataFrame({"n": g.size(), "papers": g.orig.nunique(), "pce": g.pce.mean()})
    cand = cand[(cand.n >= E.MIN_DEVICES) & (cand.papers >= E.MIN_PAPERS)]
    rows_c = R[R.key.isin(cand.index)]
    scores = {m: pd.Series(p[rows_c.ridx.values], index=rows_c.index).groupby(rows_c.key.values).mean()
              for m, p in pred.items()}
    scores["knn"] = knn
    res = []
    for subset, c in [("all", cand), ("unseen", cand[~cand.index.isin(seen)])]:
        if subset == "unseen" and len(c) < MIN_UNSEEN:
            res.append({"rep": rep, "subset": subset, "method": "-", "n_candidates": len(c), "skipped": True})
            continue
        for m, v in metrics(c, scores, tie_key(c.index)).items():
            res.append({"rep": rep, "subset": subset, "method": m, "n_candidates": len(c), "skipped": False, **v})
    return res


def ci(v):
    v = pd.Series(v).dropna()
    return v.mean(), v.quantile(0.025), v.quantile(0.975)


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / "figures").mkdir(exist_ok=True)
    ss_boot, _, _ = np.random.SeedSequence(E.SEED).spawn(3)  # identical to exp2_future.py

    F = make_features(clean()[0])
    F["key"] = combo_key(F)
    F = F[F.year <= 2019].reset_index(drop=True)
    train = F[F.year <= E.CUTOFF_YEAR].reset_index(drop=True)
    ev = F[F.year.between(2018, 2019)].reset_index(drop=True)
    t0 = time.time()
    models = {"lgbm": lgbm().fit(train, train.pce), "ridge": ridge(train)[0],
              "lgbm_process": lgbm(PROC).fit(train, train.pce)}
    X = candidate_inputs(ev)
    fed = fed_values(X)
    assert fed == {v: [str(x)] for v, x in E.PREDICT_AT.items()}, fed
    pred = {m: mod.predict(X) for m, mod in models.items()}
    knn = knn_score(ev.key.dropna().unique(), train)
    seen = set(train.key.dropna())
    print(f"train {len(train)} rows / eval {len(ev)} rows; models fitted in {time.time() - t0:.0f}s; inputs {fed}")

    base = ev.dropna(subset=["key"]).assign(orig=lambda d: d.doi, ridx=lambda d: d.index)
    runs = evaluate(base, pred, knn, seen, -1)
    groups = ev.groupby("doi").indices
    papers = np.array(list(groups))
    rng_b = np.random.default_rng(ss_boot)
    for rep in range(N_REP):
        drawn = rng_b.choice(papers, len(papers), replace=True)
        idx = np.concatenate([groups[p] for p in drawn])
        R = pd.DataFrame({"key": ev.key.values[idx], "pce": ev.pce.values[idx], "orig": ev.doi.values[idx],
                          "ridx": idx}).dropna(subset=["key"]).reset_index(drop=True)
        runs += evaluate(R, pred, knn, seen, rep)
    Rn = pd.DataFrame(runs)
    raw = OUT / f"bootstrap_phase2_{stamp}.csv"
    Rn.to_csv(raw, index=False)

    # same resamples as experiment 2? compare candidate counts per replicate with its raw file
    check = "experiment 2 raw file not found"
    e2 = sorted(EXP2.glob("bootstrap_exp2_*.csv"))
    if e2:
        old = pd.read_csv(e2[-1])
        a = old[(old.answers == "main") & (old.method == "lgbm")].set_index("rep").n_candidates
        b = Rn[(Rn.subset == "all") & (Rn.method == "lgbm")].set_index("rep").n_candidates
        j = pd.concat([a, b], axis=1, keys=["exp2", "exp3"]).dropna()
        check = f"candidate counts identical to experiment 2 in {(j.exp2 == j.exp3).sum()}/{len(j)} replicates"

    B = Rn[(Rn.rep >= 0) & (~Rn.skipped)]
    point = Rn[(Rn.rep == -1) & (~Rn.skipped)].set_index(["subset", "method"])
    rows = []
    for (subset, m), g in B.groupby(["subset", "method"]):
        for mt in ("spearman", "gain"):
            mu, lo, hi = ci(g[mt])
            rows.append({"subset": subset, "method": NAME[m], "metric": mt, "point": point[mt].get((subset, m), np.nan),
                         "boot mean": mu, "2.5%": lo, "97.5%": hi, "R": len(g)})
    S = pd.DataFrame(rows)
    W = B.pivot_table(index=["rep", "subset"], columns="method", values="spearman")
    P = []
    for a, b in [("lgbm", "knn"), ("ridge", "knn"), ("lgbm_process", "knn"), ("ridge", "lgbm"), ("lgbm_process", "lgbm")]:
        for subset, d in (W[a] - W[b]).dropna().groupby(level="subset"):
            mu, lo, hi = ci(d)
            P.append({"subset": subset, "comparison": f"{NAME[a]} − {NAME[b]}", "mean": mu, "2.5%": lo, "97.5%": hi,
                      "higher": (d > 0).mean(), "lower": (d < 0).mean(), "R": len(d),
                      "judgement": "evidence insufficient" if lo <= 0 <= hi else ("higher" if lo > 0 else "lower")})
    P = pd.DataFrame(P)
    skipped = Rn[(Rn.rep >= 0) & (Rn.subset == "unseen") & Rn.skipped].rep.nunique()

    def get(subset, m, mt):
        return S[(S.subset == subset) & (S.method == NAME[m]) & (S.metric == mt)].iloc[0]
    prim = P[(P.subset == "all") & (P.comparison == f"{NAME['lgbm']} − {NAME['knn']}")].iloc[0]
    s1, s2 = get("all", "lgbm", "spearman"), get("all", "lgbm_process", "spearman")
    main_ok, sup1 = prim["2.5%"] > 0, s1["2.5%"] > 0
    verdict = ("모델 순위가 비슷한 레시피 평균보다 실제 효율 순위와 더 잘 맞았다 (사후 분석)." if main_ok else
               "모델 순위는 무작위보다 실제와 맞지만, 비슷한 레시피 평균보다 낫다는 증거는 부족하다." if sup1 else
               "2018~2019년에서는 순위 신호를 확인하지 못했다.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, subset, title in [(ax[0], "all", "전체 후보 146개 (원래 데이터 기준)"), (ax[1], "unseen", "새 조합만 (2017년까지 없던 45개)")]:
        t = S[(S.subset == subset) & (S.metric == "spearman")].set_index("method").reindex([NAME[m] for m in METHODS][::-1])
        a.barh(t.index, t["boot mean"], xerr=[t["boot mean"] - t["2.5%"], t["97.5%"] - t["boot mean"]], capsize=3,
               color=["#4C72B0" if i == NAME["lgbm"] else "#DD8452" if i == NAME["knn"] else "#A0B4D0" for i in t.index])
        a.axvline(0, color="#C44E52", ls="--", lw=1)
        a.set(xlim=(-0.6, 1), xlabel="순위 일치도 (0 = 무작위, 1 = 완벽)", title=title)
    fig.suptitle("실험 3 2단계 (사후 분석): 2017년까지 학습 → 2018~2019년 조합 순위, 논문 재표집 1,000회", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "phase2_summary.png", dpi=120)
    plt.close(fig)

    g4 = get("all", "lgbm", "gain")
    (OUT / "results_phase2.md").write_text(
        "# experiment 3 phase 2 results (generated by experiments/exp3/exp3_phase2.py)\n\n"
        "Post-hoc analysis: second use of 2018-2019. Does not replace experiment 2's verdict. Plan: PLAN_phase2.md.\n\n"
        f"train <= {E.CUTOFF_YEAR}: {len(train)} rows; evaluate 2018-2019: {len(ev)} rows / {ev.doi.nunique()} papers; "
        f"resamples {N_REP}; raw {raw.name}; {check}; unseen subset skipped in {skipped} replicates (< {MIN_UNSEEN} candidates)\n\n"
        f"**primary (LightGBM − kNN Spearman, lower bound > 0): {'MET' if main_ok else 'NOT MET'}** — "
        f"{prim['mean']:+.3f} (95% {prim['2.5%']:+.3f} to {prim['97.5%']:+.3f})\n\n"
        f"supporting 1 (LightGBM Spearman > 0): {'met' if sup1 else 'not met'} — {s1['boot mean']:.3f} ({s1['2.5%']:.3f}–{s1['97.5%']:.3f})\n\n"
        f"supporting 2 (LightGBM process-only Spearman > 0): {'met' if s2['2.5%'] > 0 else 'not met'} — "
        f"{s2['boot mean']:.3f} ({s2['2.5%']:.3f}–{s2['97.5%']:.3f})\n\n"
        f"supporting 4 (LightGBM top-10% gain, %p): {g4['boot mean']:+.2f} ({g4['2.5%']:+.2f} to {g4['97.5%']:+.2f})\n\n"
        f"verdict sentence (pre-set): {verdict}\n\n"
        f"## metrics by method\n{md(S.round(3), index=False)}\n\n## paired differences (Spearman)\n{md(P.round(3), index=False)}\n",
        encoding="utf-8")
    print(check)
    print(f"PRIMARY LightGBM − kNN: {prim['mean']:+.3f} [{prim['2.5%']:+.3f}, {prim['97.5%']:+.3f}] -> {'MET' if main_ok else 'NOT MET'}")
    print(md(S.round(3), index=False))
    print(md(P.round(3), index=False))
    print(f"skipped unseen replicates: {skipped}; verdict: {verdict}")


if __name__ == "__main__":
    main()
