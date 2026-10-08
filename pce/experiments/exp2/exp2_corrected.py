"""Experiments 2 and 3 phase 2, corrected rerun (post-hoc; plan: experiments/PLAN_corrected_reruns.md).
Train every method once on <= 2017 (tuned inside it), rank 2018-2019 combinations, 1,000 paper-level resamples of the
evaluation papers with the models fixed. This is the third use of 2018-2019 data, so it is NOT confirmatory evidence;
the original pre-registered experiment 2 verdict (NOT MET, results_exp2.md) stands.
Usage: python pce/experiments/exp2/exp2_corrected.py [R=1000]
Outputs: experiments/exp2/results_exp2_corrected.md (experiment 2 metrics), experiments/exp3/results_phase2_corrected.md
(experiment 3 phase 2 metrics), experiments/exp2/bootstrap_exp2_corrected_<timestamp>.csv,
experiments/exp2/figures/exp2_corrected_summary.png; progress in experiments/exp2/exp2_progress.log (not committed)."""
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from src.evaluation import (N_ENS, NAME, PAIRS, REGS, combo_table, eligible, fit_methods, fmt, load_data, metrics, paired,
                            pin, resample_papers)
from src.features import md

OUT = Path(__file__).resolve().parent
OUT3 = ROOT / "experiments" / "exp3"
LOG = OUT / "exp2_progress.log"
TRAIN_END, HIGH, NEW_MIN = 2017, 15.0, 10
MAIN = ["spearman_dev", "spearman_paper", "hit_dev", "hit_paper", "gain_dev", "best_pct"]
EXTRA = ["hit_fixed", "spearman_fixed", "spearman_new", "auc_high"]
PAIRS3 = PAIRS + [("ridge_proc", "knn", "같은 공정 정보: 선형 회귀가 비슷한 조합 평균보다 나은가")]
HEADER = ("**사후 수정 재실행:** 2018~2019년 데이터를 세 번째로 쓴 분석이라 확인적 증거가 아닙니다. "
          "원래 사전 등록 실험 2의 판정(미달)은 그대로 유효합니다.\n\n")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def log(text):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%H:%M:%S} {text}\n")


def device_auc(R, knn):
    """High-efficiency (PCE >= 15%) discrimination over every evaluation device, on pinned inputs. kNN has no row score."""
    y = R.pce >= HIGH
    ens = R[[f"ens{i}" for i in range(N_ENS)]]
    s = R[REGS].assign(ens_mean=ens.mean(axis=1), ens_bonus=ens.mean(axis=1) + ens.std(axis=1))
    return {m: roc_auc_score(y, s[m]) if m in s and y.nunique() == 2 else np.nan for m in NAME}


def summary(R, metrics_, point):
    rows = []
    for m in NAME:
        g = R[(R.rep >= 0) & (R.method == m)]
        rows.append({"method": NAME[m], **{mt: fmt(point.loc[m, mt], g[mt].quantile(0.025), g[mt].quantile(0.975)) for mt in metrics_}})
    return pd.DataFrame(rows)


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    F = load_data(2019)  # 2020+ never loaded
    train = F[F.year <= TRAIN_END].reset_index(drop=True)
    ev = F[F.year > TRAIN_END].dropna(subset=["key"]).reset_index(drop=True)
    ss_fit, ss_boot = np.random.SeedSequence([E.SEED, 2]).spawn(2)
    P, knn, chosen = fit_methods(train, pin(ev, TRAIN_END), ev.key.unique(), np.random.default_rng(ss_fit))
    log(f"start {stamp}: trained on {len(train)} rows in {time.time() - t0:.0f}s; chosen {chosen}")
    base = pd.concat([ev[["key", "pce", "doi", "rid"]].assign(orig=ev.doi), P], axis=1)
    fixed, seen = eligible(base), set(train.key.dropna())

    rows, n_new_skipped = [], 0
    for rep in range(-1, n_rep):
        R = base if rep < 0 else resample_papers(base, np.random.default_rng(np.random.SeedSequence([E.SEED, 2, rep]).spawn(1)[0]))
        cand = eligible(R)
        new = cand[~cand.isin(seen)]
        m, n = metrics(combo_table(R, cand), knn)
        mf, nf = metrics(combo_table(R, fixed), knn)
        mn, nn = metrics(combo_table(R, new), knn) if len(new) >= NEW_MIN else ({}, len(new))
        n_new_skipped += len(new) < NEW_MIN and rep >= 0
        auc = device_auc(R, knn)
        for k in NAME:
            rows.append({"rep": rep, "method": k, "n_candidates": n, "n_fixed": nf, "n_new": nn, **m[k],
                         "hit_fixed": mf[k]["hit_dev"], "spearman_fixed": mf[k]["spearman_dev"],
                         "spearman_new": mn.get(k, {}).get("spearman_dev", np.nan), "auc_high": auc[k]})
        if (rep + 1) % 100 == 0:
            log(f"  resample {rep + 1}/{n_rep}")
    R = pd.DataFrame(rows)
    raw = OUT / f"bootstrap_exp2_corrected_{stamp}.csv"
    R.to_csv(raw, index=False)

    allm = MAIN + EXTRA
    point = R[R.rep == -1].set_index("method")[allm]
    pooled = R[R.rep >= 0]
    info = (f"Trained on {len(train)} rows / {train.doi.nunique()} papers (<= 2017); evaluated {len(ev)} rows / "
            f"{ev.doi.nunique()} papers (2018-2019). Candidates on original data: {len(fixed)} ({(~fixed.isin(seen)).sum()} new). "
            f"R={n_rep}; raw {raw.name}; chosen settings {chosen}. Cells: original-data point [2.5%, 97.5%] of the resamples. "
            "Main candidates are re-derived in every resample (as in the original runs); *_fixed = candidates fixed to the "
            f"original data (sensitivity); *_new = combinations absent from <= 2017 (skipped in {n_new_skipped} resamples "
            f"with < {NEW_MIN}); auc_high = device-level PCE >= {HIGH:.0f}% discrimination on pinned inputs.\n\n")

    # experiment 2: main metric hit_dev, method of record LightGBM (전체)
    lo = pooled[pooled.method == "lgbm_full"].hit_dev.quantile(0.025)
    alo = pooled[pooled.method == "lgbm_full"].auc_high.quantile(0.025)
    verdict = (f"LightGBM (전체) 주 지표 하한 {lo:.3f} → 원래 성공 기준(> 0.10) {'충족' if lo > 0.10 else '미달'} "
               f"(사후 수정 재실행이라 판정을 바꾸지 않음). 보강 근거 AUC 하한 {alo:.3f} (> 0.5: {'예' if alo > 0.5 else '아니오'}).")
    P2 = paired(pooled, ["hit_dev", "spearman_dev", "gain_dev", "auc_high"], point)
    (OUT / "results_exp2_corrected.md").write_text(
        "# experiment 2 corrected rerun (generated by experiments/exp2/exp2_corrected.py)\n\n" + HEADER + info
        + f"## verdict line\n{verdict}\n\n## methods\n{md(summary(R, ['hit_dev', 'hit_paper', 'best_pct', 'gain_dev', 'spearman_dev', 'hit_fixed', 'auc_high'], point), index=False)}\n\n"
        f"## paired comparisons\n{md(P2.round(3), index=False)}\n", encoding="utf-8")

    # experiment 3 phase 2: main metric spearman_dev
    P3 = paired(pooled, ["spearman_dev", "spearman_paper", "spearman_fixed", "spearman_new"], point, PAIRS3)
    (OUT3 / "results_phase2_corrected.md").write_text(
        "# experiment 3 phase 2 corrected rerun (generated by experiments/exp2/exp2_corrected.py)\n\n" + HEADER + info
        + f"## methods\n{md(summary(R, ['spearman_dev', 'spearman_paper', 'spearman_fixed', 'spearman_new', 'gain_dev'], point), index=False)}\n\n"
        f"## paired comparisons, main metric (spearman_dev)\n{md(P3[P3.metric == 'spearman_dev'].drop(columns='metric').round(3), index=False)}\n\n"
        f"## paired comparisons, all metrics\n{md(P3.round(3), index=False)}\n", encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for a, mt, rnd, title in ((ax[0], "hit_dev", 0.10, "(a) 실험 2 주 지표: 정답 적중 비율 (무작위 0.10)"),
                              (ax[1], "spearman_dev", 0.0, "(b) 실험 3 2단계 주 지표: 순위 일치도 (무작위 0)")):
        t = pooled.groupby("method")[mt].agg(lo=lambda v: v.quantile(0.025), hi=lambda v: v.quantile(0.975)).reindex(list(NAME)[::-1])
        pt = point[mt].reindex(t.index)
        a.barh([NAME[m] for m in t.index], pt, xerr=[np.clip(pt - t.lo, 0, None), np.clip(t.hi - pt, 0, None)], capsize=3,
               color=["#DD8452" if m == "knn" else "#4C72B0" if "full" in m or m.startswith("ens") else "#A0B4D0" for m in t.index])
        a.axvline(rnd, color="#C44E52", ls="--", lw=1)
        a.set(title=title, xlabel="점 = 원래 데이터, 선 = 재표집 95% 구간")
    fig.suptitle(f"실험 2·3 2단계 사후 수정 재실행: 2017년까지 학습 → 2018~2019년 (논문 재표집 {n_rep}회)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "exp2_corrected_summary.png", dpi=120)
    plt.close(fig)
    log(f"finished in {(time.time() - t0) / 60:.1f} min")
    print(verdict)
    print(md(P3[P3.metric == "spearman_dev"][["comparison", "point", "2.5%", "97.5%", "judgement"]].round(3), index=False))


if __name__ == "__main__":
    main()
