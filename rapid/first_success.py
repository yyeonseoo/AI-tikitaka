"""RAPID part 2 (plan: rapid/README.md, "2부 계획"): find the first 4-point result fastest for an unseen amine.
Retrospective simulation on the recorded experiments of each test amine; one experiment at a time; stop at the first 4.
Same amine-level folds as part 1 (test 20%; the rest split into train 60 / validation 20 by amine).
Usage: python rapid/first_success.py   (-> rapid/results_first_success.md, rapid/figures/first_success_curve.png)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from step1 import md  # noqa: E402
from step3 import AMINE, SEED, prepare  # noqa: E402

OUT, FIG = ROOT / "results_first_success.md", ROOT / "figures"
CAP, REPS, N_BOOT, N_START = 48, 20, 1000, 4
WEIGHTS, H_MULT = [5, 10, 20], [0.5, 1, 2]
CONC = ["_rxn_M_inorganic", "_rxn_M_organic", "_rxn_M_acid"]
LEARNED = ["prior", "neighbor", "retrain_bin", "retrain_ord"]
NAME = {"random": "무작위", "spacefill": "고르게 훑기", "prior": "다른 아민 모델 순서", "neighbor": "근처 점수 따라가기",
        "retrain_bin": "재학습 (4점 여부)", "retrain_ord": "재학습 (점수 1~4)"}


def model(kind):
    kw = dict(num_leaves=15, n_estimators=200, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=1)
    return LGBMClassifier(**kw) if kind == "bin" else LGBMRegressor(**kw)


def pool(d, g):
    s = d[d[AMINE] == g].reset_index(drop=True)
    Z = s[CONC].to_numpy()
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    return s, D, s._out_crystalscore.to_numpy().astype(int)


def run(order_fn, score):
    """Pick experiments one by one with order_fn(tested list) -> next index; return experiments to the first 4 (or CAP+1)."""
    tested = []
    while len(tested) < min(CAP, len(score)):
        i = order_fn(tested)
        tested.append(i)
        if score[i] == 4:
            return len(tested)
    return CAP + 1


def farthest(D, tested, untested):
    return untested[np.argmax(D[np.ix_(untested, tested)].min(1))] if tested else untested[0]


def s_random(score, rng):
    perm = rng.permutation(len(score))
    hit = np.flatnonzero(score[perm][:CAP] == 4)
    return hit[0] + 1 if len(hit) else CAP + 1


def s_spacefill(D, score, start):
    def nxt(tested):
        un = np.setdiff1d(np.arange(len(score)), tested)
        return start if not tested else farthest(D, tested, un)
    return run(nxt, score)


def s_prior(p, score):
    order = np.argsort(-p, kind="stable")
    hit = np.flatnonzero(score[order][:CAP] == 4)
    return hit[0] + 1 if len(hit) else CAP + 1


def s_neighbor(D, score, start, h):
    def nxt(tested):
        un = np.setdiff1d(np.arange(len(score)), tested)
        if len(tested) < N_START:
            return start if not tested else farthest(D, tested, un)
        W = np.exp(-D[np.ix_(un, tested)] ** 2 / (2 * h ** 2))
        est = (W * score[tested]).sum(1) / (W.sum(1) + 1e-12)
        best = np.flatnonzero(est >= est.max() - 1e-9)
        return farthest(D, tested, un[best]) if len(best) > 1 else un[best[0]]
    return run(nxt, score)


def s_retrain(train, s, X, score, kind, w):
    target = (train._out_crystalscore == 4).astype(int) if kind == "bin" else train._out_crystalscore
    base = model(kind).fit(train[X], target)

    def predict(m):
        return m.predict_proba(s[X])[:, 1] if kind == "bin" else m.predict(s[X])
    state = {"p": predict(base)}

    def nxt(tested):
        if tested:
            sh = s.iloc[tested]
            ty = (sh._out_crystalscore == 4).astype(int) if kind == "bin" else sh._out_crystalscore
            m = model(kind).fit(pd.concat([train[X], sh[X]]), np.concatenate([target, ty]),
                                sample_weight=np.concatenate([np.ones(len(train)), np.full(len(tested), w)]))
            state["p"] = predict(m)
        p = state["p"].copy()
        p[tested] = -np.inf
        return int(np.argmax(p))
    return run(nxt, score)


def nn_median(D):
    E = D + np.diag(np.full(len(D), np.inf))
    return np.median(E.min(1))


def evaluate_amine(d, g, train, X, fold, settings):
    """All strategies for one amine. settings: dict with 'h' (multiplier) and weights per retrain kind; None -> all grid."""
    s, D, score = pool(d, g)
    h0 = nn_median(D)
    starts = [np.random.default_rng([SEED, 11, fold, r, len(s)]).integers(len(s)) for r in range(REPS)]
    res = {"amine": g}
    res["random"] = [s_random(score, np.random.default_rng([SEED, 13, fold, r, len(s)])) for r in range(REPS)]
    res["spacefill"] = [s_spacefill(D, score, st) for st in starts]
    prior = model("bin").fit(train[X], (train._out_crystalscore == 4).astype(int)).predict_proba(s[X])[:, 1]
    res["prior"] = [s_prior(prior, score)]
    hs = H_MULT if settings is None else [settings["h"]]
    for hm in hs:
        res[f"neighbor_h{hm}"] = [s_neighbor(D, score, st, hm * h0) for st in starts]
    for kind in ("bin", "ord"):
        ws = WEIGHTS if settings is None else [settings[kind]]
        for w in ws:
            res[f"retrain_{kind}_w{w}"] = [s_retrain(train, s, X, score, kind, w)]
    return res


def main():
    FIG.mkdir(exist_ok=True)
    d, feat, X = prepare()
    amines = np.array(sorted(d[AMINE].unique()))
    has4 = d.groupby(AMINE).y.max() > 0
    rng = np.random.default_rng(SEED)
    outer = {g: i % 5 for i, g in enumerate(rng.permutation(amines))}  # same folds as part 1
    test_rows, folds = [], []
    for f in range(5):
        test_a = [g for g in amines if outer[g] == f]
        rest_a = rng.permutation([g for g in amines if outer[g] != f])
        n_val = int(round(len(rest_a) * 0.25))
        val_a, tr_a = rest_a[:n_val], rest_a[n_val:]
        # tune on validation amines (models trained on train amines only)
        tr = d[d[AMINE].isin(tr_a)]
        V = pd.DataFrame(Parallel(n_jobs=-1)(delayed(evaluate_amine)(d, g, tr, X, f, None) for g in val_a if has4[g]))
        vm = V.drop(columns="amine").map(np.mean).mean()
        h = min(H_MULT, key=lambda m: vm[f"neighbor_h{m}"])
        wb = min(WEIGHTS, key=lambda w: vm[f"retrain_bin_w{w}"])
        wo = min(WEIGHTS, key=lambda w: vm[f"retrain_ord_w{w}"])
        val_score = {"prior": vm["prior"], "neighbor": vm[f"neighbor_h{h}"], "retrain_bin": vm[f"retrain_bin_w{wb}"],
                     "retrain_ord": vm[f"retrain_ord_w{wo}"]}
        chosen = min(LEARNED, key=val_score.get)
        folds.append({"fold": f, "val amines (4점 있음)": len(V), "h배수": h, "w (4점 여부)": wb, "w (점수)": wo,
                      **{f"val {NAME[k]}": round(v, 1) for k, v in val_score.items()}, "선택 전략": NAME[chosen]})
        # test amines: models trained on train + validation amines
        trv = d[d[AMINE].isin(np.concatenate([tr_a, val_a]))]
        T = Parallel(n_jobs=-1)(delayed(evaluate_amine)(d, g, trv, X, f, {"h": h, "bin": wb, "ord": wo})
                                for g in test_a if has4[g])
        for r in T:
            row = {"amine": r["amine"], "fold": f, "random": r["random"], "spacefill": r["spacefill"], "prior": r["prior"],
                   "neighbor": r[f"neighbor_h{h}"], "retrain_bin": r[f"retrain_bin_w{wb}"], "retrain_ord": r[f"retrain_ord_w{wo}"]}
            row["chosen"] = row[chosen]
            test_rows.append(row)
    R = pd.DataFrame(test_rows).set_index("amine")
    strategies = list(NAME) + ["chosen"]
    M = R[strategies].map(np.mean)  # per amine: mean experiments to the first 4 over repeats
    ib = [np.random.default_rng([SEED, 5, b]).integers(0, len(M), len(M)) for b in range(N_BOOT)]

    def ci(v):
        v = np.asarray(v, float)
        b = np.array([v[i].mean() for i in ib])
        return v.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)

    def within(col, t):
        return R[col].map(lambda xs: np.mean(np.array(xs) <= t))
    tab = []
    for k in strategies:
        m, lo, hi = ci(M[k])
        tab.append({"전략": NAME.get(k, "선택된 학습 전략 (주)"), "첫 4점까지 실험 수": f"{m:.1f} [{lo:.1f}, {hi:.1f}]",
                    "8번 안 발견": f"{within(k, 8).mean():.0%}", "16번 안 발견": f"{within(k, 16).mean():.0%}",
                    "48번까지 못 찾음": f"{R[k].map(lambda xs: np.mean(np.array(xs) > CAP)).mean():.0%}"})

    def diff(a, b):
        m, lo, hi = ci(M[a] - M[b])
        j = "앞이 빠름" if hi < 0 else ("뒤가 빠름" if lo > 0 else "우열 판단 증거 부족")
        return {"비교": f"{NAME.get(a, '선택된 학습 전략')} − {NAME.get(b, b)}", "차이 (실험 수)": f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                "판단": j, "hi": hi}
    P = [diff("chosen", "random"), diff("chosen", "spacefill"), diff("retrain_ord", "retrain_bin")]
    P += [diff(k, "random") for k in LEARNED] + [diff("neighbor", "retrain_bin"), diff("neighbor", "prior")]
    main_ok = P[0]["hi"] < 0 and P[1]["hi"] < 0
    ord_ok = P[2]["hi"] < 0
    P = pd.DataFrame(P).drop(columns="hi")

    # curve: share of amines whose first 4 was found within t experiments
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"random": "#9a9893", "spacefill": "#52514e", "prior": "#eda100", "neighbor": "#1baf7a",
              "retrain_bin": "#2a78d6", "retrain_ord": "#eb6834"}
    ts = np.arange(1, CAP + 1)
    for k, c in colors.items():
        ax.plot(ts, [within(k, t).mean() for t in ts], color=c, lw=2, label=NAME[k])
    ax.set(xlabel="실험 수", ylabel="첫 4점을 찾은 비율 (아민 평균)", ylim=(0, 1), xlim=(1, CAP),
           title="처음 보는 아민: 실험 수에 따라 첫 큰 결정(4점)을 찾은 비율")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e4e0", lw=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "first_success_curve.png", dpi=110)
    plt.close(fig)

    text = ("# RAPID part 2 results (generated by rapid/first_success.py)\n\n"
            f"Test amines with at least one 4-point experiment: {len(M)}. Experiments to the first 4 counts the 4-point "
            f"experiment itself; no 4 within {CAP} experiments counts as {CAP + 1}. Cells: mean over amines "
            f"[95% amine bootstrap, {N_BOOT}]. Random-start strategies averaged over {REPS} repeats per amine.\n\n"
            f"## verdict\n- 주 판정 (선택된 학습 전략이 무작위·고르게 훑기보다 빠름): **{'충족' if main_ok else '미달'}**\n"
            f"- 1~3점의 가치 (재학습 점수 1~4 − 재학습 4점 여부 < 0): **{'충족' if ord_ok else '미달'}**\n\n"
            f"## strategies (test amines)\n{md(pd.DataFrame(tab), index=False)}\n\n## paired comparisons\n{md(P, index=False)}\n\n"
            f"## settings chosen on validation amines\n{md(pd.DataFrame(folds), index=False)}\n\n"
            "그림: [figures/first_success_curve.png](figures/first_success_curve.png)\n")
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
