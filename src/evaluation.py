"""Shared evaluation engine for experiments 1-3: data loading, combination keys, the method set (tuned inside the
training data only), candidate eligibility, combination scoring and ranking metrics.
Everything an experiment needs to stay comparable with the others lives here; experiment scripts only decide which
rows are training and which are evaluation, and how to report."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import exp1 as E
from config import v2 as C
from models.v2.model_v2 import ALL, Model, Prep, clean, combo_key, make_features

PROC = C.PROCESS + C.CORRECTION  # "공정+보정": 7 process + 2 correction variables (correction pinned at prediction)
SHARE, LOW_PCE, N_ENS, K_NN, CFG_MIN = 0.10, 5.0, 10, 5, 8
CFG = ["arch", "etl", "htl", "backcontact"]
PARTS = ["solvent", "dmso", "antisolvent", "additive", "solvent_annealing", "temp", "time"]  # order used in combo_key
DMSO_ORDER = {"0": 0, "0-0.35": 1, "0.35-0.9": 2, "0.9-1": 3}
ALPHAS = [0.1, 1, 10, 100]
NAME = {"lgbm_full": "LightGBM (전체)", "lgbm_proc": "LightGBM (공정+보정)", "ridge_full": "선형 회귀 (전체)",
        "ridge_proc": "선형 회귀 (공정+보정)", "knn": "비슷한 조합 평균 (kNN)", "lgbm_lowcut": "LightGBM 저효율 제외 (전체)",
        "ens_mean": "LightGBM 10개 평균 (전체)", "ens_bonus": "LightGBM 10개 + 보너스 (전체)"}
REGS = ["lgbm_full", "lgbm_proc", "ridge_full", "ridge_proc", "lgbm_lowcut"]
PAIRS = [("lgbm_proc", "ridge_proc", "같은 공정 정보: 비선형 모델이 필요한가"),
         ("lgbm_proc", "knn", "같은 공정 정보: 모델이 비슷한 조합 평균보다 나은가"),
         ("lgbm_full", "lgbm_proc", "소자 조건 정보가 도움이 되는가 (LightGBM)"),
         ("ridge_full", "ridge_proc", "소자 조건 정보가 도움이 되는가 (선형 회귀)"),
         ("lgbm_full", "ridge_full", "같은 전체 정보: 비선형 모델이 필요한가"),
         ("lgbm_lowcut", "lgbm_full", "저효율 사례 제외가 도움이 되는가"),
         ("ens_mean", "lgbm_full", "모델 10개 평균이 도움이 되는가"),
         ("ens_bonus", "ens_mean", "불확실성 보너스가 도움이 되는가"),
         ("lgbm_full", "knn", "실용 기준선 (정보량 다름)")]


# ---------------- data and keys ----------------
def load_data(max_year):
    """Cleaned rows up to max_year with combination key and a stable row id. Later years are never loaded."""
    F = make_features(clean()[0])
    F["key"] = combo_key(F)
    F = F[F.year <= max_year].reset_index(drop=True)
    assert F.year.max() <= max_year
    F["rid"] = np.arange(len(F))
    return F


def tie_key(keys):
    """Fixed per-combination tie-break."""
    return np.array([int(hashlib.sha1(k.encode()).hexdigest()[:12], 16) for k in keys])


def parts(keys):
    p = pd.Series(keys, index=keys).str.split(" / ", expand=True)
    p.columns = PARTS
    return p.astype(object)  # plain numpy object arrays broadcast; pandas string arrays don't


def pin(X, year):
    """Model inputs for ranking: real rows, correction variables pinned (year, scan Reverse)."""
    return X.assign(year=int(year), scan_direction="Reversed")


# ---------------- methods ----------------
class RidgeModel:
    def __init__(self, alpha, features=ALL):
        self.alpha, self.features = alpha, features

    def _enc(self, X):
        t = self.prep.transform(X)  # categoricals use the train-fitted category lists, so columns stay fixed
        return pd.get_dummies(t, columns=[c for c in t if isinstance(t[c].dtype, pd.CategoricalDtype)], dtype=float)

    def fit(self, X, y):
        self.prep = Prep(self.features, impute=True).fit(X)  # median fill + missing flags
        Z = self._enc(X)
        self.cols = Z.columns
        self.num = [c for c in Z if c in C.NUMERIC]  # standardize numeric inputs only; one-hot and flags stay 0/1
        self.mu, self.sd = Z[self.num].mean(), Z[self.num].std().replace(0, 1)
        self.m = Ridge(alpha=self.alpha).fit(self._scale(Z), y)
        return self

    def _scale(self, Z):
        Z = Z.reindex(columns=self.cols, fill_value=0.0)
        Z[self.num] = (Z[self.num] - self.mu) / self.sd
        return Z.values

    def predict(self, X):
        return self.m.predict(self._scale(self._enc(X)))


def ridge(train, features=ALL, groups=None, folds=C.CV_FOLDS):
    """Alpha chosen by grouped CV inside `train`. Inside a paper bootstrap pass the ORIGINAL paper id as groups."""
    groups = train["doi"] if groups is None else groups
    mse = {}
    for a in ALPHAS:
        err = []
        for i, j in GroupKFold(folds).split(train, groups=groups):
            p = RidgeModel(a, features).fit(train.iloc[i], train.pce.iloc[i]).predict(train.iloc[j])
            err.append(np.mean((p - train.pce.iloc[j].values) ** 2))
        mse[a] = np.mean(err)
    alpha = min(mse, key=mse.get)
    return RidgeModel(alpha, features).fit(train, train.pce), alpha


def tune_lgbm(train, feats, groups=None, folds=C.CV_FOLDS):
    """LightGBM config + missing-value handling chosen by grouped CV (MAE) inside `train` only."""
    groups = train["doi"] if groups is None else groups
    cv = list(GroupKFold(folds).split(train, groups=groups))
    best, best_mae = None, np.inf
    for impute in (True, False):
        for params in C.GRID["lgbm"]:
            err = [np.mean(np.abs(Model("lgbm", params, impute, feats).fit(train.iloc[a], train.pce.iloc[a])
                                  .predict(train.iloc[b]) - train.pce.iloc[b].values)) for a, b in cv]
            if np.mean(err) < best_mae:
                best, best_mae = (params, impute), np.mean(err)
    return best


def knn_score(keys, train, exclude_self=False):
    """Mean PCE of the K nearest training combinations by process distance (scaled with the training range).
    exclude_self: never use the candidate's own combination (needed when it can be in the training rows)."""
    known = train.dropna(subset=["key"]).groupby("key").pce.mean()
    pc, pk = parts(pd.Index(keys)), parts(known.index)
    D = np.zeros((len(pc), len(pk)))
    for v in ("solvent", "antisolvent", "additive", "solvent_annealing"):
        D += pc[v].to_numpy()[:, None] != pk[v].to_numpy()[None, :]
    D += np.abs(pc.dmso.map(DMSO_ORDER).to_numpy(float)[:, None] - pk.dmso.map(DMSO_ORDER).to_numpy(float)[None, :]) / 3
    for v in ("temp", "time"):
        x = pk[v].astype(float)
        D += np.abs(pc[v].to_numpy(float)[:, None] - x.to_numpy()[None, :]) / max(x.max() - x.min(), 1)
    if exclude_self:
        D[pc.index.to_numpy(object)[:, None] == pk.index.to_numpy(object)[None, :]] = np.inf
    ktie = tie_key(pk.index)
    return pd.Series([known.values[np.lexsort((ktie, D[i]))[:K_NN]].mean() for i in range(len(pc))], index=pc.index)


def fit_methods(train, X, keys, rng, groups_col="doi", folds=C.CV_FOLDS, exclude_self=False):
    """Train the whole method set on `train` (tuned inside it) and predict rows X once.
    Returns (row predictions incl. 10 ensemble members, kNN score per key, chosen settings)."""
    g = train[groups_col]
    (pf, imf), (pp, imp) = tune_lgbm(train, ALL, g, folds), tune_lgbm(train, PROC, g, folds)
    rf, af = ridge(train, ALL, g, folds)
    rp, ap = ridge(train, PROC, g, folds)
    t = train[train.pce >= LOW_PCE]
    pred = {"lgbm_full": Model("lgbm", pf, imf, ALL).fit(train, train.pce).predict(X),
            "lgbm_proc": Model("lgbm", pp, imp, PROC).fit(train, train.pce).predict(X),
            "ridge_full": rf.predict(X), "ridge_proc": rp.predict(X),
            "lgbm_lowcut": Model("lgbm", pf, imf, ALL).fit(t, t.pce).predict(X)}
    groups = train.groupby(groups_col).indices  # ensemble bootstrap over (original) papers
    papers = np.array(list(groups))
    for i in range(N_ENS):
        b = train.iloc[np.concatenate([groups[p] for p in rng.choice(papers, len(papers), replace=True)])]
        pred[f"ens{i}"] = Model("lgbm", pf, imf, ALL).fit(b, b.pce).predict(X)
    chosen = {"lgbm_full": {**pf, "impute": imf}, "lgbm_proc": {**pp, "impute": imp}, "ridge_full": af, "ridge_proc": ap}
    return pd.DataFrame(pred, index=X.index), knn_score(keys, train, exclude_self), chosen


# ---------------- candidates and metrics ----------------
def eligible(R):
    """Candidate keys: >= 3 distinct original devices and >= 2 distinct original papers (copies counted once)."""
    u = R.drop_duplicates("rid")
    g = u.groupby("key").agg(n=("rid", "size"), p=("orig", "nunique"))
    return g[(g.n >= E.MIN_DEVICES) & (g.p >= E.MIN_PAPERS)].index


def combo_table(R, keys):
    """Per candidate: actual device mean, paper mean, and every method's score (mean of row predictions)."""
    S = R[R.key.isin(keys)]
    cols = REGS + [f"ens{i}" for i in range(N_ENS)]
    T = S.groupby("key")[cols + ["pce"]].mean()
    T["pce_paper"] = S.groupby(["key", "doi"]).pce.mean().groupby(level="key").mean()
    ens = T[[f"ens{i}" for i in range(N_ENS)]]
    T["ens_mean"], T["ens_bonus"] = ens.mean(axis=1), ens.mean(axis=1) + ens.std(axis=1)
    return T


def metrics(T, knn, answers_from=None):
    """Spearman (device / paper mean), top-10% hit and gain for every method on candidate table T.
    answers_from: optional set of answer keys (default: top 10% of T by device mean)."""
    T = T.assign(knn=knn.reindex(T.index).values)
    k = max(1, int(round(len(T) * SHARE)))
    tie = pd.Series(tie_key(T.index), index=T.index)
    ans = set(T.pce.nlargest(k).index) if answers_from is None else answers_from
    ans_paper, best = set(T.pce_paper.nlargest(k).index), T.pce.idxmax()
    out = {}
    for m in NAME:
        order = pd.DataFrame({"s": T[m], "t": tie}).sort_values(["s", "t"], ascending=[False, True]).index
        top = order[:k]
        out[m] = {"spearman_dev": spearmanr(T[m], T.pce).statistic, "spearman_paper": spearmanr(T[m], T.pce_paper).statistic,
                  "hit_dev": top.isin(ans).sum() / max(len(ans), 1), "gain_dev": T.pce.loc[top].mean() - T.pce.mean(),
                  "hit_paper": top.isin(ans_paper).sum() / k,
                  "best_pct": 1 - order.get_loc(best) / max(len(T) - 1, 1)}  # 1 = best actual combination ranked first
    return out, len(T)


def config_spearman(R, cfg_keys):
    """Fixed device configuration: rank combos inside each configuration with the configuration's own devices."""
    res = {m: [] for m in NAME if m != "knn"}
    weights = []
    for cfg, keys in cfg_keys.items():
        S = R[(R.cfg == cfg) & R.key.isin(keys)]
        T = combo_table(S, keys)
        if len(T) < CFG_MIN:
            continue
        weights.append(len(T))
        for m in res:
            res[m].append(spearmanr(T[m], T.pce).statistic)
    if not weights:
        return {}
    w = np.array(weights, float)
    return {m: float(np.nansum(np.array(v) * w) / w[~np.isnan(v)].sum()) for m, v in res.items()} | {"n_groups": len(w)}


def config_label(ev):
    return ev[CFG].astype(object).fillna("missing").astype(str).agg(" | ".join, axis=1)


def resample_papers(base, rng, col="orig"):
    """Papers drawn with replacement; a paper drawn twice becomes two papers (doi gets a suffix, orig is kept)."""
    groups = base.groupby(col).indices
    papers = np.array(list(groups))
    seen, parts_ = {}, []
    for p in rng.choice(papers, len(papers), replace=True):
        seen[p] = seen.get(p, 0) + 1
        parts_.append(base.iloc[groups[p]].assign(doi=f"{p}#{seen[p]}"))
    return pd.concat(parts_, ignore_index=True)


def paired(pooled, metrics_, point, pairs=PAIRS):
    """Paired differences (same resamples) for every pre-set comparison. pooled: rows rep x method; point: method x metric."""
    W = pooled.pivot(index="rep", columns="method", values=metrics_)
    rows = []
    for a, b, q in pairs:
        for mt in metrics_:
            d = (W[mt][a] - W[mt][b]).dropna()
            if d.empty:
                continue
            lo, hi = d.quantile(0.025), d.quantile(0.975)
            rows.append({"question": q, "comparison": f"{NAME[a]} − {NAME[b]}", "metric": mt,
                         "point": point.loc[a, mt] - point.loc[b, mt], "2.5%": lo, "97.5%": hi,
                         "higher": (d > 0).mean(), "lower": (d < 0).mean(),
                         "judgement": "우열 판단 증거 부족" if lo <= 0 <= hi else ("앞이 높음" if lo > 0 else "뒤가 높음")})
    return pd.DataFrame(rows)


def fmt(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"
