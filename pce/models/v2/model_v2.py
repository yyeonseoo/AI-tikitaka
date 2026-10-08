"""v2: process + condition + correction variables -> PCE (MAPbI3, one-step spin coating).
Usage: python pce/models/v2/model_v2.py
Prints markdown sections (summarised in reports/model_v2_report.md), saves plots to models/v2/figures/
and the chosen setup to models/v2/best_params.json. Variable choices live in config/v2.py."""
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # repo root, so `src` / `config` import from any cwd
from config import v2 as C
from src.data import load_nomad, mapbi_onestep, split_from_file
from src.features import antisolvent, dmso_frac, first_step, md

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
P = "Perovskite_deposition_"
ID_COLS = ["entry_id", "Ref_ID", "Ref_ID_temp", "Ref_internal_sample_id"]  # unique per row, so not part of "duplicate"
ALL = [v for g in C.GROUPS.values() for v in g]


RESULTS = OUT / "results.md"  # full tables; the console only gets a summary
CACHE = OUT / "cache" / "cv.json"  # finished CV runs, so an interrupted run resumes where it stopped


def out(text="", console=False):
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(str(text) + "\n")
    if console:
        print(text, flush=True)


def num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------- cleaning ----------------
def clean():
    steps = []

    def log(name, d, note=""):
        steps.append((name, len(d), d["Ref_DOI_number"].nunique(), note))

    df = load_nomad()
    log("NOMAD all", df)
    df = mapbi_onestep(df)
    log("MAPbI3 + one-step spin coating", df)
    df = df[df["Ref_extraction_method"] != "LLM"]
    log("drop LLM-extracted rows", df)
    df = df[num(df["JV_light_intensity"]).between(*C.SUN_RANGE)]
    log("drop non-1-sun (indoor) rows", df)
    df = df[num(df["JV_default_PCE"]) > 0]
    log("drop PCE missing or <= 0", df)
    df = df[~df.drop(columns=ID_COLS).duplicated()]
    log("drop exact duplicates (all columns except row IDs)", df)
    df = df.copy()
    ff = num(df["JV_default_FF"])
    pct = ff > 1
    df["JV_default_FF"] = ff.where(~pct, ff / 100)
    log("fix FF entered in % (/100)", df, f"{pct.sum()} rows fixed")
    # PCE (%) = 100 * Voc [V] * Jsc [mA/cm2] * FF / light intensity [mW/cm2]
    calc = 100 * num(df["JV_default_Voc"]) * num(df["JV_default_Jsc"]) * df["JV_default_FF"] / num(df["JV_light_intensity"])
    gap = (calc - num(df["JV_default_PCE"])).abs()
    df = df[~(gap > C.PCE_CONSISTENCY_TOL)]
    log(f"drop abs(100*Voc*Jsc*FF/intensity - PCE) > {C.PCE_CONSISTENCY_TOL}%p", df, f"{gap.isna().sum()} unverifiable rows kept")
    df = df[df["Ref_DOI_number"].notna()]
    log("drop missing DOI (needed for paper split)", df)
    return df.reset_index(drop=True), pd.DataFrame(steps, columns=["step", "rows", "papers", "note"])


def parse_steps(s, how):
    # "65; 100" (stepwise anneal) -> max temperature / total time.
    # If any step can't be read (e.g. "120 | Unknown", multi-layer "50; 100 | 50; 100"), the value is missing:
    # computing from the readable steps only would make an incomplete record look complete.
    def f(v):
        if pd.isna(v):
            return np.nan
        parts = num(pd.Series([x.strip() for x in re.split(r";|>>", str(v))]))
        if parts.isna().any():
            return np.nan
        return parts.max() if how == "max" else parts.sum()
    return s.map(f)


def make_features(df):
    arch = df["Cell_architecture"]
    return pd.DataFrame({
        "doi": df["Ref_DOI_number"],
        "pce": num(df["JV_default_PCE"]),
        "solvent": first_step(df[P + "solvents"]),
        "dmso_frac": [dmso_frac(a, b) for a, b in zip(df[P + "solvents"], df[P + "solvents_mixing_ratios"])],
        "antisolvent": [antisolvent(a, b) for a, b in zip(df[P + "quenching_media"],
                                                          df[P + "quenching_induced_crystallisation"])],
        "anneal_temp": parse_steps(df[P + "thermal_annealing_temperature"], "max"),
        "anneal_time": parse_steps(df[P + "thermal_annealing_time"], "sum"),
        "additive": df["Perovskite_additives_compounds"],  # NaN = not recorded, "Undoped" = explicitly none
        "solvent_annealing": (df[P + "solvent_annealing"].astype(str) == "True").astype(int),
        "arch": arch.where(arch.isin(["nip", "pin"]) | arch.isna(), "other"),
        "backcontact": df["Backcontact_stack_sequence"],
        "flexible": (df["Cell_flexible"].astype(str) == "True").astype(int),
        "etl": df["ETL_stack_sequence"],
        "htl": df["HTL_stack_sequence"],
        "year": pd.to_datetime(df["Ref_publication_date"], errors="coerce", utc=True).dt.year,
        "scan_direction": df["JV_default_PCE_scan_direction"],
    })


# ---------------- model ----------------
class Prep:
    """Fit on train only: top-N categories, numeric medians, which numerics get a missing flag."""

    def __init__(self, features, impute):
        self.features, self.impute = features, impute

    def fit(self, X):
        self.cats = {}
        for c in self.features:
            if c in C.NUMERIC:
                continue
            vc = X[c].dropna().value_counts()
            if c == "additive":  # keep "Undoped" (explicit none) apart from top named additives
                keep = [k for k in vc.index if k != "Undoped"][:C.TOP_N[c]] + ["Undoped"]
            else:
                keep = list(vc.index[:C.TOP_N.get(c, len(vc))])
            self.cats[c] = list(dict.fromkeys(keep + ["other", C.MISSING]))  # raw data can already contain "other"
        nums = [c for c in self.features if c in C.NUMERIC]
        self.medians = X[nums].median()
        self.flag = [c for c in nums if X[c].isna().any()]
        return self

    def transform(self, X):
        out = {}
        for c in self.features:
            if c in self.cats:
                v = X[c].where(X[c].isin(self.cats[c]) | X[c].isna(), "other").fillna(C.MISSING)
                out[c] = pd.Categorical(v, categories=self.cats[c])
            elif self.impute:
                out[c] = X[c].astype(float).fillna(self.medians[c])
                if c in self.flag:
                    out[c + "_missing"] = X[c].isna().astype(int)
            else:
                out[c] = X[c].astype(float)
        return pd.DataFrame(out, index=X.index)


class Model:
    def __init__(self, kind, params, impute, features):
        self.kind, self.params, self.impute, self.features = kind, params, impute, features

    def _matrix(self, Xt):  # RF has no categorical support: integer codes are fine for trees
        if self.kind != "rf":
            return Xt
        return Xt.apply(lambda s: s.cat.codes if isinstance(s.dtype, pd.CategoricalDtype) else s)

    def fit(self, X, y):
        self.prep = Prep(self.features, self.impute).fit(X)
        if self.kind == "rf":
            self.m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=C.SEED, **self.params)
        elif self.kind == "lgbm":
            self.m = LGBMRegressor(random_state=C.SEED, verbose=-1, **self.params)
        else:
            self.m = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=C.SEED, **self.params)
        self.m.fit(self._matrix(self.prep.transform(X)), y)
        return self

    def predict(self, X):
        return self.m.predict(self._matrix(self.prep.transform(X)))


def scores(y, p):
    return mean_absolute_error(y, p), r2_score(y, p)


def cv(kind, params, impute, features, tr):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    # key covers the exact rows (hash of inputs, target and paper) plus every setting that changes the folds or model
    data_hash = hashlib.sha1(pd.util.hash_pandas_object(tr[[*features, "pce", "doi"]].astype(str), index=False)
                             .values.tobytes()).hexdigest()
    key = json.dumps([kind, params, impute, features, data_hash, C.SEED, C.CV_FOLDS, C.TOP_N, C.MISSING])
    if key in cache:
        return np.array(cache[key])
    res = []
    for a, b in GroupKFold(C.CV_FOLDS).split(tr, groups=tr["doi"]):
        m = Model(kind, params, impute, features).fit(tr.iloc[a], tr["pce"].iloc[a])
        res.append(scores(tr["pce"].iloc[b], m.predict(tr.iloc[b])))
    cache[key] = list(np.mean(res, axis=0))
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return np.array(cache[key])


def tune(tr, features):
    rows, n = [], sum(len(g) for g in C.GRID.values()) * 2
    for kind, grid in C.GRID.items():
        for impute in (True, False):
            for params in grid:
                mae, r2 = cv(kind, params, impute, features, tr)
                rows.append({"model": kind, "missing": "median+flag" if impute else "native NaN",
                             "params": params, "cv_MAE": mae, "cv_R2": r2})
                print(f"  tuning {len(rows)}/{n}: {kind} {rows[-1]['missing']} MAE {mae:.3f}", flush=True)
    return pd.DataFrame(rows)


def permutation(model, X, y, features, repeats=5):
    rng = np.random.default_rng(C.SEED)
    base_mae, base_r2 = scores(y, model.predict(X))
    out = []
    for c in features:
        d = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].values)
            mae, r2 = scores(y, model.predict(Xp))
            d.append((mae - base_mae, base_r2 - r2))
        out.append((c, *np.mean(d, axis=0), np.std([x[1] for x in d])))
    return pd.DataFrame(out, columns=["feature", "MAE_increase", "R2_drop", "R2_drop_sd"]).sort_values("R2_drop", ascending=False)


# ---------------- combination bins (experiments 1/2) ----------------
def combo_key(F):
    cats = pd.DataFrame({
        "solvent": F["solvent"].fillna(C.MISSING),
        "dmso": pd.cut(F["dmso_frac"], C.DMSO_BINS, labels=["0", "0-0.35", "0.35-0.9", "0.9-1"]).astype(str),
        "antisolvent": F["antisolvent"],
        "additive": F["additive"].fillna(C.MISSING),
        "solvent_annealing": F["solvent_annealing"].astype(str),
        "temp": (F["anneal_temp"] // C.TEMP_BIN * C.TEMP_BIN).astype("Int64").astype(str),
        "time": (F["anneal_time"] // C.TIME_BIN * C.TIME_BIN).astype("Int64").astype(str),
    })
    ok = F[["dmso_frac", "anneal_temp", "anneal_time"]].notna().all(axis=1)  # numeric process vars must be known
    # pd.cut / Int64 keep NaN through astype(str); rows with NaN are masked by `ok` anyway
    return cats.astype(object).fillna("nan").astype(str).agg(" / ".join, axis=1).where(ok)


def describe(s):
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return f"n={len(s)}, mean {s.mean():.1f}, p10 {q[0.1]:.1f}, p25 {q[0.25]:.1f}, median {q[0.5]:.1f}, p75 {q[0.75]:.1f}, p90 {q[0.9]:.1f}"


def main():
    FIG.mkdir(exist_ok=True)
    RESULTS.write_text("# model_v2 results (generated by models/v2/model_v2.py)\n\n", encoding="utf-8")
    df, steps = clean()
    out("## cleaning\n" + md(steps, index=False))
    F = make_features(df)
    out("\n## missing after parsing (all cleaned rows)\n" + md(F[ALL].isna().mean().map("{:.1%}".format).to_frame("missing")))
    raw_t = df[P + "thermal_annealing_temperature"]
    out(f"stepwise anneal values parsed: temp {(raw_t.notna() & num(raw_t).isna() & F['anneal_temp'].notna()).sum()} rows")

    tr_i, te_i = split_from_file(F["doi"])  # roles saved in splits/doi_split.csv
    tr, te = F.iloc[tr_i].reset_index(drop=True), F.iloc[te_i].reset_index(drop=True)
    out(f"\ntrain {len(tr)} rows / {tr.doi.nunique()} papers; test {len(te)} rows / {te.doi.nunique()} papers")

    # ---- 1. tuning (GroupKFold on train only; no per-year look) ----
    out("\n## tuning (GroupKFold on train, all variables)")
    t = tune(tr, ALL)
    out(md(t.assign(cv_MAE=t.cv_MAE.round(3), cv_R2=t.cv_R2.round(3)), index=False))
    best_each = t.loc[t.groupby("model")["cv_MAE"].idxmin()]
    best = t.loc[t["cv_MAE"].idxmin()]
    kind, params, impute = best["model"], best["params"], best["missing"] == "median+flag"

    out("\n## test scores")
    rows = [("train mean", *scores(te.pce, np.full(len(te), tr.pce.mean())))]
    for _, b in best_each.iterrows():
        m = Model(b["model"], b["params"], b["missing"] == "median+flag", ALL).fit(tr, tr.pce)
        rows.append((f"{b['model']} ({b['missing']})", *scores(te.pce, m.predict(te))))
    out(md(pd.DataFrame(rows, columns=["model", "MAE", "R2"]).round(3), index=False))
    final = Model(kind, params, impute, ALL).fit(tr, tr.pce)
    te["pred"] = final.predict(te)

    # v1 setup on the v2 data / split (v1 drops rows it can't parse, so compare on that subset too)
    F["v1_temp"] = num(df[P + "thermal_annealing_temperature"])
    F["v1_time"] = num(df[P + "thermal_annealing_time"])
    F["v1_solvent"] = F["solvent"]
    F["v1_add"] = (df["Perovskite_additives_compounds"].fillna("Undoped") != "Undoped").astype(int)
    trv, tev = F.iloc[tr_i], F.iloc[te_i]
    trv = trv.dropna(subset=["v1_temp", "v1_time", "v1_solvent"])
    tev_ok = tev[["v1_temp", "v1_time", "v1_solvent"]].notna().all(axis=1).values
    tev = tev[tev_ok]
    topv1 = trv["v1_solvent"].value_counts().index[:4]
    cat = lambda s: pd.Categorical(s.where(s.isin(topv1), "other"), categories=[*topv1, "other"])
    Xv = lambda X: pd.DataFrame({"temp": X.v1_temp, "time": X.v1_time, "solvent": cat(X.v1_solvent), "additive": X.v1_add})
    mv1 = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0).fit(Xv(trv), trv.pce)
    out("\n## v1 vs v2 on the same test rows (v1-parsable subset)")
    out(md(pd.DataFrame([
        ("v1 setup (4 vars, HGB default)", len(tev), *scores(tev.pce, mv1.predict(Xv(tev)))),
        (f"v2 ({kind})", len(tev), *scores(tev.pce, te.loc[tev_ok, "pred"])),
        (f"v2 ({kind}), full test", len(te), *scores(te.pce, te.pred)),
    ], columns=["model", "test rows", "MAE", "R2"]).round(3), index=False))

    # ---- group ablation ----
    out("\n## adding variable groups (same model/params)")
    rows, feats = [], []
    for g, vs in C.GROUPS.items():
        feats = feats + vs
        cmae, cr2 = cv(kind, params, impute, feats, tr)
        m = Model(kind, params, impute, feats).fit(tr, tr.pce)
        rows.append((" + ".join(list(C.GROUPS)[:list(C.GROUPS).index(g) + 1]), len(feats), cmae, cr2,
                     *scores(te.pce, m.predict(te))))
    out(md(pd.DataFrame(rows, columns=["groups", "n vars", "cv_MAE", "cv_R2", "test_MAE", "test_R2"]).round(3), index=False))

    # ---- 2. sensitivity: explicit antisolvent + additive only ----
    explicit = lambda X: (X["antisolvent"] != "none_or_unreported") & X["additive"].notna()
    tr_e, te_e = tr[explicit(tr)], te[explicit(te)]
    m_e = Model(kind, params, impute, ALL).fit(tr_e, tr_e.pce)
    out("\n## sensitivity: rows with antisolvent and additive explicitly recorded")
    out(md(pd.DataFrame([
        ("full-data model", len(tr), len(te_e), *scores(te_e.pce, te.loc[te_e.index, "pred"])),
        ("explicit-only model", len(tr_e), len(te_e), *scores(te_e.pce, m_e.predict(te_e))),
    ], columns=["model", "train rows", "test rows (explicit)", "MAE", "R2"]).round(3), index=False))

    # ---- 3. permutation importance ----
    imp = permutation(final, te, te.pce, ALL)
    out("\n## permutation importance (test, 5 repeats)\n" + md(imp.head(15).round(3), index=False))
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.head(15)[::-1].plot.barh(x="feature", y="R2_drop", xerr="R2_drop_sd", ax=ax, legend=False, color="#4C72B0")
    ax.set_xlabel("R² drop when shuffled (test)")
    fig.tight_layout()
    fig.savefig(FIG / "importance.png", dpi=110)
    plt.close(fig)

    # ---- 4. error analysis ----
    te["key"] = combo_key(te)
    te["abs_err"] = (te.pce - te.pred).abs()
    g = te.dropna(subset=["key"]).groupby("key").agg(n=("pce", "size"), papers=("doi", "nunique"), actual=("pce", "mean"),
                                                    pred=("pred", "mean"), MAE=("abs_err", "mean"))
    worst = g[g.n >= C.MIN_DEVICES].nlargest(20, "MAE")
    out("\n## 20 combinations with largest test error (>= 3 devices)\n" + md(worst.round(2)))
    w = te[te.key.isin(worst.index)]
    out(f"\nworst-20 rows: {len(w)}, papers {w.doi.nunique()}, mean actual {w.pce.mean():.1f} vs pred {w.pred.mean():.1f};"
          f" all test: actual {te.pce.mean():.1f} vs pred {te.pred.mean():.1f}")
    out(f"share of worst rows with actual PCE < 5%: {(w.pce < 5).mean():.0%} (all test {(te.pce < 5).mean():.0%})")
    for c in ["solvent", "antisolvent", "additive", "arch", "backcontact", "scan_direction"]:
        a = w[c].fillna(C.MISSING).value_counts(normalize=True).head(4)
        b = te[c].fillna(C.MISSING).value_counts(normalize=True)
        out(f"{c}: " + ", ".join(f"{k} {v:.0%} (all {b.get(k, 0):.0%})" for k, v in a.items()))
    out(f"year median: worst {w.year.median():.0f} vs all {te.year.median():.0f}")

    # ---- 5. Cl additive check (train papers) ----
    cl, blank = tr[tr.additive == "Cl"], tr[tr.additive.isna()]
    out("\n## Cl vs additive-not-recorded (train)")
    rows = []
    for name, s in [("Cl", cl), ("not recorded", blank)]:
        rows.append((name, len(s), s.pce.median(), s.anneal_time.median(), (s.anneal_time >= 30).mean(),
                     (s.antisolvent != "none_or_unreported").mean(), s.year.median(), (s.year <= 2015).mean()))
    out(md(pd.DataFrame(rows, columns=["group", "rows", "PCE median", "anneal time median", "time>=30min share",
                                         "antisolvent used share", "year median", "year<=2015 share"]).round(2), index=False))
    for name, s in [("Cl", cl), ("not recorded", blank)]:
        out(f"{name} anneal_time quartiles: {s.anneal_time.quantile([0.25, 0.5, 0.75]).round(0).tolist()}")
    long = tr[tr.anneal_time >= 30]
    lc, lb = long[long.additive == "Cl"], long[long.additive.isna()]
    out(f"anneal >= 30 min: Cl median {lc.pce.median():.1f} (n={len(lc)}) vs not recorded {lb.pce.median():.1f} (n={len(lb)})")
    for extra, mask in [("+ no antisolvent", lambda s: s.antisolvent == "none_or_unreported"),
                        ("+ year 2014-2016", lambda s: s.year.between(2014, 2016))]:
        a, b = lc[mask(lc)], lb[mask(lb)]
        out(f"anneal >= 30 min {extra}: Cl {a.pce.median():.1f} (n={len(a)}) vs not recorded {b.pce.median():.1f} (n={len(b)})")

    # ---- 6. combination bins (all cleaned rows) ----
    F["key"] = combo_key(F)
    cg = F.dropna(subset=["key"]).groupby("key").agg(n=("pce", "size"), papers=("doi", "nunique"), pce=("pce", "mean"))
    big = cg[cg.n >= C.MIN_DEVICES]
    out("\n## process-variable combinations (all cleaned rows)")
    out(f"rows with all numeric process vars: {F.key.notna().sum()} / {len(F)}")
    out(f"combinations: {len(cg)}; with >= {C.MIN_DEVICES} devices: {len(big)} (covering {big.n.sum()} rows);"
          f" with >= 3 devices from >= 2 papers: {(big.papers >= 2).sum()}")
    out("mean PCE per combination (>= 3 devices): " + describe(big.pce))
    out("devices per combination (>= 3): " + describe(big.n.astype(float)))

    # ---- 7. experiment-2 split: train <= 2017, evaluate 2018-2019 ----
    old = F[(F.year <= C.SPLIT_YEAR) & F.key.notna()]
    new = F[F.year.between(*C.EVAL_YEARS) & F.key.notna()]
    seen = set(old.key)
    ng = new.groupby("key").agg(n=("pce", "size"), papers=("doi", "nunique"), pce=("pce", "mean"))
    ng["first_seen"] = ~ng.index.isin(seen)
    out(f"\n## experiment 2 split: <= {C.SPLIT_YEAR} ({len(old)} rows) vs {C.EVAL_YEARS[0]}-{C.EVAL_YEARS[1]} ({len(new)} rows)")
    for flag, name in [(True, "first seen in eval period"), (False, "already seen <= 2017")]:
        s = ng[ng.first_seen == flag]
        s3 = s[s.n >= C.MIN_DEVICES]
        out(f"{name}: {len(s)} combos ({s.n.sum()} rows); >= 3 devices: {len(s3)} combos ({s3.n.sum()} rows)")
        if len(s3):
            out("   mean PCE (>= 3 devices): " + describe(s3.pce))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(big.pce, bins=30, color="#4C72B0")
    axes[0].set_title(f"mean PCE per combination (>= 3 devices, n={len(big)})")
    for flag, name, col in [(False, "seen <= 2017", "#999999"), (True, "first seen 2018-19", "#DD8452")]:
        s = ng[(ng.first_seen == flag) & (ng.n >= C.MIN_DEVICES)].pce
        axes[1].hist(s, bins=20, alpha=0.6, label=f"{name} (n={len(s)})", color=col)
    axes[1].set_title("2018-19 combinations (>= 3 devices)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("mean PCE (%)")
    fig.tight_layout()
    fig.savefig(FIG / "combinations.png", dpi=110)
    plt.close(fig)

    mae, r2 = scores(te.pce, te.pred)
    for line in [
        "\n=== v2 summary ===",
        f"cleaned: {len(F)} rows / {F.doi.nunique()} papers (train {len(tr)}, test {len(te)})",
        f"best: {kind} ({best['missing']}) {params}  CV MAE {best['cv_MAE']:.2f} R2 {best['cv_R2']:.3f}",
        f"test: MAE {mae:.2f} R2 {r2:.3f}   (v1 baseline: MAE 3.50 R2 0.136)",
        "top importance: " + ", ".join(imp.feature.head(5)),
        f"combinations >= {C.MIN_DEVICES} devices: {len(big)};  first seen in 2018-19: {ng.first_seen.sum()}",
        f"details: {RESULTS.relative_to(ROOT)}  figures: {FIG.relative_to(ROOT)}",
    ]:
        out(line, console=True)

    (OUT / "best_params.json").write_text(json.dumps({
        "model": kind, "missing": best["missing"], "params": params, "features": C.GROUPS,
        "cv_MAE": round(best["cv_MAE"], 4), "cv_R2": round(best["cv_R2"], 4)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
