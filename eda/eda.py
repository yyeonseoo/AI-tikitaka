"""EDA for choosing process variables (MAPbI3 + one-step spin coating).
Usage: python eda/eda.py [scan]   (scan = section 9 column scan)
Prints markdown tables (copied into reports/eda_report.md) and saves plots to eda/figures/."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `src` imports from any cwd
from src.data import ORIG_CSV, drop_llm_and_non_1sun, load_nomad, mapbi_onestep, split_by_paper
from src.features import antisolvent, dmso_frac, first_step, md, top

FIG = Path(__file__).resolve().parent / "figures"
P = "Perovskite_deposition_"

# name -> (raw DB column, kind). Derived values are built in derive().
VARS = {
    "conc_PbI2_M": (P + "reaction_solutions_concentrations", "num"),
    "solvent": (P + "solvents", "cat"),
    "solvent_mix": (P + "solvents_mixing_ratios", "cat"),
    "dmso_frac": (P + "solvents_mixing_ratios", "num"),
    "antisolvent": (P + "quenching_media", "cat"),
    "anneal_temp": (P + "thermal_annealing_temperature", "num"),
    "anneal_time": (P + "thermal_annealing_time", "num"),
    "atmosphere": (P + "synthesis_atmosphere", "cat"),
    "rh": (P + "synthesis_atmosphere_relative_humidity", "num"),
}
REFS = {
    "year": ("Ref_publication_date", "num"),
    "arch": ("Cell_architecture", "cat"),
    "etl": ("ETL_stack_sequence", "cat"),
    "htl": ("HTL_stack_sequence", "cat"),
}
NUM = [k for k, (_, t) in VARS.items() if t == "num"]
CAT = [k for k, (_, t) in VARS.items() if t == "cat"]


def pbi2_conc(compounds, concs):
    # concentration listed per compound, ";"-separated in the same order; keep only molar units
    if not isinstance(compounds, str) or not isinstance(concs, str):
        return np.nan
    names = [c.strip() for c in compounds.split(">>")[0].split(";")]
    vals = [c.strip() for c in concs.split(">>")[0].split(";")]
    if "PbI2" not in names or len(vals) != len(names):
        return np.nan
    m = re.fullmatch(r"([\d.]+)\s*(M|mol/L)", vals[names.index("PbI2")])
    return float(m.group(1)) if m else np.nan


def solvent_mix(solvents, ratios):
    # "DMF; DMSO" + "4; 1" and "0.8; 0.2" -> same label "DMF:DMSO 80:20"
    if not isinstance(solvents, str):
        return np.nan
    names = [s.strip() for s in solvents.split(";")]
    if len(names) == 1:
        return names[0]
    r = pd.to_numeric(pd.Series(str(ratios).split(";")), errors="coerce")
    if len(r) != len(names) or r.isna().any() or r.sum() == 0:
        return ":".join(names) + " (ratio ?)"
    return ":".join(names) + " " + ":".join(f"{v:.0f}" for v in (r / r.sum() * 100).round())


def atmosphere(s):
    s = first_step(s).str.lower()
    m = s.map({"air": "Air", "ambient air": "Air", "ambient": "Air", "dry air": "Dry air",
               "n2": "N2", "ar": "Ar", "vacuum": "Vacuum"})
    return m.where(s.isna() | m.notna(), "other")


def load():
    return drop_llm_and_non_1sun(mapbi_onestep(load_nomad())).reset_index(drop=True)


def derive(df):
    solv = first_step(df[P + "solvents"])
    ratio = first_step(df[P + "solvents_mixing_ratios"])
    d = pd.DataFrame({
        "doi": df["Ref_DOI_number"],
        "pce": pd.to_numeric(df["JV_default_PCE"], errors="coerce"),
        "conc_PbI2_M": [pbi2_conc(a, b) for a, b in zip(df[P + "reaction_solutions_compounds"],
                                                         df[P + "reaction_solutions_concentrations"])],
        "solvent": top(solv, 6),
        "solvent_mix": top(pd.Series([solvent_mix(a, b) for a, b in zip(solv, ratio)]), 8),
        "dmso_frac": [dmso_frac(a, b) for a, b in zip(df[P + "solvents"], df[P + "solvents_mixing_ratios"])],
        "antisolvent": top(pd.Series([antisolvent(a, b) for a, b in
                                      zip(df[P + "quenching_media"], df[P + "quenching_induced_crystallisation"])]), 7),
        "anneal_temp": pd.to_numeric(df[P + "thermal_annealing_temperature"], errors="coerce"),
        "anneal_time": pd.to_numeric(df[P + "thermal_annealing_time"], errors="coerce"),
        "atmosphere": atmosphere(df[P + "synthesis_atmosphere"]),
        "rh": pd.to_numeric(first_step(df[P + "synthesis_atmosphere_relative_humidity"]), errors="coerce"),
        "year": pd.to_datetime(df["Ref_publication_date"], errors="coerce", utc=True).dt.year,
        "arch": df["Cell_architecture"].where(df["Cell_architecture"].isin(["nip", "pin"])),
        "etl": top(df["ETL_stack_sequence"], 6),
        "htl": top(df["HTL_stack_sequence"], 6),
    })
    return d


def summary(df, d):
    rows = []
    for name, (col, kind) in {**VARS, **REFS}.items():
        raw = df[col]
        nn = raw.dropna().astype(str)
        vc = nn.value_counts()
        rows.append({
            "var": name, "DB column": col, "missing": f"{raw.isna().mean():.1%}",
            "missing after parse": f"{d[name].isna().mean():.1%}",
            "unique": raw.nunique(), "mode share": f"{vc.iloc[0] / len(nn):.1%}" if len(nn) else "-",
            "top5": "; ".join(f"{k} ({v})" for k, v in vc.head(5).items()).replace("|", "/"),
            "';' share": f"{nn.str.contains(';').mean():.1%}" if len(nn) else "-",
            "'>>' share": f"{nn.str.contains('>>').mean():.1%}" if len(nn) else "-",
        })
    return pd.DataFrame(rows)


def grid(n, w=4.2, h=3.2):
    cols = 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(w * cols, h * ((n + cols - 1) // cols)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


def plot_distributions(d):
    fig, axes = grid(len(NUM))
    for ax, v in zip(axes, NUM):
        x = d[v].dropna()
        ax.hist(x.clip(*x.quantile([0.005, 0.995])), bins=40, color="#4C72B0")
        ax.set_title(f"{v}  (n={len(x)})")
    save(fig, "dist_continuous.png")
    fig, axes = grid(len(CAT), w=5)
    for ax, v in zip(axes, CAT):
        vc = d[v].fillna("(missing)").value_counts()
        ax.barh(vc.index[::-1], vc.values[::-1], color="#55A868")
        ax.set_title(v)
    save(fig, "dist_categorical.png")


def plot_pce(tr):
    fig, axes = grid(len(NUM))
    for ax, v in zip(axes, NUM):
        s = tr[[v, "pce"]].dropna()
        ax.scatter(s[v], s["pce"], s=4, alpha=0.15)
        if s[v].nunique() > 5:  # binned median to show the trend through the cloud
            b = pd.qcut(s[v], 10, duplicates="drop")
            med = s.groupby(b, observed=True).agg(x=(v, "median"), y=("pce", "median"))
            ax.plot(med.x, med.y, "r-o", ms=3)
        ax.set_xlim(*s[v].quantile([0.005, 0.995]))
        ax.set_title(f"{v}  (n={len(s)}, rho={s[v].corr(s['pce'], method='spearman'):.2f})")
        ax.set_ylabel("PCE (%)")
    save(fig, "pce_vs_continuous.png")
    fig, axes = grid(len(CAT), w=5.5, h=3.6)
    for ax, v in zip(axes, CAT):
        s = tr[[v, "pce"]].dropna(subset=["pce"]).fillna({v: "(missing)"})
        order = s.groupby(v)["pce"].median().sort_values().index
        ax.boxplot([s.loc[s[v] == k, "pce"] for k in order], vert=False, showfliers=False)
        ax.set_yticks(range(1, len(order) + 1), [f"{k} ({(s[v] == k).sum()})" for k in order], fontsize=8)
        ax.set_title(v)
        ax.set_xlabel("PCE (%)")
    save(fig, "pce_vs_categorical.png")


def heat(ax, m, title, fmt="{:.2f}", cmap="RdBu_r", vmin=-1, vmax=1):
    im = ax.imshow(m.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if pd.notna(m.iat[i, j]):
                ax.text(j, i, fmt.format(m.iat[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


def cramers_v(a, b):
    t = pd.crosstab(a, b)
    if min(t.shape) < 2:
        return np.nan
    e = np.outer(t.sum(1), t.sum(0)) / t.values.sum()
    chi2 = ((t.values - e) ** 2 / e).sum()
    return np.sqrt(chi2 / t.values.sum() / (min(t.shape) - 1))


def plot_relations(d):
    allv = NUM + CAT
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    heat(axes[0, 0], d[NUM].corr(method="spearman"), "Spearman (continuous)")
    cv = pd.DataFrame([[cramers_v(d[a], d[b]) for b in CAT] for a in CAT], CAT, CAT)
    heat(axes[0, 1], cv, "Cramer's V (categorical, non-missing pairs)", cmap="Blues", vmin=0)
    both = pd.DataFrame([[(d[a].notna() & d[b].notna()).mean() for b in allv] for a in allv], allv, allv)
    heat(axes[1, 0], both, "both reported (share of rows)", fmt="{:.0%}", cmap="Greens", vmin=0)
    ct = pd.crosstab(d["solvent"], d["antisolvent"])
    heat(axes[1, 1], ct, "solvent x antisolvent (rows)", fmt="{:.0f}", cmap="Purples", vmin=0, vmax=ct.values.max())
    save(fig, "relations.png")
    return d[NUM].corr(method="spearman"), cv, both


def plot_confounders(d):
    n_year = d["year"].value_counts()
    d = d[d["year"].isin(n_year[n_year >= 100].index)]  # sparse years (2020+ after LLM exclusion) are noise
    fig, axes = grid(len(NUM) + len(CAT) + 1)
    yr = d.groupby("year")
    axes[0].plot(yr["pce"].median(), "k-o", ms=3)
    axes[0].set_title("median PCE by year")
    for ax, v in zip(axes[1:], NUM):
        ax.plot(yr[v].median(), "-o", ms=3)
        ax2 = ax.twinx()
        ax2.plot(yr[v].apply(lambda s: s.notna().mean()), "--", c="gray", lw=1)
        ax2.set_ylim(0, 1)
        ax.set_title(f"{v}: median / reported share(--)", fontsize=9)
    for ax, v in zip(axes[1 + len(NUM):], CAT):
        share = pd.crosstab(d["year"], d[v].fillna("(missing)"), normalize="index")
        share.plot.area(ax=ax, legend=False, lw=0)
        ax.set_title(f"{v} share by year")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
    save(fig, "year_trends.png")

    a = d.dropna(subset=["arch"])
    fig, axes = grid(len(NUM) + len(CAT) + 1)
    axes[0].boxplot([a.loc[a.arch == k, "pce"].dropna() for k in ["nip", "pin"]], showfliers=False)
    axes[0].set_xticks([1, 2], ["nip", "pin"])
    axes[0].set_title("PCE by architecture")
    for ax, v in zip(axes[1:], NUM):
        ax.boxplot([a.loc[a.arch == k, v].dropna() for k in ["nip", "pin"]], showfliers=False)
        ax.set_xticks([1, 2], ["nip", "pin"])
        ax.set_title(v)
    for ax, v in zip(axes[1 + len(NUM):], CAT):
        pd.crosstab(a[v].fillna("(missing)"), a["arch"], normalize="columns").plot.barh(ax=ax, fontsize=7)
        ax.set_title(f"{v} share within architecture")
    save(fig, "architecture_diff.png")
    num_arch = a.groupby("arch")[["pce", *NUM]].median().T
    return num_arch


def compare_files():
    n = load_nomad()
    o = pd.read_csv(ORIG_CSV, low_memory=False, na_values=["Unknown"], encoding="utf-8-sig")
    print("\n## file comparison")
    print(f"rows: orig {len(o)} / nomad {len(n)};  cols: orig {o.shape[1]} / nomad {n.shape[1]}")
    common = set(o.columns) & set(n.columns)
    print(f"common columns: {len(common)}")
    strip = lambda cs: sorted(c for c in cs if not re.search(r"\.\d+\.", c))  # drop exploded ions/authors lists
    print("only in orig:", strip(set(o.columns) - common))
    print("only in nomad (excl. ions/authors lists):", strip(set(n.columns) - common - {"entry_id"}))
    norm = lambda s: set(s.dropna().astype(str).str.strip().str.lower())
    do, dn = norm(o["Ref_DOI_number"]), norm(n["Ref_DOI_number"])
    print(f"DOIs: orig {len(do)}, nomad {len(dn)}, both {len(do & dn)}, orig-only {len(do - dn)}, nomad-only {len(dn - do)}")
    for name, f in [("orig", o), ("nomad", n)]:
        y = pd.to_datetime(f["Ref_publication_date"], errors="coerce", utc=True, format="mixed").dt.year
        print(f"{name} publication years: {y.min():.0f}-{y.max():.0f} (missing {y.isna().mean():.1%})")
    cols = list(dict.fromkeys(c for c, _ in {**VARS, **REFS}.values())) + ["JV_default_PCE"]
    sub = mapbi_onestep
    t = pd.DataFrame({
        "orig all": o.reindex(columns=cols).notna().mean(), "nomad all": n.reindex(columns=cols).notna().mean(),
        "orig MAPbI 1-step": sub(o).reindex(columns=cols).notna().mean(),
        "nomad MAPbI 1-step": sub(n).reindex(columns=cols).notna().mean(),
    }).map("{:.1%}".format)
    print(f"MAPbI one-step rows: orig {len(sub(o))}, nomad {len(sub(n))}")
    print("\nfilled share of candidate columns:\n" + md(t))


def main():
    FIG.mkdir(exist_ok=True)
    df = load()
    d = derive(df)
    print(f"MAPbI + one-step spin-coating: {len(d)} rows, {d.doi.nunique()} papers")
    print("\n## variable summary\n" + md(summary(df, d), index=False))
    plot_distributions(d)

    has = d.dropna(subset=["doi", "pce"])
    tr_idx, _ = split_by_paper(has["doi"])
    tr = has.iloc[tr_idx]
    print(f"\ntrain (80% of papers): {len(tr)} rows, {tr.doi.nunique()} papers")
    plot_pce(tr)
    print("\n## PCE by category (train): median / n")
    for v in CAT + ["arch"]:
        g = tr.fillna({v: "(missing)"}).groupby(v)["pce"].agg(["median", "count"]).sort_values("median")
        print(f"\n{v}: " + ", ".join(f"{k} {r['median']:.1f} (n={r['count']:.0f})" for k, r in g.iterrows()))
    print("\n## Spearman with PCE (train)")
    print(tr[NUM + ["year"]].corrwith(tr["pce"], method="spearman").round(3).to_string())

    print("\n## within-year check (train, 2015-2019): does the PCE link survive fixing the year?")
    w = tr[tr["year"].between(2015, 2019)]
    print("Spearman with PCE inside each year:")
    print(pd.DataFrame({y: g[NUM].corrwith(g["pce"], method="spearman") for y, g in w.groupby("year")}).round(2).to_string())
    for v in ["antisolvent", "solvent", "arch"]:
        print(f"\nmedian PCE by {v} x year:")
        print(w.pivot_table("pce", v, "year", "median").round(1).to_string())

    sp, cv, both = plot_relations(d)
    print("\n## Spearman among continuous\n" + md(sp.round(2)))
    print("\n## Cramer's V among categorical\n" + md(cv.round(2)))
    print("\n## both reported share\n" + md(both.map("{:.0%}".format)))
    print("\n## median by architecture\n" + md(plot_confounders(d).round(2)))
    print("\n## Spearman with year (all rows)")
    print(d[NUM + ["pce"]].corrwith(d["year"], method="spearman").round(3).to_string())
    compare_files()


# ---- section 9: scan all columns for extra candidates (python eda.py scan) ----
PLACEHOLDER = r"(nan|None)?(\s*;\s*(nan|None)?)*|0000:00:00:00:00"  # schema defaults that mean "not reported"
ANALYZED = {c for c, _ in {**VARS, **REFS}.values()} | {P + "solvents_mixing_ratios", P + "quenching_induced_crystallisation",
                                                        "JV_default_PCE", "JV_light_intensity", "JV_light_spectra"}
# (column, group, kind). group: a=process, b=condition, c=measurement correction, d=target
SCAN = [
    ("Perovskite_additives_compounds", "a", "additive"),
    (P + "solvent_annealing", "a", "bool"),
    ("Perovskite_thickness", "a", "num"),
    ("Substrate_stack_sequence", "b", "cat"),
    ("Backcontact_stack_sequence", "b", "cat"),
    ("Backcontact_thickness_list", "b", "num"),
    ("Backcontact_deposition_procedure", "b", "cat"),
    ("ETL_thickness", "b", "num"),
    ("ETL_deposition_procedure", "b", "cat"),
    ("HTL_deposition_procedure", "b", "cat"),
    ("HTL_additives_compounds", "b", "cat"),
    ("Cell_flexible", "b", "bool"),
    ("Add_lay_front", "b", "bool"),
    ("Add_lay_back", "b", "bool"),
    ("Encapsulation_Encapsulation", "b", "bool"),
    ("Cell_area_measured", "c", "num"),
    ("JV_default_PCE_scan_direction", "c", "cat"),
    ("JV_scan_speed", "c", "num"),
    ("JV_test_atmosphere", "c", "cat"),
    ("JV_average_over_n_number_of_cells", "c", "num"),
    ("JV_certified_values", "c", "bool"),
    ("JV_light_masked_cell", "c", "bool"),
    ("JV_hysteresis_index", "c", "num"),  # 10% rule for measurement-correction columns
    ("rev_minus_fwd_PCE", "c", "num"),  # derived: JV_reverse_scan_PCE - JV_forward_scan_PCE
    ("JV_default_Voc", "d", "num"),
    ("JV_default_Jsc", "d", "num"),
    ("JV_default_FF", "d", "num"),
    ("Stability_measured", "d", "bool"),
]
BELOW = [P + "reaction_solutions_compounds", "Perovskite_additives_concentrations", P + "substrate_temperature",
         P + "reaction_solutions_temperature", P + "reaction_solutions_age", P + "thermal_annealing_atmosphere",
         P + "thermal_annealing_relative_humidity", "Perovskite_surface_treatment_before_next_deposition_step",
         P + "after_treatment_of_formed_perovskite", P + "quenching_media_volume", "HTL_thickness_list",
         "Stability_PCE_T80", "Stability_PCE_end_of_experiment", "Stability_PCE_initial_value",
         "Stability_time_total_exposure", "Stability_PCE_after_1000_h"]


def train_mask(doi, pce):
    has = doi.notna() & pce.notna()
    idx = has[has].index
    tr, _ = split_by_paper(doi[idx])
    return doi.index.isin(idx[tr])


def is_bool(s):
    return set(s.dropna().astype(str).unique()) <= {"True", "False"} and s.notna().any()


def leftover_reason(c, s):
    if c in ANALYZED or c.startswith(("Ref_", "entry_id")):
        return "analyzed in 1-8 / reference metadata"
    if c.startswith(("Perovskite_ions.", "Perovskite_composition", "Perovskite_dimension", "Perovskite_single",
                     "Perovskite_band_gap", "Perovskite_deposition_procedure", "Perovskite_deposition_number",
                     "Perovskite_deposition_aggregation")):
        return "fixed by the MAPbI3 one-step filter"
    if c.startswith(("Outdoor_", "Stability_", "Module_", "Stabilised_", "EQE_")):
        return "only defaults/flags (no metric filled >=30%)"
    if c.startswith(("JV_reverse", "JV_default", "JV_forward")):
        return "same measurement as PCE / Voc / Jsc / FF"
    if c in ("Cell_stack_sequence", "ETL_stack_sequence", "HTL_stack_sequence", "Cell_architecture"):
        return "covered by architecture / ETL / HTL"
    if s.nunique() <= 1:
        return "single value"
    return "device-count / measurement flags"


def rel_num(x, y, year):
    rho = x.corr(y, method="spearman")
    per = [g.iloc[:, 0].corr(g.iloc[:, 1], method="spearman") for _, g in pd.concat([x, y], axis=1).groupby(year)
           if len(g.dropna()) >= 30]
    per = [r for r in per if pd.notna(r)]
    return f"rho {rho:.2f}", (f"{min(per):.2f} ~ {max(per):.2f} ({len(per)} yrs)" if per else "n/a")


def rel_cat(x, y, year, ref):
    out = []
    for k in x.dropna().unique():
        diffs = []
        for yr in range(2015, 2020):
            a, b = y[(x == k) & (year == yr)], y[(x == ref) & (year == yr)]
            if len(a) >= 10 and len(b) >= 10:
                diffs.append(a.median() - b.median())
        all_k = y[x == k]
        out.append((k, all_k.median(), len(all_k.dropna()),
                    f"{min(diffs):+.1f} ~ {max(diffs):+.1f} ({len(diffs)} yrs)" if diffs else "n/a"))
    return sorted(out, key=lambda r: -r[2])


def scan():
    df = load()
    df = df.mask(df.astype(str).apply(lambda s: s.str.fullmatch(PLACEHOLDER)))
    df["rev_minus_fwd_PCE"] = (pd.to_numeric(df["JV_reverse_scan_PCE"], errors="coerce")
                               - pd.to_numeric(df["JV_forward_scan_PCE"], errors="coerce"))
    fill, uniq = df.notna().mean(), df.nunique()
    bools = [c for c in df.columns if is_bool(df[c])]
    true_share = {c: (df[c].astype(str) == "True").mean() for c in bools}

    print(f"rows {len(df)}, columns {df.shape[1] - 1} (+1 derived)")
    keep = [c for c in df.columns if (c in bools and true_share[c] >= 0.01) or (c not in bools and fill[c] >= 0.3)]
    print(f"bool columns: {len(bools)} (True share >= 1%: {sum(v >= 0.01 for v in true_share.values())});"
          f" non-bool with fill >= 30%: {sum(fill[c] >= 0.3 for c in df.columns if c not in bools)}; kept {len(keep)}")
    scanned = {c for c, _, _ in SCAN}
    left = pd.DataFrame([(c, f"{fill[c]:.0%}" if c not in bools else f"True {true_share[c]:.1%}", uniq[c],
                          leftover_reason(c, df[c])) for c in keep if c not in scanned],
                        columns=["column", "fill", "unique", "why not a new candidate"])
    print("\n## kept but not new candidates (grouped)\n" + md(left.groupby("why not a new candidate")["column"]
                                                           .agg(["count", lambda s: ", ".join(s[:6]) + (" ..." if len(s) > 6 else "")])))

    pce = pd.to_numeric(df["JV_default_PCE"], errors="coerce")
    year = pd.to_datetime(df["Ref_publication_date"], errors="coerce", utc=True).dt.year
    tr = train_mask(df["Ref_DOI_number"], pce)
    print(f"\ntrain rows {tr.sum()}")

    rows, details = [], []
    for c, grp, kind in SCAN:
        s = df[c]
        vc = s.dropna().astype(str).value_counts()
        fill_txt = f"True {true_share[c]:.1%}" if kind == "bool" else f"{fill[c]:.1%}"
        if kind == "additive":
            fill_txt = (f"{fill[c]:.1%} (named {((s.notna()) & (s != 'Undoped')).mean():.1%},"
                        f" Undoped {(s == 'Undoped').mean():.1%}, blank {s.isna().mean():.1%})")
        rows.append((grp, c, fill_txt, uniq[c], "; ".join(f"{k} ({v})" for k, v in vc.head(5).items()).replace("|", "/")))
        x, y, yr = s[tr], pce[tr], year[tr]
        if kind == "num":
            x = pd.to_numeric(x, errors="coerce")
            o, w = rel_num(x, y, yr)
            details.append(f"\n**{c}** (n={x.notna().sum()}): train {o}; within-year rho {w}")
        else:
            if kind == "additive":
                x = x.fillna("(blank)").where(lambda v: v.isin([*vc.index[:5], "(blank)"]), "other")
                ref = "(blank)"
            elif kind == "bool":
                ref = "False"
                x = x.astype(str)
            else:
                x = x.where(x.isin(vc.index[:5]) | x.isna(), "other")
                ref = vc.index[0]
            details.append(f"\n**{c}** (ref = {ref}): " + "; ".join(
                f"{k}: median {m:.1f} (n={n}), vs ref within-year {d}" for k, m, n, d in rel_cat(x, y, yr, ref)))
    print("\n## candidates\n" + md(pd.DataFrame(rows, columns=["group", "column", "fill", "unique", "top5"]), index=False))
    print("\n## PCE relation (train)" + "".join(details))
    print("\n## below threshold (fill only)\n" + md(pd.DataFrame(
        [(c, f"{fill.get(c, 0):.1%}") for c in BELOW], columns=["column", "fill"]), index=False))


if __name__ == "__main__":
    scan() if sys.argv[1:] == ["scan"] else main()
