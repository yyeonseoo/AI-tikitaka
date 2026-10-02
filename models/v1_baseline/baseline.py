"""v1 baseline. Usage: python models/v1_baseline/baseline.py [1]  (1 = download + data report; default = steps 2-4)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `src` imports from any cwd
from src.data import load_nomad as load, mapbi_onestep, split_by_paper


def step1():
    df = load()
    print(f"\nrows: {len(df)}, columns: {df.shape[1]}")
    print("\n[all columns]\n" + ", ".join(df.columns))
    cols = ["Perovskite_composition_short_form", "Perovskite_composition_long_form",
            "Perovskite_deposition_procedure", "Perovskite_deposition_solvents",
            "Perovskite_additives_compounds", "Perovskite_deposition_thermal_annealing_temperature",
            "Perovskite_deposition_thermal_annealing_time", "JV_default_PCE", "Ref_DOI_number"]
    print("\n[missing rate]")
    print(df.reindex(columns=cols).isna().mean().map("{:.1%}".format).to_string())
    for c in ["Perovskite_composition_short_form", "Perovskite_deposition_procedure", "Perovskite_deposition_solvents"]:
        print(f"\n[top 5 {c}]\n{df[c].value_counts().head(5).to_string()}")


FEATURES = ["temp", "time", "solvent", "additive"]


def step2(df):
    df = mapbi_onestep(df)
    print(f"MAPbI + one-step spin-coating: {len(df)} rows")
    d = pd.DataFrame({
        "doi": df["Ref_DOI_number"],
        "temp": pd.to_numeric(df["Perovskite_deposition_thermal_annealing_temperature"], errors="coerce"),  # degC
        "time": pd.to_numeric(df["Perovskite_deposition_thermal_annealing_time"], errors="coerce"),  # min
        # main solvent = precursor solvent(s) of the first step, before any ">>" (e.g. antisolvent drip)
        "solvent": df["Perovskite_deposition_solvents"].str.split(">>").str[0].str.strip(),
        # missing / "Undoped" = no additive; missing is 65% of rows, dropping them would gut the data
        "additive": (df["Perovskite_additives_compounds"].fillna("Undoped") != "Undoped").astype(int),
        "pce": pd.to_numeric(df["JV_default_PCE"], errors="coerce"),
    }).dropna()
    top = d["solvent"].value_counts().index[:4]
    d["solvent"] = pd.Categorical(d["solvent"].where(d["solvent"].isin(top), "other"), categories=[*top, "other"])
    print(f"after cleaning: {len(d)} rows, {d['doi'].nunique()} papers")
    print(d["solvent"].value_counts().to_string())
    return d.reset_index(drop=True)


def step3(d):
    tr, te = split_by_paper(d["doi"])
    tr, te = d.iloc[tr], d.iloc[te]
    assert not set(tr["doi"]) & set(te["doi"])  # no paper on both sides
    print(f"\ntrain {len(tr)} rows / test {len(te)} rows")
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
    model.fit(tr[FEATURES], tr["pce"])
    for name, pred in [("mean baseline", np.full(len(te), tr["pce"].mean())), ("HistGBR", model.predict(te[FEATURES]))]:
        print(f"{name:14s} MAE {mean_absolute_error(te['pce'], pred):.2f}  R2 {r2_score(te['pce'], pred):.3f}")
    return model.fit(d[FEATURES], d["pce"])  # refit on all data for recommendation


def step4(d, model):
    cats = d["solvent"].cat.categories
    grid = pd.MultiIndex.from_product([range(60, 161, 10), [5, 10, 15, 20, 30, 45, 60, 90, 120],
                                       [c for c in cats if c != "other"], [0, 1]], names=FEATURES).to_frame(index=False)
    grid["solvent"] = pd.Categorical(grid["solvent"], categories=cats)
    grid["pred"] = model.predict(grid[FEATURES])
    print("\n[top 5 predicted conditions]  (temp degC, time min)")
    for _, g in grid.nlargest(5, "pred").iterrows():
        print(f"\n{g.temp} C, {g.time} min, {g.solvent}, additive={g.additive} -> predicted PCE {g.pred:.2f}%")
        # ponytail: nearest = same solvent/additive, scaled temp/time distance; no kNN index needed at this size
        same = d[(d["solvent"] == g.solvent) & (d["additive"] == g.additive)].drop_duplicates()  # DB has repeated rows
        dist = ((same["temp"] - g.temp) / d["temp"].std()) ** 2 + ((same["time"] - g.time) / d["time"].std()) ** 2
        for _, r in same.loc[dist.nsmallest(3).index].iterrows():
            print(f"    {r.doi}  {r.temp:g} C, {r.time:g} min  actual PCE {r.pce:.2f}%")


if __name__ == "__main__":
    if sys.argv[1:] == ["1"]:
        step1()
    else:
        d = step2(load())
        step4(d, step3(d))
