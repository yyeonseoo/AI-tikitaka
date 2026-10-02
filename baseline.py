"""Perovskite PCE baseline. Usage: python baseline.py [1]  (1 = download + data report; default = steps 2-4)"""
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

API = "https://nomad-lab.eu/prod/v1/api/v1/"
QUERY = {"section_defs.definition_qualified_name:all": ["perovskite_solar_cell_database.schema.PerovskiteSolarCell"]}
SECTIONS = ["ref", "cell", "module", "substrate", "etl", "perovskite", "perovskite_deposition", "htl",
            "backcontact", "add", "encapsulation", "jv", "stabilised", "eqe", "stability", "outdoor"]
PREFIX = {"jv": "JV", "etl": "ETL", "htl": "HTL", "eqe": "EQE"}  # legacy CSV column prefixes
CSV = Path("data/perovskite_db.csv")
RAW = Path("data/raw")


def post(endpoint, body, tries=10):
    req = urllib.request.Request(API + endpoint, json.dumps(body).encode(), {"Content-Type": "application/json",
                                                                          "Accept-Encoding": "gzip"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
                return json.loads(gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body)
        except Exception as e:
            if i == tries - 1:
                raise
            wait = int(getattr(e, "headers", {}).get("Retry-After") or 10 * (i + 1))
            print(f"  retry {endpoint} in {wait}s: {e}")
            time.sleep(wait)


def entry_ids():
    ids, after = [], None
    while True:
        page = {"page_size": 10000, **({"page_after_value": after} if after else {})}
        d = post("entries/query", {"owner": "visible", "query": QUERY, "pagination": page,
                                   "required": {"include": ["entry_id"]}})
        ids += [e["entry_id"] for e in d["data"]]
        after = d["pagination"].get("next_page_after_value")
        if not after:
            return ids


def flatten(quantities):
    # search index holds every data.* field as {path_archive, <type>_value}; far faster than the archive API
    row = {}
    for q in quantities:
        sec, _, k = q.get("path_archive", "").removeprefix("data.").partition(".")
        if sec in SECTIONS and k:
            v = next((v for f, v in q.items() if f.endswith("_value")), None)
            row[f"{PREFIX.get(sec, sec.capitalize())}_{k}"] = v
    return row


def fetch_chunk(args):
    i, ids = args
    f = RAW / f"{i}.json"  # per-chunk cache so a crashed run resumes
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    d = post("entries/query", {"owner": "visible", "query": {"entry_id:any": ids},
                               "pagination": {"page_size": len(ids)}})
    rows = [{"entry_id": e["entry_id"], **flatten(e.get("search_quantities", []))} for e in d["data"]]
    f.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def download():
    ids = entry_ids()
    print(f"{len(ids)} entries, downloading archives...")
    RAW.mkdir(parents=True, exist_ok=True)
    chunks = list(enumerate(ids[i:i + 500] for i in range(0, len(ids), 500)))
    rows = []
    with ThreadPoolExecutor(3) as ex:  # ~10s per 500-entry page
        for n, part in enumerate(ex.map(fetch_chunk, chunks), 1):
            rows += part
            if n % 10 == 0:
                print(f"  {n}/{len(chunks)} chunks")
    pd.DataFrame(rows).to_csv(CSV, index=False)


def load():
    if not CSV.exists():
        download()
    return pd.read_csv(CSV, low_memory=False, na_values=["Unknown"])  # DB uses "Unknown" for missing


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
    df = df[(df["Perovskite_composition_short_form"] == "MAPbI")  # DB short form for MAPbI3
            & (df["Perovskite_deposition_procedure"] == "Spin-coating")]  # one-step: no ">>" second step
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
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(d, groups=d["doi"]))
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
