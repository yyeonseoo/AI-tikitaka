"""Shared data code: NOMAD download, loading, filtering and the paper-level train/test split."""
import gzip
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOMAD_CSV = DATA / "perovskite_db.csv"
ORIG_CSV = DATA / "Perovskite_database_content_all_data.csv"  # 2022 archive: github.com/Jesperkemist/perovskitedatabase_data
RAW = DATA / "raw"

API = "https://nomad-lab.eu/prod/v1/api/v1/"
QUERY = {"section_defs.definition_qualified_name:all": ["perovskite_solar_cell_database.schema.PerovskiteSolarCell"]}
SECTIONS = ["ref", "cell", "module", "substrate", "etl", "perovskite", "perovskite_deposition", "htl",
            "backcontact", "add", "encapsulation", "jv", "stabilised", "eqe", "stability", "outdoor"]
PREFIX = {"jv": "JV", "etl": "ETL", "htl": "HTL", "eqe": "EQE"}  # legacy CSV column prefixes


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
    print(f"{len(ids)} entries, downloading...")
    RAW.mkdir(parents=True, exist_ok=True)
    chunks = list(enumerate(ids[i:i + 500] for i in range(0, len(ids), 500)))
    rows = []
    with ThreadPoolExecutor(3) as ex:  # ~10s per 500-entry page
        for n, part in enumerate(ex.map(fetch_chunk, chunks), 1):
            rows += part
            if n % 10 == 0:
                print(f"  {n}/{len(chunks)} chunks")
    pd.DataFrame(rows).to_csv(NOMAD_CSV, index=False)


def load_nomad():
    if not NOMAD_CSV.exists():
        download()
    return pd.read_csv(NOMAD_CSV, low_memory=False, na_values=["Unknown"])  # DB uses "Unknown" for missing


def mapbi_onestep(df):
    return df[(df["Perovskite_composition_short_form"] == "MAPbI")  # DB short form for MAPbI3
              & (df["Perovskite_deposition_procedure"] == "Spin-coating")]  # one-step: no ">>" second step


def drop_llm_and_non_1sun(df):
    llm = df["Ref_extraction_method"] == "LLM"  # NOMAD-only auto-extracted rows; kept out for data quality
    # 1 sun only: low-light (indoor) runs report PCE > 30% that isn't comparable
    sun = pd.to_numeric(df["JV_light_intensity"], errors="coerce").between(90, 110)
    print(f"excluded LLM-extracted rows: {llm.sum()}, non-1-sun rows: {(~llm & ~sun).sum()}")
    return df[~llm & sun]


def split_by_paper(groups, test_size=0.2, seed=0):
    """Positional (train, test) indices with no paper (DOI) on both sides. Only used to build splits/doi_split.csv;
    scripts read the saved roles through split_from_file so every script sees the same papers in test."""
    return next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(groups, groups=groups))


SPLIT_CSV = ROOT / "splits" / "doi_split.csv"


def split_roles():
    return pd.read_csv(SPLIT_CSV).set_index("doi")["role"]


def split_from_file(dois):
    """Positional (train, test) indices from splits/doi_split.csv. Fails loudly on a DOI the file doesn't know."""
    role = dois.map(split_roles())
    missing = dois[role.isna()].unique()
    if len(missing):
        raise ValueError(f"{len(missing)} DOIs not in {SPLIT_CSV.name} (e.g. {missing[:3]}); rebuild with splits/make_split.py")
    return np.flatnonzero(role.values == "train"), np.flatnonzero(role.values == "test")
