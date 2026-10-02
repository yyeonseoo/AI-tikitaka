"""Build splits/doi_split.csv: one train/test role per paper (DOI), shared by EDA, v2 and experiment 1.
Usage: python splits/make_split.py   (no model is trained or scored here)

- Papers in the v2 cleaned data keep the role they had in v2 (GroupShuffleSplit, test 20%, seed 0),
  so the v2 numbers already reported stay valid.
- Other MAPbI3 one-step papers (e.g. rows dropped by v2 cleaning, LLM-extracted papers) get a fixed
  hash-based role: test if sha1(doi) mod 5 == 0."""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import v2 as C
from models.v2.model_v2 import clean, make_features
from src.data import SPLIT_CSV, load_nomad, mapbi_onestep, split_by_paper


def hash_role(doi):
    return "test" if int(hashlib.sha1(doi.strip().lower().encode()).hexdigest(), 16) % 5 == 0 else "train"


def main():
    df, _ = clean()
    F = make_features(df)
    tr, te = split_by_paper(F["doi"], C.TEST_SIZE, C.SEED)
    v2 = pd.concat([pd.DataFrame({"doi": F.doi.iloc[tr].unique(), "role": "train"}),
                    pd.DataFrame({"doi": F.doi.iloc[te].unique(), "role": "test"})]).assign(source="v2 split (seed 0)")
    others = pd.Series(mapbi_onestep(load_nomad())["Ref_DOI_number"].dropna().unique())
    others = others[~others.isin(v2.doi)]
    extra = pd.DataFrame({"doi": others, "role": others.map(hash_role), "source": "sha1 mod 5"})
    out = pd.concat([v2, extra]).sort_values("doi")
    assert out.doi.is_unique
    SPLIT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(SPLIT_CSV, index=False)
    print(out.groupby(["source", "role"]).size().to_string())


if __name__ == "__main__":
    main()
