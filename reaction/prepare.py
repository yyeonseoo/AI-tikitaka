"""Reaction-yield data loading and reagent features (plan: reaction/README.md).
load_bh(): Buchwald-Hartwig (3,955 reactions) with substrate split into halogen and arene core.
load_sm(): Suzuki-Miyaura (5,760 reactions), names only; a blank reagent slot means the reagent was not used.
descriptors(smiles): RDKit 2D descriptors + Morgan fingerprint bits for each distinct reagent (feature set F1)."""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")
DATA = Path(__file__).resolve().parent / "data"
BH_SLOTS = ["ligand", "base", "additive", "aryl_halide"]
SM_SLOTS = ["electrophile", "nucleophile", "ligand", "base", "solvent"]
HALOGEN_RANK = {"Cl": 0, "Br": 1, "I": 2}  # oxidative-addition reactivity order
DESC = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "NumAromaticRings",
        "FractionCSP3", "HeavyAtomCount", "LabuteASA", "MolMR"]
N_BITS = 64


def split_halide(smi):
    """Aryl halide -> (halogen symbol, canonical SMILES of the arene with the halogen replaced by a dummy atom '*',
    so 2- and 3-halopyridine stay different)."""
    mol = Chem.MolFromSmiles(smi)
    hal = [a for a in mol.GetAtoms() if a.GetSymbol() in HALOGEN_RANK]
    assert len(hal) == 1, smi
    symbol = hal[0].GetSymbol()
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(hal[0].GetIdx()).SetAtomicNum(0)
    return symbol, Chem.MolToSmiles(rw.GetMol())


def load_bh():
    d = pd.read_excel(DATA / "buchwald_hartwig.xlsx", "FullCV_01").rename(
        columns={"Ligand": "ligand", "Base": "base", "Additive": "additive", "Aryl halide": "aryl_halide", "Output": "y"})
    parts = d.aryl_halide.map(split_halide)
    d["halogen"], d["arene"] = parts.str[0], parts.str[1]
    d["halogen_rank"] = d.halogen.map(HALOGEN_RANK)
    return d.reset_index(drop=True)


def load_sm():
    s = pd.read_excel(DATA / "suzuki.xlsx", "Sheet1").rename(columns={
        "Reactant_1_Short_Hand": "electrophile", "Reactant_2_Name": "nucleophile", "Ligand_Short_Hand": "ligand",
        "Reagent_1_Short_Hand": "base", "Solvent_1_Short_Hand": "solvent", "Product_Yield_PCT_Area_UV": "y"})
    s = s.assign(**{c: s[c].fillna("없음").astype(str).str.strip() for c in SM_SLOTS})
    return s[SM_SLOTS + ["y"]].dropna(subset=["y"]).reset_index(drop=True)


def descriptors(smiles):
    """One row per distinct SMILES: RDKit 2D descriptors and Morgan (radius 2) bits; constant columns dropped."""
    rows = {}
    for smi in pd.unique(smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = np.zeros(N_BITS)
        for b in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS).GetOnBits():
            fp[b] = 1
        rows[smi] = [getattr(Descriptors, n)(mol) for n in DESC] + list(fp)
    D = pd.DataFrame.from_dict(rows, orient="index", columns=DESC + [f"fp{i}" for i in range(N_BITS)])
    return D.loc[:, D.nunique() > 1]


if __name__ == "__main__":
    bh, sm = load_bh(), load_sm()
    print("BH", bh.shape, {c: bh[c].nunique() for c in BH_SLOTS + ["halogen", "arene"]})
    print("SM", sm.shape, {c: sm[c].nunique() for c in SM_SLOTS})
    for c in BH_SLOTS:
        print(c, descriptors(bh[c]).shape)
