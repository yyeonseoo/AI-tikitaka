"""Reaction-yield HTE datasets: quick exploratory check of headroom for ML over simple models.
Buchwald-Hartwig (Ahneman et al., Science 2018; 3,955 reactions) and Suzuki-Miyaura (Perera et al., Science 2018;
5,760 reactions), from github.com/rxn4chemistry/rxn_yields. Inputs: which reagent was used in each slot (categorical).
Models: average yield per reagent added up (linear, one-hot), linear with all pairwise reagent interactions,
LightGBM, random forest. Splits: random 5-fold, and 5-fold by one reagent slot (that reagent unseen in training).
Usage: python reaction/quick_check.py   (prints a summary)"""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

ROOT = Path(__file__).resolve().parent


def load():
    bh = pd.read_excel(ROOT / "data" / "buchwald_hartwig.xlsx", "FullCV_01")
    bh = bh.rename(columns={"Output": "y"})
    sm = pd.read_excel(ROOT / "data" / "suzuki.xlsx", "Sheet1")
    sm = sm.rename(columns={"Reactant_1_Short_Hand": "electrophile", "Reactant_2_Name": "nucleophile", "Ligand_Short_Hand": "ligand",
                            "Reagent_1_Short_Hand": "base", "Solvent_1_Short_Hand": "solvent", "Product_Yield_PCT_Area_UV": "y"})
    return {"Buchwald-Hartwig": (bh, ["Ligand", "Additive", "Base", "Aryl halide"], "Additive"),
            "Suzuki-Miyaura": (sm, ["electrophile", "nucleophile", "ligand", "base", "solvent"], "ligand")}


def models(cols):
    oh = lambda: OneHotEncoder(handle_unknown="ignore")
    return {
        "선형 (시약별 평균 더하기)": lambda: make_pipeline(oh(), Ridge(1.0)),
        "선형 + 시약 짝 상호작용": lambda: make_pipeline(oh(), PolynomialFeatures(2, interaction_only=True, include_bias=False), Ridge(1.0)),
        "LightGBM": lambda: LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, verbose=-1),
        "랜덤 포레스트": lambda: RandomForestRegressor(500, n_jobs=-1, random_state=0),
    }


def run(d, cols, make, splitter, groups=None):
    X = d[cols].astype(str)
    p = np.zeros(len(d))
    for tr, te in splitter.split(X, groups=groups):
        m = make()
        if isinstance(m, (LGBMRegressor, RandomForestRegressor)):
            Xc = X.astype("category")
            m.fit(Xc.iloc[tr] if isinstance(m, LGBMRegressor) else pd.get_dummies(X).iloc[tr], d.y.iloc[tr])
            p[te] = m.predict(Xc.iloc[te] if isinstance(m, LGBMRegressor) else pd.get_dummies(X).iloc[te])
        else:
            p[te] = m.fit(X.iloc[tr], d.y.iloc[tr]).predict(X.iloc[te])
    return r2_score(d.y, p), mean_absolute_error(d.y, p)


def main():
    for name, (d, cols, slot) in load().items():
        d = d.assign(**{c: d[c].fillna("없음") for c in cols}).dropna(subset=["y"]).reset_index(drop=True)  # blank = reagent not used
        print(f"\n== {name}: {len(d)} reactions; levels per slot: " + ", ".join(f"{c} {d[c].nunique()}" for c in cols)
              + f"; yield mean {d.y.mean():.1f}%")
        rows = []
        for mname, make in models(cols).items():
            r_rand = run(d, cols, make, KFold(5, shuffle=True, random_state=0))
            r_grp = run(d, cols, make, GroupKFold(5), d[slot])
            rows.append({"모델": mname, "무작위 R²": round(r_rand[0], 3), "무작위 MAE": round(r_rand[1], 1),
                         f"처음 보는 {slot} R²": round(r_grp[0], 3), f"처음 보는 {slot} MAE": round(r_grp[1], 1)})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
