"""Electrolyte conductivity (Rahmanian et al., Sci. Data 2023, Zenodo 10.5281/zenodo.7244939): quick exploratory check.
Does ML beat a simple equation when predicting the conductivity of compositions it has never seen?
Target: ln(conductivity). Inputs: salt molality, EC/(EC+PC), (EC+PC)/(EC+PC+EMC), temperature.
Split: 5-fold by composition (all replicates and temperatures of a composition stay together).
Usage: python electrolyte/quick_check.py   (prints a summary)"""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
LIPF6 = 151.905  # g/mol


def load():
    d = pd.read_csv(ROOT / "data" / "conductivity.csv", sep=";", skiprows=[1, 2], low_memory=False)
    for c in ["temperature", "PC", "EC", "EMC", "LiPF_6", "EIS_conductivity"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    solv = d.PC + d.EC + d.EMC
    x = pd.DataFrame({"m": d.LiPF_6 / LIPF6 / (solv / 1000), "ecpc": d.EC / (d.EC + d.PC), "cyc": (d.EC + d.PC) / solv,
                      "T": d.temperature, "invT": 1000 / (d.temperature + 273.15), "y": np.log(d.EIS_conductivity),
                      "id": d.experimentID})
    x["comp"] = x[["m", "ecpc", "cyc"]].round(2).astype(str).agg("|".join, axis=1)
    return x.dropna().reset_index(drop=True)


def main():
    x = load()
    print(f"{len(x)} measurements, {x.id.nunique()} formulations, {x.comp.nunique()} distinct compositions")
    models = {
        "아레니우스 (조성 무시, 1/T만)": (["invT"], lambda: LinearRegression()),
        "단순 식: 2차 다항식 (조성 3개 + 1/T)": (["m", "ecpc", "cyc", "invT"], lambda: make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(1e-3))),
        "단순 식: 3차 다항식": (["m", "ecpc", "cyc", "invT"], lambda: make_pipeline(PolynomialFeatures(3), StandardScaler(), Ridge(1e-3))),
        "kNN (가까운 조성 5개 평균)": (["m", "ecpc", "cyc", "invT"], lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5))),
        "LightGBM": (["m", "ecpc", "cyc", "invT"], lambda: LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15, verbose=-1)),
    }
    rows = []
    for name, (cols, make) in models.items():
        p = np.zeros(len(x))
        for tr, te in GroupKFold(5).split(x, groups=x.comp):
            p[te] = make().fit(x.loc[tr, cols], x.y[tr]).predict(x.loc[te, cols])
        rel = np.abs(np.exp(p - x.y) - 1)  # relative error of conductivity
        rows.append({"모델": name, "R² (ln σ)": round(r2_score(x.y, p), 4), "MAE (ln σ)": round(mean_absolute_error(x.y, p), 4),
                     "상대 오차 중앙값": f"{np.median(rel):.1%}"})
    # replicate ceiling: predict each measurement by the mean of the other replicates of the same composition and temperature
    g = x.groupby(["comp", "T"]).y
    n, s = g.transform("count"), g.transform("sum")
    ok = n > 1
    loo = (s[ok] - x.y[ok]) / (n[ok] - 1)
    rel = np.abs(np.exp(loo - x.y[ok]) - 1)
    rows.append({"모델": "재현 한계 (같은 조성·온도의 다른 반복 평균)", "R² (ln σ)": round(r2_score(x.y[ok], loo), 4),
                 "MAE (ln σ)": round(mean_absolute_error(x.y[ok], loo), 4), "상대 오차 중앙값": f"{np.median(rel):.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
