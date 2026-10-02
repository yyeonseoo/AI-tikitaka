"""Shared parsers that turn raw DB strings into model/EDA variables, plus a markdown table helper."""
import numpy as np
import pandas as pd

GAS = {"ar", "n2", "air", "dry air", "nitrogen", "argon"}


def md(t, index=True):  # markdown table without the tabulate dependency
    t = t.reset_index() if index else t
    rows = [list(map(str, t.columns))] + [list(map(str, r)) for r in t.values.tolist()]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                     + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def first_step(s):
    return s.str.split(">>").str[0].str.strip()


def top(s, n):
    keep = s.value_counts().index[:n]
    return s.where(s.isin(keep) | s.isna(), "other")


def dmso_frac(solvents, ratios):
    if not isinstance(solvents, str):
        return np.nan
    names = [s.strip() for s in solvents.split(">>")[0].split(";")]
    if len(names) == 1:
        return float(names[0] == "DMSO")
    r = pd.to_numeric(pd.Series(str(ratios).split(">>")[0].split(";")), errors="coerce")
    if len(r) != len(names) or r.isna().any() or r.sum() == 0:
        return np.nan
    return float(sum(v for n, v in zip(names, r) if n == "DMSO") / r.sum())


def antisolvent(media, used):
    if isinstance(media, str):
        m = media.split(">>")[0].strip()
        return "gas" if m.lower() in GAS else m
    return "used_unknown" if str(used) == "True" else "none_or_unreported"
