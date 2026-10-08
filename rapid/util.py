def md(t, index=True):  # markdown table without the tabulate dependency
    t = t.reset_index() if index else t
    rows = [list(map(str, t.columns))] + [list(map(str, r)) for r in t.values.tolist()]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                     + ["| " + " | ".join(r) + " |" for r in rows[1:]])
