"""Cross-sorter agreement: which units did Kilosort 4 and Lupin both find?"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import spikeinterface.comparison as sc

log = logging.getLogger(__name__)


def compare_sorters(analyzers: dict, tables: dict, cfg: dict) -> dict:
    c = cfg["comparison"]
    out_dir = Path(cfg["paths"]["output_dir"]) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    (n1, n2) = list(analyzers)[:2]

    def kept(name):
        t = tables[name]
        return t.index[~t["label"].isin(c["exclude_labels"])].to_numpy()

    s1 = analyzers[n1].sorting.select_units(kept(n1))
    s2 = analyzers[n2].sorting.select_units(kept(n2))
    cmp = sc.compare_two_sorters(
        s1, s2, sorting1_name=n1, sorting2_name=n2,
        delta_time=c["delta_time_ms"], match_score=c["match_score"],
    )
    m12, m21 = cmp.get_matching()
    scores = cmp.agreement_scores

    def match_columns(matching, own_name, other_name, transpose):
        rows = []
        for unit, other in matching.items():
            score = np.nan
            if other != -1:
                score = scores.at[other, unit] if transpose else scores.at[unit, other]
            rows.append({"unit_id": unit, f"match_{other_name}": other if other != -1 else None, "agreement": score})
        df = pd.DataFrame(rows).set_index("unit_id")
        df["found_by_both"] = df["agreement"] >= c["match_score"]
        return df

    per_unit = {
        n1: match_columns(m12, n1, n2, transpose=False),
        n2: match_columns(m21, n2, n1, transpose=True),
    }
    for name, df in per_unit.items():
        df.to_csv(out_dir / f"{name}_matches.tsv", sep="\t")
    scores.to_csv(out_dir / "agreement_scores.csv")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import spikeinterface.widgets as sw

        w = sw.plot_agreement_matrix(cmp, backend="matplotlib")
        w.figure.set_size_inches(12, 10)
        w.figure.savefig(out_dir / "agreement_matrix.png", dpi=110)
        plt.close(w.figure)
    except Exception:
        log.exception("Could not draw the agreement matrix (results are still saved).")

    summary = {}
    for name in (n1, n2):
        t = tables[name].join(per_unit[name], how="left")
        good = t["label"] == "good"
        summary[name] = {
            "units_compared": int(len(per_unit[name])),
            "found_by_both": int(per_unit[name]["found_by_both"].sum()),
            "good_units": int(good.sum()),
            "good_and_found_by_both": int((good & t["found_by_both"].fillna(False).astype(bool)).sum()),
        }
    log.info("Comparison: %s", summary)
    return {"per_unit": per_unit, "summary": summary}
