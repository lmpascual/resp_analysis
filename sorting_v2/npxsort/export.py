"""Write outputs: unit table, ALF spike arrays, Phy folder with labels, per-unit report figures."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import spikeinterface.exporters as sexp

log = logging.getLogger(__name__)

PHY_GROUP = {"good": "good", "mua": "mua", "non_soma": "mua", "noise": "noise"}


def write_alf(analyzer, table: pd.DataFrame, out_dir: Path, t0_s: float) -> None:
    """spike_times.npy (s, from the start of the sorted window) and spike_clusters.npy, all units.
    Filter on units.tsv 'label' downstream. t0_s is written so times can be mapped to the original file."""
    alf = out_dir / "alf"
    alf.mkdir(parents=True, exist_ok=True)
    sorting = analyzer.sorting
    fs = sorting.get_sampling_frequency()
    vec = sorting.to_spike_vector()
    unit_ids = np.asarray(sorting.unit_ids)
    times = vec["sample_index"].astype(np.float64) / fs
    clusters = unit_ids[vec["unit_index"]]
    order = np.argsort(times, kind="stable")
    np.save(alf / "spike_times.npy", times[order])
    np.save(alf / "spike_clusters.npy", clusters[order])
    table.to_csv(alf / "units.tsv", sep="\t")
    (alf / "README.txt").write_text(
        "spike_times.npy: seconds from the start of the sorted window.\n"
        f"Add {t0_s:.6f} s to express them in the time base of the input file.\n"
        "spike_clusters.npy: unit id per spike (all units; filter with units.tsv 'label').\n"
    )


def write_phy(analyzer, table: pd.DataFrame, out_dir: Path, job_kwargs: dict) -> None:
    phy = out_dir / "phy"
    sexp.export_to_phy(
        analyzer,
        output_folder=phy,
        compute_pc_features=True,
        compute_amplitudes=True,
        copy_binary=False,  # params.py points at the preprocessed binary in scratch
        remove_if_exists=True,
        verbose=False,
        **job_kwargs,
    )
    # Phy's cluster ids are 0..N-1 in analyzer.unit_ids order (also saved as cluster_si_unit_ids.tsv).
    groups = pd.DataFrame(
        {"cluster_id": np.arange(len(analyzer.unit_ids)), "group": [PHY_GROUP.get(l, "unsorted") for l in table.loc[analyzer.unit_ids, "label"]]}
    )
    groups.to_csv(phy / "cluster_group.tsv", sep="\t", index=False)
    bc = pd.DataFrame({"cluster_id": np.arange(len(analyzer.unit_ids)), "bombcell": table.loc[analyzer.unit_ids, "label"].to_numpy()})
    bc.to_csv(phy / "cluster_bombcell.tsv", sep="\t", index=False)


def write_report(analyzer, out_dir: Path, job_kwargs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    sexp.export_report(analyzer, out_dir / "report", remove_if_exists=True, format="png", **job_kwargs)


def export_sorter(name: str, analyzer, table: pd.DataFrame, cfg: dict, t0_s: float, job_kwargs: dict) -> None:
    out_dir = Path(cfg["paths"]["output_dir"]) / name
    exp = cfg["export"]
    table.to_csv(out_dir / "units.tsv", sep="\t")
    if exp["alf"]:
        write_alf(analyzer, table, out_dir, t0_s)
    if exp["phy"]:
        write_phy(analyzer, table, out_dir, job_kwargs)
    if exp["report"]:
        write_report(analyzer, out_dir, job_kwargs)
    log.info("%s exports written to %s", name, out_dir)
