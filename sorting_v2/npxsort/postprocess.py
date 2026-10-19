"""Clean sorter output, build a SortingAnalyzer, compute real waveform-based quality metrics,
and label units with bombcell.

Every metric here is computed by SpikeInterface from the spikes and the preprocessed voltage;
nothing is estimated from spike counts.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import spikeinterface.core as si
import spikeinterface.curation as scur

from .extract import extract_units

log = logging.getLogger(__name__)

QUALITY_METRICS = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "amplitude_cutoff",
    "amplitude_median",
    "drift",
    "sd_ratio",
]
PCA_METRICS = ["mahalanobis", "d_prime"]  # mahalanobis -> isolation_distance, l_ratio


def clean_sorting(sorting: si.BaseSorting, rec: si.BaseRecording, cfg: dict, name: str) -> tuple[si.BaseSorting, dict]:
    pp = cfg["postprocessing"]
    report = {"units_from_sorter": int(len(sorting.unit_ids)), "spikes_from_sorter": int(sorting.count_total_num_spikes())}

    sorting = scur.remove_excess_spikes(sorting, rec)
    sorting = scur.remove_duplicated_spikes(sorting, censored_period_ms=pp["censor_duplicate_ms"], method="keep_first")
    sorting = sorting.remove_empty_units()
    report["spikes_after_duplicate_removal"] = int(sorting.count_total_num_spikes())

    if pp["remove_redundant_units"] and len(sorting.unit_ids) > 1:
        sorting, pairs = scur.remove_redundant_units(
            sorting,
            align=False,
            duplicate_threshold=pp["redundant_agreement"],
            remove_strategy="max_spikes",
            extra_outputs=True,
        )
        report["redundant_pairs"] = [[str(a), str(b)] for a, b in pairs]
    report["units_after_cleaning"] = int(len(sorting.unit_ids))
    log.info("%s cleaning: %s", name, {k: v for k, v in report.items() if k != "redundant_pairs"})
    return sorting, report


def build_analyzer(sorting: si.BaseSorting, rec: si.BaseRecording, cfg: dict, folder: Path, job_kwargs: dict):
    pp = cfg["postprocessing"]
    analyzer = si.create_sorting_analyzer(
        sorting,
        rec,
        format="binary_folder",
        folder=folder,
        sparse=True,
        sparsity_kwargs={"method": "radius", "radius_um": pp["sparsity_radius_um"]},
        overwrite=True,
        seed=pp["seed"],
        job_kwargs=job_kwargs,
    )
    steps = {
        "random_spikes": {"method": "uniform", "max_spikes_per_unit": pp["max_spikes_per_unit"], "seed": pp["seed"]},
        "waveforms": {"ms_before": pp["ms_before"], "ms_after": pp["ms_after"]},
        "templates": {"operators": ["average", "median", "std"]},
        "noise_levels": {},
        "spike_amplitudes": {},
        "unit_locations": {"method": "monopolar_triangulation"},
        "spike_locations": {},
        "correlograms": {"window_ms": 100.0, "bin_ms": 1.0},
        "isi_histograms": {"window_ms": 100.0, "bin_ms": 1.0},
        "template_similarity": {},
    }
    analyzer.compute(steps, **job_kwargs)

    metric_names = list(QUALITY_METRICS)
    if pp["pca_metrics"]:
        analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)
        metric_names += PCA_METRICS

    analyzer.compute("template_metrics", include_multi_channel_metrics=True, **job_kwargs)
    analyzer.compute("quality_metrics", metric_names=metric_names, **job_kwargs)
    return analyzer


def bombcell_thresholds(cfg: dict) -> dict:
    thresholds = copy.deepcopy(scur.bombcell_get_default_thresholds())
    for section, metrics in cfg["curation"]["threshold_overrides"].items():
        if section not in thresholds:
            raise ValueError(f"curation.threshold_overrides: unknown section '{section}' (noise | mua | non-somatic)")
        for metric, bounds in metrics.items():
            if bounds is None:  # explicit null removes the metric from labeling entirely
                thresholds[section].pop(metric, None)
            else:
                thresholds[section][metric] = {**thresholds[section].get(metric, {}), **bounds}
    return thresholds


def label_units(analyzer, cfg: dict, out_dir: Path) -> pd.DataFrame:
    thresholds = bombcell_thresholds(cfg)
    labels = scur.bombcell_label_units(sorting_analyzer=analyzer, thresholds=thresholds, split_non_somatic=False)
    col = "bombcell_label" if "bombcell_label" in labels.columns else labels.columns[0]
    labels = labels.rename(columns={col: "label"})[["label"]]

    metrics = analyzer.get_metrics_extension_data()
    bc_dir = out_dir / "bombcell"
    scur.save_bombcell_results(metrics, labels["label"].to_numpy(), thresholds, bc_dir)
    with open(bc_dir / "thresholds_used.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Labels: %s", labels["label"].value_counts().to_dict())
    return labels


def merge_suggestions(analyzer, labels: pd.DataFrame, job_kwargs: dict) -> pd.DataFrame:
    keep = labels.index[labels["label"] != "noise"].to_numpy()
    if len(keep) < 2:
        return pd.DataFrame(columns=["group", "unit_ids"])
    sub = analyzer.select_units(keep, format="memory")
    groups = scur.compute_merge_unit_groups(sub, preset="similarity_correlograms", resolve_graph=True, **job_kwargs)
    rows = [{"group": i, "unit_ids": " ".join(str(u) for u in g)} for i, g in enumerate(groups)]
    return pd.DataFrame(rows, columns=["group", "unit_ids"])


def unit_table(analyzer, labels: pd.DataFrame) -> pd.DataFrame:
    metrics = analyzer.get_metrics_extension_data()
    locs = analyzer.get_extension("unit_locations").get_data()
    peak_ch = si.get_template_extremum_channel(analyzer, peak_sign="both", outputs="id")
    peak_idx = si.get_template_extremum_channel(analyzer, peak_sign="both", outputs="index")
    table = pd.DataFrame(index=analyzer.unit_ids)
    table.index.name = "unit_id"
    table["label"] = labels.loc[analyzer.unit_ids, "label"].to_numpy()
    table["peak_channel_id"] = [peak_ch[u] for u in analyzer.unit_ids]
    table["peak_channel_index"] = [int(peak_idx[u]) for u in analyzer.unit_ids]
    table["x_um"] = locs[:, 0]
    table["depth_um"] = locs[:, 1]
    table = table.join(metrics)
    if "extracted_as" in analyzer.sorting.get_property_keys():
        table["extracted_as"] = analyzer.sorting.get_property("extracted_as")
    return table


def postprocess_sorter(name: str, sorting: si.BaseSorting, rec: si.BaseRecording, cfg: dict, job_kwargs: dict):
    out_dir = Path(cfg["paths"]["output_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    sorting, extract_report = extract_units(sorting, rec, cfg, name)
    sorting, clean_report = clean_sorting(sorting, rec, cfg, name)
    if extract_report:
        clean_report["extracted_units"] = extract_report
    analyzer = build_analyzer(sorting, rec, cfg, out_dir / "analyzer", job_kwargs)
    labels = label_units(analyzer, cfg, out_dir)
    table = unit_table(analyzer, labels)

    if cfg["curation"]["merge_suggestions"]:
        suggestions = merge_suggestions(analyzer, labels, job_kwargs)
        suggestions.to_csv(out_dir / "merge_suggestions.csv", index=False)
        clean_report["merge_suggestion_groups"] = int(len(suggestions))

    clean_report["label_counts"] = {k: int(v) for k, v in table["label"].value_counts().items()}
    with open(out_dir / "postprocessing_report.json", "w") as f:
        json.dump(clean_report, f, indent=2)
    table.to_csv(out_dir / "units.tsv", sep="\t")
    return analyzer, table
