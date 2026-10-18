"""End-to-end test on a simulated drifting Neuropixels recording with known spike trains.

Runs the full pipeline, then scores each sorter against the ground truth and checks whether
bombcell's 'good' label actually picks out well-isolated true units.

    python tests/simulate_and_test.py --workdir D:/tmp/npxsort_test [--duration 120] [--units 40] [--cpu]

Writes ground_truth_report.json in the workdir. Takes minutes on a GPU machine.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def simulate(workdir: Path, duration: float, n_units: int, seed: int):
    from spikeinterface.generation import generate_drifting_recording

    static, drifting, gt_sorting = generate_drifting_recording(
        num_units=n_units,
        duration=duration,
        probe_name="Neuropixels1-128",
        generate_displacement_vector_kwargs={
            "displacement_sampling_frequency": 5.0,
            "drift_start_um": [0, 15],
            "drift_stop_um": [0, -15],
            "drift_step_um": 1,
            "motion_list": [
                {"drift_mode": "zigzag", "non_rigid_gradient": None, "t_start_drift": 10.0, "t_end_drift": None, "period_s": 60}
            ],
        },
        generate_sorting_kwargs={"firing_rates": (3.0, 12.0), "refractory_period_ms": 4.0},
        seed=seed,
    )
    rec_folder = workdir / "simulated_recording"
    drifting.save(folder=rec_folder, format="binary", overwrite=True, n_jobs=1)
    gt_folder = workdir / "gt_sorting"
    gt_sorting.save(folder=gt_folder, overwrite=True)
    return rec_folder, gt_folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", required=True)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--units", type=int, default=40)
    p.add_argument("--cpu", action="store_true", help="run Kilosort 4 on CPU (slow)")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sorters", nargs="+", default=["kilosort4", "lupin"], choices=["kilosort4", "lupin"])
    args = p.parse_args()

    import spikeinterface.core as si
    import spikeinterface.comparison as sc
    from npxsort import load_config, run_pipeline

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    rec_folder, gt_folder = simulate(workdir, args.duration, args.units, args.seed)

    cfg_dict = {
        "experiment_name": "ground_truth_test",
        "paths": {"output_dir": str(workdir / "out"), "scratch_dir": str(workdir / "scratch")},
        "input": {"source": "si_folder", "folder": str(rec_folder)},
        "preprocessing": {"bad_channels": {"detect": False}, "save_dtype": "float32"},
        "motion": {"enabled": True, "preset": "dredge"},
        "sorters": {
            "kilosort4": {"enabled": "kilosort4" in args.sorters, "params": {"torch_device": "cpu" if args.cpu else "cuda"}},
            "lupin": {"enabled": "lupin" in args.sorters, "params": {}},
        },
        "job_kwargs": {"n_jobs": args.n_jobs, "progress_bar": False},
        "export": {"phy": True, "alf": True, "report": False},
    }
    cfg_path = workdir / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    cfg = load_config(cfg_path)
    manifest = run_pipeline(cfg)
    evaluate(workdir, cfg, manifest, args.duration, args.units)


def evaluate(workdir: Path, cfg: dict, manifest: dict, duration: float, n_units: int) -> dict:
    import pandas as pd
    import spikeinterface.core as si
    import spikeinterface.comparison as sc

    gt = si.load(workdir / "gt_sorting")
    report = {"duration_s": duration, "gt_units": n_units, "pipeline": manifest.get("label_counts"), "sorters": {}}
    out = Path(cfg["paths"]["output_dir"])
    for name in [n for n, s in cfg["sorters"].items() if s["enabled"]]:
        analyzer = si.load_sorting_analyzer(out / name / "analyzer")
        table = pd.read_csv(out / name / "units.tsv", sep="\t", index_col="unit_id")
        cmp = sc.compare_sorter_to_ground_truth(gt, analyzer.sorting, exhaustive_gt=True)
        perf = cmp.get_performance(method="by_unit")

        # Best agreement of each sorted unit with any GT unit: is a 'good' unit really one neuron?
        best = cmp.agreement_scores.max(axis=0)
        table["best_gt_agreement"] = best.reindex(table.index).to_numpy()
        by_label = table.groupby("label")["best_gt_agreement"].agg(["count", "median"]).round(3).to_dict(orient="index")
        good = table[table["label"] == "good"]

        report["sorters"][name] = {
            "units": int(len(table)),
            "gt_well_detected(>=0.8)": int(len(cmp.get_well_detected_units(well_detected_score=0.8))),
            "gt_accuracy_median": float(np.nanmedian(perf["accuracy"])),
            "gt_precision_median": float(np.nanmedian(perf["precision"])),
            "gt_recall_median": float(np.nanmedian(perf["recall"])),
            "false_positive_units": int(len(cmp.get_false_positive_units())),
            "redundant_units": int(len(cmp.get_redundant_units())),
            "overmerged_units": int(len(cmp.get_overmerged_units())),
            "good_units": int(len(good)),
            "good_units_matching_a_gt_unit(>=0.8)": int((good["best_gt_agreement"] >= 0.8).sum()),
            "best_gt_agreement_by_label": by_label,
        }
    report["comparison"] = manifest.get("comparison")
    (workdir / "ground_truth_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
