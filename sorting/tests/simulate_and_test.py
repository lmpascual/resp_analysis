"""End-to-end test on a simulated drifting Neuropixels recording with known spike trains.

Runs the full pipeline, then scores each sorter against the ground truth and checks whether
bombcell's 'good' label actually picks out well-isolated true units.

    python tests/simulate_and_test.py --workdir D:/tmp/npxsort_test [--duration 120] [--units 40] [--cpu]

Writes ground_truth_report.json in the workdir. Takes minutes on a GPU machine.

    python tests/simulate_and_test.py --workdir D:/tmp/npxsort_test --test remove_spikes

checks curation.remove_spikes without running a sorter (the ground truth with planted contamination
stands in for Kilosort's output) and writes remove_spikes_report.json; exits 1 if a check fails.
"""
import argparse
import gc
import json
import shutil
import sys
import time
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
    p.add_argument("--test", default="ground_truth", choices=["ground_truth", "remove_spikes"])
    args = p.parse_args()

    import spikeinterface.core as si
    import spikeinterface.comparison as sc
    from npxsort import load_config, run_pipeline

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if args.test == "remove_spikes":
        report = remove_spikes_test(workdir, min(args.duration, 60.0), args.n_jobs, args.seed)
        sys.exit(0 if all(report["checks"].values()) else 1)
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


def plant_contamination(gt, fs: float):
    """Turn a ground-truth sorting into a 'sorter output' with two kinds of contamination: unit B also
    gets unit A's spikes in 10-20 s (one event in two units, as KS4 #435 had #428's), and unit D's spikes
    in 30-35 s go to unit C instead. Planted spikes are >= 1 ms from the receiving unit's own spikes.
    Returns (sorting with unit ids 0..n-1, {A, B, C, D}, planted spike samples)."""
    import spikeinterface.core as si

    trains = {i: gt.get_unit_spike_train(u).astype(np.int64) for i, u in enumerate(gt.unit_ids)}
    loc = np.asarray(gt.get_property("gt_unit_locations"))[:, :2]
    dist = np.linalg.norm(loc[:, None] - loc[None, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    a, b = np.unravel_index(np.argmin(dist), dist.shape)  # closest pair
    dist[[a, b], :] = np.inf
    dist[:, [a, b]] = np.inf
    c, d = np.unravel_index(np.argmin(dist), dist.shape)  # next closest pair

    def away(src, dst, t0, t1):
        s = src[(src >= t0 * fs) & (src < t1 * fs)]
        j = np.searchsorted(dst, s)
        gap = np.minimum(np.abs(s - dst[np.clip(j - 1, 0, dst.size - 1)]), np.abs(dst[np.clip(j, 0, dst.size - 1)] - s))
        return s[gap >= int(0.001 * fs)]

    duplicated = away(trains[a], trains[b], 10, 20)
    moved = away(trains[d], trains[c], 30, 35)
    trains[b] = np.sort(np.r_[trains[b], duplicated])
    trains[c] = np.sort(np.r_[trains[c], moved])
    trains[d] = np.setdiff1d(trains[d], moved)
    units = {"A": int(a), "B": int(b), "C": int(c), "D": int(d)}
    return si.NumpySorting.from_unit_dict(trains, sampling_frequency=fs), units, {"duplicated": duplicated, "moved": moved}


def remove_spikes_test(workdir: Path, duration: float, n_jobs: int, seed: int) -> dict:
    """curation.remove_spikes end to end: a run without rules, the review browser's flags on it, then
    --redo postprocess with three rules. The removed spikes must be exactly those the rules select (with
    the browser's flags from the first run), and gone from units.tsv, the analyzer, ALF, Phy and the
    browser, while every other spike stays."""
    import pandas as pd
    import spikeinterface.core as si
    from npxsort import ConfigError, load_config, run_pipeline
    from npxsort.pipeline import Stages
    from npxsort.remove import remove_spikes
    from npxsort.review import ReviewData, SorterData

    root = workdir / "remove_spikes"
    shutil.rmtree(root, ignore_errors=True)
    fs = 30000.0
    rec, gt = si.generate_ground_truth_recording(
        durations=[duration], sampling_frequency=fs, num_channels=64, num_units=12,
        generate_sorting_kwargs={"firing_rates": 10.0, "refractory_period_ms": 4.0}, seed=seed,
    )
    rec.save(folder=root / "recording", format="binary", n_jobs=1)
    planted, u, plants = plant_contamination(gt, fs)
    name = "kilosort4"  # the planted sorting stands in for Kilosort's output; no sorter runs
    out = root / "out"
    planted.save(folder=out / name / "sorting_raw")
    Stages(out).mark(f"sort_{name}")

    cfg_dict = {
        "experiment_name": "remove_spikes_test",
        "paths": {"output_dir": str(out), "scratch_dir": str(root / "scratch")},
        "input": {"source": "si_folder", "folder": str(root / "recording")},
        "preprocessing": {"bad_channels": {"detect": False}, "save_dtype": "float32"},
        "motion": {"enabled": False},
        "sorters": {"kilosort4": {"enabled": True}, "lupin": {"enabled": False}},
        "job_kwargs": {"n_jobs": n_jobs, "progress_bar": False},
        "export": {"phy": True, "alf": True, "report": False},
    }
    cfg_path = root / "config.yaml"

    def run(redo=None) -> dict:
        cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
        cfg = load_config(cfg_path)
        run_pipeline(cfg, redo=redo)
        gc.collect()  # release the analyzer's memory maps so the rerun can overwrite it (Windows)
        return cfg

    # 1. no rules; what does the review browser flag?
    run()
    unit_dir = out / name
    table_before = pd.read_csv(unit_dir / "units.tsv", sep="\t", index_col="unit_id")
    data = ReviewData(out)
    t = time.time()
    while not data.flags_ready():
        if time.time() - t > 900:
            raise RuntimeError("The review browser's flags were not ready after 15 min.")
        time.sleep(1)
    s = data.sorters[name]
    before = {int(v): s.spikes[v].copy() for v in s.unit_ids}
    flags = {v: s.unit_flags(v) for v in (u["B"], u["C"])}
    del data, s
    gc.collect()

    w_dup, w_all, w_moved = (10.0, 20.0), (40.0, 41.0), (30.0, 35.0)

    def inside(x, w):
        return (x >= w[0] * fs) & (x < w[1] * fs)

    def flagged(v, key):
        m = np.zeros(before[v].size, dtype=bool)
        m[flags[v][key]] = True
        return m

    expected = {
        u["B"]: before[u["B"]][inside(before[u["B"]], w_dup) & flagged(u["B"], "dup")],
        u["A"]: before[u["A"]][inside(before[u["A"]], w_all)],
        u["C"]: before[u["C"]][inside(before[u["C"]], w_moved) & flagged(u["C"], "mismatch")],
    }

    # 2. the same run with three rules
    cfg_dict["curation"] = {"remove_spikes": [
        {"sorter": name, "unit_id": u["B"], "start_s": w_dup[0], "end_s": w_dup[1],
         "reason": "unit A's spikes also assigned to B", "only_flagged": ["duplicate"]},
        {"sorter": name, "unit_id": u["A"], "start_s": w_all[0], "end_s": w_all[1],
         "reason": "every spike in a 1-s window", "expect_removed": int(expected[u["A"]].size)},
        {"sorter": name, "unit_id": u["C"], "start_s": w_moved[0], "end_s": w_moved[1],
         "reason": "unit D's spikes assigned to C", "only_flagged": ["footprint_mismatch"]},
    ]}
    cfg = run(redo="postprocess")

    rep = json.loads((unit_dir / "postprocessing_report.json").read_text())["removed_spikes"]
    removed = pd.read_csv(unit_dir / "removed_spikes.tsv", sep="\t")
    table = pd.read_csv(unit_dir / "units.tsv", sep="\t", index_col="unit_id")
    analyzer = si.load_sorting_analyzer(unit_dir / "analyzer", load_extensions=False)
    after = {int(v): analyzer.sorting.get_unit_spike_train(v).astype(np.int64) for v in analyzer.unit_ids}
    del analyzer
    alf_t, alf_c = np.load(unit_dir / "alf" / "spike_times.npy"), np.load(unit_dir / "alf" / "spike_clusters.npy")
    phy = unit_dir / "phy"
    phy_t, phy_c = np.load(phy / "spike_times.npy").ravel(), np.load(phy / "spike_clusters.npy").ravel()
    phy_ids = pd.read_csv(phy / "cluster_si_unit_ids.tsv", sep="\t").set_index("cluster_id")["si_unit_id"]
    phy_removed = pd.read_csv(phy / "cluster_removed_spikes.tsv", sep="\t").set_index("cluster_id")["removed_spikes"]
    browser = SorterData(name, out, None)

    want = {v: np.setdiff1d(sp, expected.get(v, []), assume_unique=True) for v, sp in before.items()}
    n_cut = {v: before[v].size - want[v].size for v in before}
    untouched = [v for v in before if v not in expected]
    # Side effects on the units no rule touched (e.g. PCA metrics: the per-channel PCA is refitted)
    numeric = [c for c in table_before.columns if c in table.columns and pd.api.types.is_numeric_dtype(table_before[c])]
    x0, x1 = table_before.loc[untouched, numeric].astype(float), table.loc[untouched, numeric].astype(float)
    rel_change = ((x1 - x0).abs() / x0.abs().where(x0.abs() > 0)).max()
    changed = {c: round(float(v), 6) for c, v in rel_change.items() if v > 1e-9}
    labels_changed = [v for v in untouched if table_before.at[v, "label"] != table.at[v, "label"]]

    checks = {
        "every planted duplicate in B is flagged and removed": bool(np.isin(plants["duplicated"], expected[u["B"]]).all()),
        "the mismatch rule removes only D's spikes from C": bool(expected[u["C"]].size and np.isin(expected[u["C"]], plants["moved"]).all()),
        "postprocessing_report.json counts per rule": [r["n_removed"] for r in rep] == [int(expected[v].size) for v in (u["B"], u["A"], u["C"])],
        "removed_spikes.tsv lists exactly those spikes": len(removed) == sum(e.size for e in expected.values()) and all(
            np.array_equal(np.sort(removed.loc[removed["unit_id"] == v, "sample_index"].to_numpy()), e) for v, e in expected.items()),
        "analyzer: those spikes gone, all others kept": set(after) == set(before) and all(np.array_equal(after[v], want[v]) for v in before),
        "units.tsv num_spikes and removed_spikes": all(
            table.at[v, "num_spikes"] == want[v].size and table.at[v, "removed_spikes"] == n_cut[v] for v in before),
        "ALF spike_times / spike_clusters": all(np.array_equal(np.round(alf_t[alf_c == v] * fs).astype(np.int64), want[v]) for v in before),
        "Phy spike_times / spike_clusters": all(
            np.array_equal(np.sort(phy_t[phy_c == k].astype(np.int64)), want[int(phy_ids[k])]) and phy_removed[k] == n_cut[int(phy_ids[k])]
            for k in phy_ids.index),
        "review browser": all(np.array_equal(browser.spikes[v], want[v]) for v in before),
    }

    # 3. bad rules are refused
    ok_rule = {"sorter": name, "unit_id": u["A"], "start_s": w_all[0], "end_s": w_all[1], "reason": "test"}

    def config_refuses(rule) -> bool:
        p = root / "bad_config.yaml"
        p.write_text(yaml.safe_dump({**cfg_dict, "curation": {"remove_spikes": [rule]}}, sort_keys=False))
        try:
            load_config(p)
        except ConfigError:
            return True
        return False

    def run_refuses(rule) -> bool:
        try:
            remove_spikes(planted, si.load(root / "scratch" / "preprocessed"),
                          {**cfg, "curation": {**cfg["curation"], "remove_spikes": [rule]}}, name, {})
        except ValueError:
            return True
        return False

    checks.update({
        "config refuses an unknown key": config_refuses({**ok_rule, "unit": 3}),
        "config refuses an unknown flag": config_refuses({**ok_rule, "only_flagged": ["mismatch"]}),
        "config refuses a rule without reason": config_refuses({k: v for k, v in ok_rule.items() if k != "reason"}),
        "config refuses end_s <= start_s": config_refuses({**ok_rule, "end_s": w_all[0]}),
        "run stops when expect_removed differs": run_refuses({**ok_rule, "expect_removed": int(expected[u["A"]].size) + 1}),
        "run stops on a unit that does not exist": run_refuses({**ok_rule, "unit_id": 999}),
    })

    report = {
        "units": u, "planted": {k: int(v.size) for k, v in plants.items()},
        "moved_spikes_flagged_as_mismatch": f"{int(np.isin(plants['moved'], expected[u['C']]).sum())} of {plants['moved'].size}",
        "removed_spikes": rep,
        "untouched_units": {"max_relative_change_by_column": changed, "labels_changed": labels_changed},
        "checks": checks,
    }
    (workdir / "remove_spikes_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
