#!/usr/bin/env python3
"""
CLI tool to export any spike sorting output (Kilosort 4, Lupin, MountainSort 5, SpyKING-CIRCUS 2)
to a complete Phy GUI package with waveforms, templates, amplitudes, and parameters.
"""
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import spikeinterface.full as si
import spikeinterface.exporters as se_exp
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PHY_EXPORTER")

def export_sorter_to_phy(
    recording_path: Path,
    sorting_folder: Path,
    out_phy_dir: Path,
    n_jobs: int = 12,
    max_spikes_per_unit: int = 300
) -> Path:
    """
    Constructs a SortingAnalyzer and exports complete Phy files.
    """
    recording_path = Path(recording_path)
    sorting_folder = Path(sorting_folder)
    out_phy_dir = Path(out_phy_dir)
    out_phy_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading recording from: {recording_path}")
    if (recording_path / "traces_cached_seg0.raw").exists():
        rec = si.load(recording_path)
    else:
        rec = read_spikeglx_ap_safe(recording_path)

    logger.info(f"Loading sorting results from: {sorting_folder}")
    alf_dir = sorting_folder / "alf"
    if alf_dir.exists() and (alf_dir / "spike_times.npy").exists():
        # Load from ALF
        times = np.load(alf_dir / "spike_times.npy")
        clusters = np.load(alf_dir / "spike_clusters.npy")
        frames = (times * rec.get_sampling_frequency()).astype(np.int64)
        sorting = si.NumpySorting.from_samples_and_labels(
            samples_list=[frames],
            labels_list=[clusters],
            sampling_frequency=rec.get_sampling_frequency()
        )
    elif (sorting_folder / "spikeinterface_log.json").exists() or (sorting_folder / "spikes.npy").exists():
        sorting = si.load(sorting_folder)
    else:
        raise FileNotFoundError(f"Could not locate valid sorting output in {sorting_folder}")

    total_spikes = sorting.count_total_num_spikes() if hasattr(sorting, "count_total_num_spikes") else len(times)
    logger.info(f"Loaded Sorting: {len(sorting.get_unit_ids())} units, {total_spikes:,} spikes.")

    # Create SortingAnalyzer for fast waveform & template computation
    logger.info("Computing waveforms, templates, and amplitudes for Phy...")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec,
        format="memory",
        sparse=True
    )
    
    analyzer.compute({
        "random_spikes": {"max_spikes_per_unit": max_spikes_per_unit},
        "waveforms": {"ms_before": 1.5, "ms_after": 2.0},
        "templates": {},
        "noise_levels": {},
        "unit_locations": {},
        "spike_amplitudes": {},
    }, n_jobs=n_jobs, progress_bar=True)

    # Export to Phy format
    logger.info(f"Exporting to Phy directory: {out_phy_dir}")
    se_exp.export_to_phy(
        sorting_analyzer=analyzer,
        output_folder=out_phy_dir,
        compute_pc_features=False,
        compute_amplitudes=True,
        copy_binary=False,
        remove_if_exists=True,
        verbose=True
    )

    # Attach cluster_info.tsv if available
    tsv_source = sorting_folder / "cluster_info.tsv"
    if tsv_source.exists():
        import shutil
        shutil.copy2(tsv_source, out_phy_dir / "cluster_info.tsv")

    print("\n" + "=" * 85)
    print(" [SUCCESS] Exported to Phy GUI format!")
    print(f" Phy Directory: {out_phy_dir}")
    print("\n To launch Phy interactive GUI, run in terminal:")
    print(f"   cd '{out_phy_dir}'")
    print("   phy template-gui params.py")
    print("=" * 85)
    return out_phy_dir

def main():
    parser = argparse.ArgumentParser(description="Export any spike sorter output to Phy GUI")
    parser.add_argument("--rec-dir", type=str, required=True, help="Path to recording (preprocessed or drift-corrected folder)")
    parser.add_argument("--sorting-dir", type=str, required=True, help="Path to sorter output directory (e.g. output_LP21_fent_EXPT_g1/kilosort4)")
    parser.add_argument("--out-dir", type=str, default=None, help="Destination directory for Phy files (default: <sorting-dir>/phy)")
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of worker jobs")
    parser.add_argument("--max-spikes", type=int, default=300, help="Max spikes per unit for template waveforms")
    args = parser.parse_args()

    sort_dir = Path(args.sorting_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (sort_dir / "phy")
    export_sorter_to_phy(
        recording_path=Path(args.rec_dir),
        sorting_folder=sort_dir,
        out_phy_dir=out_dir,
        n_jobs=args.n_jobs,
        max_spikes_per_unit=args.max_spikes
    )

if __name__ == "__main__":
    main()
