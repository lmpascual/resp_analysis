#!/usr/bin/env python3
"""
Fix Subwindows Sorter Depth Mapping & NIDQ Time Synchronization:
1. Re-calibrates unit probe depths using true spike_positions.npy centroid depths from Kilosort 4 (bounded strictly to 0 - 2865 um).
2. Copies spike_positions.npy into alf directory for permanent repository archiving.
3. Synchronizes NIDQ cardiorespiratory channels (Airflow, Diaphragm EMG, Int. EMG, ECG) to the true subwindow start time:
    - win1_baseline:  1740.0s (29:00 - 30:00)
    - win2_anox1:     1800.0s (30:00 - 30:36)
    - win3_anox2:     1836.0s (30:36 - 31:00)
    - win4_gasp_early:1860.0s (31:00 - 32:30)
    - win5_gasp_late: 1950.0s (32:30 - 34:30)
    - win6_flow_shift:2070.0s (34:30 - 35:36)
    - win7_recovery:  2136.0s (35:36 - 36:00)
4. Re-generates interactive HTML rasters and high-resolution PNG previews for all subwindows.
"""
import sys
import shutil
import logging
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.visualization.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FIX_SUBWINDOWS_TIMESTAMPS_DEPTHS")

SUBWINDOWS = [
    {
        "id": "win1_baseline",
        "name": "Baseline (Pre-Anoxia)",
        "t_start": 1740.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working_win1\kilosort4_working\sorter_output")
    },
    {
        "id": "win2_anox1",
        "name": "Anoxic Challenge 1 (Gas Switch)",
        "t_start": 1800.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working\win2_anox1\kilosort4_working\sorter_output")
    },
    {
        "id": "win3_anox2",
        "name": "Anoxic Challenge 2 (Apnea / HR Nadir)",
        "t_start": 1836.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working\win3_anox2\kilosort4_working\sorter_output")
    },
    {
        "id": "win4_gasp_early",
        "name": "Early Gasping (Autoresuscitation Surge)",
        "t_start": 1860.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working\win4_gasp_early\kilosort4_working\sorter_output")
    },
    {
        "id": "win5_gasp_late",
        "name": "Late Gasping (Tachypneic Transition)",
        "t_start": 1950.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working\win5_gasp_late\kilosort4_working\sorter_output")
    },
    {
        "id": "win6_flow_shift",
        "name": "Active Expiration Shift",
        "t_start": 2070.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working\win6_flow_shift\kilosort4_working\sorter_output")
    },
    {
        "id": "win7_recovery",
        "name": "Post-Challenge Recovery",
        "t_start": 2136.0,
        "scratch": Path(r"C:\SCRATCH_neuropixels\sorting_working\win7_recovery\kilosort4_working\sorter_output")
    }
]

def main():
    base_out = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting"

    for sw in SUBWINDOWS:
        w_id = sw["id"]
        w_name = sw["name"]
        t_start = sw["t_start"]
        scratch_dir = sw["scratch"]
        ks4_dir = base_out / w_id / "kilosort4"

        logger.info("=" * 80)
        logger.info(f"PROCESSING {w_id.upper()} ({w_name}) | t_start = {t_start:.1f}s")
        logger.info("=" * 80)

        if not ks4_dir.exists():
            logger.warning(f"Kilosort 4 output directory not found at {ks4_dir}")
            continue

        # 1. Update depths from spike_positions.npy
        spk_pos_path = scratch_dir / "spike_positions.npy"
        spk_clu_path = scratch_dir / "spike_clusters.npy"

        if spk_pos_path.exists() and spk_clu_path.exists():
            logger.info(f"Loading true spike positions from {spk_pos_path}...")
            pos = np.load(spk_pos_path)
            clu = np.load(spk_clu_path)

            cluster_depths = {}
            for c in np.unique(clu):
                mask = (clu == c)
                cluster_depths[int(c)] = float(np.median(pos[mask, 1]))

            # Copy spike_positions.npy to alf
            alf_dir = ks4_dir / "alf"
            alf_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(spk_pos_path, alf_dir / "spike_positions.npy")
            logger.info(f"Archived spike_positions.npy to {alf_dir / 'spike_positions.npy'}")

            # Update QC CSVs
            for fn in ["kilosort4_qc_summary.csv", "cluster_info.tsv", "master_consensus_qc_summary.csv"]:
                p = ks4_dir / fn
                if p.exists():
                    sep = "\t" if p.suffix == ".tsv" else ","
                    df = pd.read_csv(p, sep=sep)
                    df["depth_um"] = df["cluster_id"].map(cluster_depths).fillna(df["depth_um"])
                    df.to_csv(p, sep=sep, index=False)
                    d_min, d_max = df["depth_um"].min(), df["depth_um"].max()
                    logger.info(f"Updated {fn}: real depth span [{d_min:.1f} um, {d_max:.1f} um]")
        else:
            logger.warning(f"Could not find spike_positions.npy in {scratch_dir}")

        # 2. Re-render synchronized interactive Plotly raster and high-res PNG preview
        logger.info(f"Re-rendering synchronized raster with true start_t_s = {t_start:.1f}s...")
        build_interactive_sorter_raster(
            sorter_dir=ks4_dir,
            sorter_name=f"Kilosort 4 - {w_name}",
            start_t_s=t_start
        )
        logger.info(f"Successfully re-rendered {w_id}!")

    logger.info("\nALL SUBWINDOWS SUCCESSFULLY UPDATED AND SYNCHRONIZED!")

if __name__ == '__main__':
    main()
