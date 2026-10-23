#!/usr/bin/env python3
"""
Dedicated runner for Lupin Spike Sorting on the intact 420s recording.
"""
import sys
import time
import shutil
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.sorting.sorter_runner import run_single_sorter
from src.pipeline.config import SorterConfig, LupinConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("INTACT_LUPIN_RUNNER")

def main():
    catgt_folder = Path(r"C:\SCRATCH_neuropixels\catgt_e097_intact_420s\catgt_e097_g2")
    out_dir = REPO_ROOT / "outputs" / "e097" / "intact_window"
    scratch_dir = Path(r"C:\SCRATCH_neuropixels\sorting_intact_window")

    rec = read_spikeglx_ap_safe(catgt_folder)
    logger.info(f"Loaded intact recording: {rec.get_num_channels()} chans, {rec.get_total_duration():.2f}s.")

    sorter_cfg = SorterConfig(
        enabled_sorters=["lupin"],
        do_CAR=False,
        save_phy=True,
        export_alf=True,
        isi_threshold_s=0.0015,
        qc_isi_cutoff=0.15,
        qc_presence_ratio=0.05,
        qc_amp_cutoff=0.25,
        qc_min_fr_hz=0.05,
        n_jobs=12
    )

    lupin_cfg = LupinConfig(
        enabled=True,
        sparsity_radius_um=100.0,
        clustering_method="circus",
        matching_method="circus-omp-svd",
        ms_before=1.5,
        ms_after=2.0,
        templates_from_svd=True,
        matching_amplitudes=[0.5, 2.2]
    )

    logger.info("Launching Lupin Spike Sorting...")
    t0 = time.time()
    sort_obj, df_qc, runtime = run_single_sorter(
        sorter_name="lupin",
        recording=rec,
        output_dir=out_dir,
        scratch_dir=scratch_dir,
        config=sorter_cfg,
        lupin_config=lupin_cfg,
        verbose=True
    )
    logger.info(f"Lupin completed in {runtime:.1f}s ({runtime/60:.1f} min). Units: {len(sort_obj.get_unit_ids())}")

if __name__ == '__main__':
    main()
