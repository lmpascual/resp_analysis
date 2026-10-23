#!/usr/bin/env python3
"""
Master End-to-End Pipeline:
1. CatGT AP Preprocessing (re-run from raw SpikeGLX recording)
2. Spike Sorting:
   - Pipeline A: Sliding-Window Kilosort 4 (13 windows, 50ch window, 20ch overlap, consensus master)
   - Pipeline B: Global Full-Probe Kilosort 4 (single GPU pass across 384 channels)
3. Head-to-Head Comparative Metrics (Lupin vs Kilosort4-Sliding vs Kilosort4-Global)
4. Figure Recreation Suite (5-panel raster, TI-sorted rasters + gallery, Pub Figs 2 & 3, PETH summary, diagnostics)
5. Phy-Like Cluster Validation Suite (cluster_validation_browser.html + unit deep dives)
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.full as si

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import PipelineConfig, CatGTConfig, SlidingWindowConfig, SorterConfig, Kilosort4Config
from src.pipeline.io.io_catgt import run_catgt
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.sorting.sliding_window_runner import run_sliding_window_sorting
from src.pipeline.sorting.sliding_window_consensus import compute_sliding_window_consensus
from src.pipeline.sorting.sorter_runner import run_single_sorter
from scripts.visualization.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("E097_KS4_MASTER_PIPELINE")

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Kilosort4 Pipeline on e097_g2 (60s pre-laser)")
    parser.add_argument("--config", type=str, default="pipeline_config_e097_prelaser_60s_kilosort.json")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_imec0")
    parser.add_argument("--nidq-bin", type=str, default=r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    parser.add_argument("--run-mode", type=str, default="both", choices=["both", "sliding", "global"])
    parser.add_argument("--th-universal", type=float, default=9.0)
    parser.add_argument("--th-learned", type=float, default=8.0)
    parser.add_argument("--do-car", action="store_true", default=False, help="Enable internal CAR in Kilosort4 (default False since CatGT loccar was applied)")
    parser.add_argument("--nblocks", type=int, default=1, help="Drift correction blocks (1 = rigid)")
    parser.add_argument("--force-catgt", action="store_true", default=True, help="Force re-running CatGT")
    args = parser.parse_args()

    # Load base config
    cfg_file = REPO_ROOT / args.config
    import json
    with open(cfg_file, "r") as f:
        cfg = PipelineConfig.from_dict(json.load(f))

    cfg.kilosort4.Th_universal = args.th_universal
    cfg.kilosort4.Th_learned = args.th_learned
    cfg.kilosort4.do_CAR = args.do_car
    cfg.kilosort4.nblocks = args.nblocks

    raw_dir = Path(args.raw_dir)
    nidq_bin = Path(args.nidq_bin)
    out_root = REPO_ROOT / "outputs" / "e097" / "output_e097_prelaser_60s"
    out_root.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(cfg.catgt.dest_dir) if cfg.catgt.dest_dir else Path(r"C:\SCRATCH_neuropixels\catgt_output")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    start_s = cfg.start_time_s
    dur_s = cfg.duration_s

    # -------------------------------------------------------------
    # 1. CatGT AP Preprocessing
    # -------------------------------------------------------------
    logger.info("=" * 85)
    logger.info(" STEP 1: EXECUTING CATGT AP PREPROCESSING FROM RAW SPIKEGLX RECORDING")
    logger.info(f" Source: {raw_dir}")
    logger.info(f" Window: {start_s}s - {start_s + dur_s}s ({dur_s}s duration)")
    logger.info(f" Filter: {cfg.catgt.apfilter} | loccar_um: {cfg.catgt.loccar_um}")
    logger.info("=" * 85)

    catgt_bin, catgt_meta = run_catgt(
        config=cfg.catgt,
        raw_dir=raw_dir,
        dest_dir=scratch_dir,
        start_secs=start_s,
        max_secs=dur_s
    )
    logger.info(f"CatGT preprocessing complete: {catgt_bin}")
    recording = read_spikeglx_ap_safe(catgt_bin.parent)
    logger.info(f"Loaded preprocessed recording: {recording.get_num_channels()} channels, {recording.sampling_frequency:.1f} Hz, {recording.get_total_duration():.2f}s")

    results_summary = {}

    # -------------------------------------------------------------
    # 2. Pipeline A: Sliding-Window Kilosort 4
    # -------------------------------------------------------------
    sliding_master_dir = None
    if args.run_mode in ["both", "sliding"]:
        logger.info("\n" + "=" * 85)
        logger.info(" STEP 2A: SLIDING-WINDOW KILOSORT 4 SPIKE SORTING & CONSENSUS")
        logger.info("=" * 85)

        sw_out_dir = out_root / "kilosort4_sliding_window"
        sw_windows_dir = sw_out_dir / "windows"
        sw_master_dir = sw_out_dir / "consensus_master"
        sliding_master_dir = sw_master_dir

        t0_sw = time.time()
        window_results = run_sliding_window_sorting(
            recording=recording,
            config=cfg,
            out_dir=sw_windows_dir,
            sorter_name="kilosort4",
            max_windows=None
        )

        consensus_res = compute_sliding_window_consensus(
            window_results=window_results,
            recording=recording,
            out_dir=sw_master_dir,
            agreement_threshold=cfg.sliding_window.agreement_threshold,
            sorter_name="kilosort4",
            pipeline_config=cfg
        )
        t_sw = time.time() - t0_sw

        try:
            build_interactive_sorter_raster(
                sorter_dir=sw_master_dir,
                sorter_name="Kilosort 4 (Sliding-Window Consensus Master)"
            )
        except Exception as e:
            logger.warning(f"Could not build sliding raster: {e}")

        results_summary["kilosort4_sliding_window"] = {
            "runtime_s": t_sw,
            "total_units": consensus_res["n_total"],
            "sua_units": consensus_res["n_sua"],
            "cross_window_confirmed": consensus_res["n_consensus"],
            "dir": sw_master_dir
        }

    # -------------------------------------------------------------
    # 3. Pipeline B: Global Full-Probe Kilosort 4
    # -------------------------------------------------------------
    global_dir = None
    if args.run_mode in ["both", "global"]:
        logger.info("\n" + "=" * 85)
        logger.info(" STEP 2B: GLOBAL FULL-PROBE KILOSORT 4 SPIKE SORTING (384 CHANNELS)")
        logger.info("=" * 85)

        glob_out_dir = out_root / "kilosort4_global"
        global_dir = glob_out_dir
        scratch_glob = Path(r"C:\SCRATCH_neuropixels\kilosort4_global_working")

        t0_glob = time.time()
        sorting_glob, df_qc_glob, dur_glob = run_single_sorter(
            sorter_name="kilosort4",
            recording=recording,
            output_dir=out_root,
            scratch_dir=scratch_glob,
            config=cfg.sorters,
            kilosort4_config=cfg.kilosort4,
            custom_sorter_dir=glob_out_dir
        )

        df_qc_glob.to_csv(glob_out_dir / "master_consensus_qc_summary.csv", index=False)

        try:
            build_interactive_sorter_raster(
                sorter_dir=glob_out_dir,
                sorter_name="Kilosort 4 (Global Full-Probe 384ch)"
            )
        except Exception as e:
            logger.warning(f"Could not build global raster: {e}")

        results_summary["kilosort4_global"] = {
            "runtime_s": dur_glob,
            "total_units": len(df_qc_glob),
            "sua_units": int(df_qc_glob["passed_allen_qc"].sum()) if "passed_allen_qc" in df_qc_glob else len(df_qc_glob),
            "cross_window_confirmed": "N/A (Global)",
            "dir": glob_out_dir
        }

    # -------------------------------------------------------------
    # 4. Compare with Lupin Baseline
    # -------------------------------------------------------------
    lupin_journal = out_root / "lupin" / "consensus_master" / "run_journal.json"
    if lupin_journal.exists():
        with open(lupin_journal, "r") as f:
            lup_data = json.load(f)
        results_summary["lupin_baseline"] = {
            "runtime_s": "N/A",
            "total_units": lup_data["results"]["total_unified_units"],
            "sua_units": lup_data["results"]["passing_single_units_sua"],
            "cross_window_confirmed": lup_data["results"]["cross_window_confirmed_units"],
            "dir": out_root / "lupin" / "consensus_master"
        }

    print("\n" + "=" * 85)
    print(" [COMPARISON SUMMARY] LUPIN VS KILOSORT 4 SPIKE SORTING PIPELINES")
    print("=" * 85)
    df_cmp = pd.DataFrame(results_summary).T
    print(df_cmp[["total_units", "sua_units", "cross_window_confirmed", "runtime_s"]])
    df_cmp.to_csv(out_root / "sorter_pipeline_head_to_head_comparison.csv")
    print(f"Saved comparison table to: {out_root / 'sorter_pipeline_head_to_head_comparison.csv'}")

    # -------------------------------------------------------------
    # 5. Downstream Figures Recreation (Using Sliding-Window Kilosort 4 by default)
    # -------------------------------------------------------------
    target_cons_dir = sliding_master_dir if sliding_master_dir and (sliding_master_dir / "alf" / "spike_times.npy").exists() else global_dir
    if target_cons_dir:
        logger.info("\n" + "=" * 85)
        logger.info(" STEP 3: RECREATING PHYSIOLOGICAL & PUBLICATION FIGURES")
        logger.info(f" Target Consensus Output: {target_cons_dir}")
        logger.info("=" * 85)

        import subprocess
        python_exe = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

        figs_cmd = [
            str(python_exe),
            str(REPO_ROOT / "scripts" / "experiments" / "e097" / "run_e097_prelaser_figures.py"),
            "--sorter", "kilosort4",
            "--consensus-dir", str(target_cons_dir),
            "--raw-dir", str(raw_dir.parent),
            "--start-time", str(start_s),
            "--duration", str(dur_s)
        ]
        subprocess.run(figs_cmd, check=True)

        # -------------------------------------------------------------
        # 6. Cluster Validation Suite (mirroring LP21 sorting_validation)
        # -------------------------------------------------------------
        logger.info("\n" + "=" * 85)
        logger.info(" STEP 4: GENERATING INTERACTIVE CLUSTER VALIDATION SUITE")
        logger.info("=" * 85)

        val_cmd = [
            str(python_exe),
            str(REPO_ROOT / "scripts" / "visualization" / "generate_phy_like_cluster_validator.py"),
            "--output-dir", str(target_cons_dir),
            "--dest-dir", str(target_cons_dir.parent / "sorting_validation"),
            "--ap-bin", str(catgt_bin),
            "--nidq-bin", str(nidq_bin),
            "--insp-offset-source", "flow",
            "--n-channels", "8",
            "--max-units", "60",
            "--start-time", str(start_s),
            "--individual-pages"
        ]
        subprocess.run(val_cmd, check=True)

    logger.info("\n" + "=" * 85)
    logger.info(" [SUCCESS] END-TO-END KILOSORT 4 PIPELINE EXECUTION COMPLETED")
    logger.info("=" * 85)

if __name__ == "__main__":
    main()
