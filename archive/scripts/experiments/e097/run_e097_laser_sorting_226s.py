#!/usr/bin/env python3
"""
Continuous 226-second Lupin Sliding-Window Spike Sorting & Recruitment Pipeline for e097
Window: t = 796.827s to 1023.070s (Pre-laser baseline, 3 laser trains, and post-laser recovery)
Slices all 384 channels (13 overlapping 50-channel windows)
"""
import sys
import os
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spikeinterface.full as si
from src.pipeline.config import PipelineConfig
from src.pipeline.io.io_catgt import run_catgt
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.sorting.sliding_window_runner import run_sliding_window_sorting
from src.pipeline.sorting.sliding_window_consensus import compute_sliding_window_consensus
from scripts.plot_interactive_sorter_raster import build_interactive_sorter_raster

# Configure logging
LOG_DIR = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "sorting_execution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("E097_226S_SORTING")

def main():
    config_path = PROJECT_ROOT / "pipeline_config_e097_prelaser_through_laser3_226s.json"
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = PipelineConfig.from_dict(cfg_dict)

    raw_dir = Path(cfg.raw_data_dir)
    dur_s = cfg.duration_s
    start_s = cfg.start_time_s
    dest_scratch = Path(cfg.catgt.dest_dir)
    dest_scratch.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 85)
    logger.info(" STARTING E097 226s CONTINUOUS LUPIN SLIDING-WINDOW SPIKE SORTING PIPELINE")
    logger.info(f" Window: {start_s:.3f}s to {start_s + dur_s:.3f}s ({dur_s:.3f}s duration)")
    logger.info(f" Raw Directory: {raw_dir}")
    logger.info(f" CatGT Scratch: {dest_scratch}")
    logger.info(f" Output Root  : {LOG_DIR}")
    logger.info(f" QC Presence Ratio: {cfg.sorters.qc_presence_ratio}")
    logger.info("=" * 85)

    # 1. CatGT Preprocessing
    t_start_catgt = time.time()
    logger.info("\n>>> STAGE 1: EXECUTING CATGT PREPROCESSING")
    catgt_bin, catgt_meta = run_catgt(
        config=cfg.catgt,
        raw_dir=raw_dir,
        dest_dir=dest_scratch,
        start_secs=start_s,
        max_secs=dur_s
    )
    t_catgt = time.time() - t_start_catgt
    logger.info(f"CatGT completed in {t_catgt:.1f}s ({t_catgt/60:.2f} min). AP binary: {catgt_bin}")

    # 2. Ingest Preprocessed AP Data
    logger.info("\n>>> STAGE 2: INGESTING PREPROCESSED AP DATA VIA SPIKEINTERFACE")
    recording = read_spikeglx_ap_safe(catgt_bin.parent)
    logger.info(f"Loaded recording: {recording.get_num_channels()} channels, {recording.sampling_frequency:.1f} Hz, {recording.get_total_duration():.2f}s duration")

    # 3. Sliding-Window Sorting across all 13 windows
    out_dir = LOG_DIR / "lupin"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n>>> STAGE 3: EXECUTING 13 SLIDING-WINDOW SORTING JOBS (LUPIN / SC2)")
    t_sort_start = time.time()
    window_results = run_sliding_window_sorting(
        recording=recording,
        config=cfg,
        out_dir=out_dir / "windows",
        sorter_name="lupin",
        max_windows=None
    )
    t_sort = time.time() - t_sort_start
    logger.info(f"All sliding windows completed in {t_sort/60:.1f} minutes.")

    # 4. Cross-Window Agreement Consensus & Allen QC
    logger.info("\n>>> STAGE 4: CROSS-WINDOW CONSENSUS & ALLEN QC EVALUATION")
    consensus_res = compute_sliding_window_consensus(
        window_results=window_results,
        recording=recording,
        out_dir=out_dir / "consensus_master",
        agreement_threshold=cfg.sliding_window.agreement_threshold,
        sorter_name="lupin",
        pipeline_config=cfg
    )
    master_dir = out_dir / "consensus_master"

    # 5. Build 5-Panel Interactive HTML Raster
    logger.info("\n>>> STAGE 5: GENERATING SYNCHRONIZED INTERACTIVE MASTER RASTER")
    try:
        build_interactive_sorter_raster(
            sorter_dir=master_dir,
            sorter_name="Lupin (Sliding-Window Consensus Master 226s)"
        )
    except Exception as e:
        logger.warning(f"Could not build master interactive raster: {e}")

    # 6. Post-Sorting Population Recruitment & Epoch Rate Analysis
    logger.info("\n>>> STAGE 6: COMPUTING POPULATION RECRUITMENT & ROSTRAL ACTIVATION")
    analyze_population_recruitment(master_dir, start_s, dur_s)

    total_time = time.time() - t_start_catgt
    logger.info("=" * 85)
    logger.info(f" [COMPLETE] E097 226S SORTING PIPELINE FINISHED IN {total_time/60:.1f} MINUTES!")
    logger.info(f" Master Output: {master_dir.resolve()}")
    logger.info("=" * 85)

def analyze_population_recruitment(master_dir: Path, t_global_start: float, duration_s: float):
    """
    Computes epoch-by-epoch firing rates for all isolated units across:
    1. Pre-laser baseline: 0 to 60s
    2. Laser Train 1: 60.0 to 61.955s
    3. Inter-train 1: 61.955 to 119.664s
    4. Laser Train 2: 119.664 to 121.619s
    5. Inter-train 2: 121.619 to 164.288s
    6. Laser Train 3: 164.288 to 166.243s
    7. Post-laser recovery: 166.243 to 226.243s
    """
    alf_dir = master_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    qc_df = pd.read_csv(master_dir / "master_consensus_qc_summary.csv")

    epochs = {
        "pre_laser": (0.0, 60.0),
        "laser_train1": (60.0, 61.955),
        "inter_train1": (61.955, 119.664),
        "laser_train2": (119.664, 121.619),
        "inter_train2": (121.619, 164.288),
        "laser_train3": (164.288, 166.243),
        "post_laser": (166.243, 226.243),
    }

    records = []
    for _, row in qc_df.iterrows():
        cid = row["cluster_id"]
        u_spks = spk_times[spk_clusters == cid]

        rec = {
            "cluster_id": cid,
            "depth_um": row["depth_um"],
            "total_spikes": len(u_spks),
            "global_fr_hz": row["firing_rate_hz"],
            "isi_violations": row["isi_violations"],
            "presence_ratio": row["presence_ratio"],
            "passed_allen_qc": row["passed_allen_qc"],
            "is_consensus": row["is_cross_window_consensus"],
        }

        # Epoch rates
        for ep_name, (t_start, t_end) in epochs.items():
            dur = t_end - t_start
            n_spk = np.sum((u_spks >= t_start) & (u_spks < t_end))
            rec[f"fr_{ep_name}"] = round(n_spk / dur, 3)

        # Recruitment metric: max rate during any laser train vs baseline
        fr_pre = rec["fr_pre_laser"]
        max_laser_fr = max(rec["fr_laser_train1"], rec["fr_laser_train2"], rec["fr_laser_train3"])
        rec["max_laser_fr"] = max_laser_fr
        rec["delta_laser_fr"] = round(max_laser_fr - fr_pre, 3)

        # Flag newly recruited neurons (baseline dormant, awakened by laser)
        is_newly_recruited = (fr_pre < 0.2) and (max_laser_fr >= 3.0) and (row["isi_violations"] < 0.10)
        rec["is_newly_recruited"] = is_newly_recruited
        
        # Flag rostral zone
        rec["is_rostral_vrc"] = row["depth_um"] <= 1080.0

        records.append(rec)

    res_df = pd.DataFrame(records)
    out_csv = master_dir / "population_recruitment_summary_226s.csv"
    res_df.to_csv(out_csv, index=False)
    logger.info(f"Saved population recruitment summary to: {out_csv}")

    n_recruited = np.sum(res_df["is_newly_recruited"])
    n_rostral_recruited = np.sum(res_df["is_newly_recruited"] & res_df["is_rostral_vrc"])
    logger.info(f"Newly recruited units detected: {n_recruited} total ({n_rostral_recruited} in rostral VRC/BötC)")

if __name__ == "__main__":
    main()
