#!/usr/bin/env python3
"""
Master Sliding-Window Spike Sorting & Consensus Pipeline Runner
Executes:
1. (Optional) Interactive Configuration Wizard
2. Recording Ingestion & CatGT / Motion Verification
3. Iterative Sliding-Window Sorting across channel sub-bands
4. Cross-Window Unit Agreement Consensus & Deduplication
5. 5-Panel Synchronized Interactive HTML Raster & Dashboard
"""
import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import spikeinterface.full as si

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.config import PipelineConfig
from scripts.configure_pipeline_wizard import run_wizard
from src.pipeline.sorting.sliding_window_runner import run_sliding_window_sorting
from src.pipeline.sorting.sliding_window_consensus import compute_sliding_window_consensus
from scripts.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SLIDING_PIPELINE_RUNNER")

def main():
    parser = argparse.ArgumentParser(description="Sliding-Window Spike Sorting & Consensus Pipeline")
    parser.add_argument("--config", type=str, default="pipeline_config.json", help="Path to pipeline_config.json")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive configuration wizard first")
    parser.add_argument("--rec-dir", type=str, default=None, help="SpikeGLX raw/preprocessed recording directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Root directory for pipeline outputs")
    parser.add_argument("--sorter", type=str, default=None, help="Sorter name (lupin, mountainsort5, kilosort4)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds to sort (e.g. 60.0)")
    parser.add_argument("--start-time", type=float, default=None, help="Start time offset in seconds (e.g. 0.0)")
    parser.add_argument("--max-windows", type=int, default=None, help="Limit number of windows to process (for testing)")
    parser.add_argument("--no-motion", action="store_true", help="Bypass motion correction even if enabled")
    args = parser.parse_args()

    config_path = Path(args.config)

    # 1. Interactive Config Wizard (if requested or config missing)
    if args.interactive or not config_path.exists():
        cfg = run_wizard(config_path)
    else:
        import json
        with open(config_path, "r") as f:
            cfg = PipelineConfig.from_dict(json.load(f))
        logger.info(f"Loaded pipeline configuration from: {config_path.resolve()}")

    if args.sorter:
        cfg.sorters.enabled_sorters = [args.sorter.lower()]

    target_sorter = cfg.sorters.enabled_sorters[0] if cfg.sorters.enabled_sorters else "lupin"

    # 2. Ingest Preprocessed / Raw Recording
    rec_candidate = None
    if args.rec_dir:
        rec_candidate = Path(args.rec_dir)
    elif cfg.raw_data_dir and Path(cfg.raw_data_dir).exists():
        rec_candidate = Path(cfg.raw_data_dir)
    else:
        scratch_default = Path(r"C:\SCRATCH_neuropixels_test\catgt_output\catgt_LP21_fent_EXPT_g1")
        if scratch_default.exists():
            rec_candidate = scratch_default

    if rec_candidate is None or not rec_candidate.exists():
        raise FileNotFoundError(f"Could not locate recording directory. Please specify --rec-dir or set raw_data_dir in {config_path}")

    start_s = args.start_time if args.start_time is not None else (cfg.start_time_s or 0.0)
    dur_s = args.duration if args.duration is not None else (cfg.duration_s or (cfg.catgt.max_secs if cfg.catgt.enabled else None))

    # Check if CatGT Preprocessing should be run
    from src.pipeline.io.io_catgt import run_catgt
    from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe

    catgt_exe = Path(cfg.catgt.catgt_path)
    # Check if candidate is raw (contains .ap.bin without CatGT in parent path or filename)
    is_already_catgt = "catgt" in str(rec_candidate).lower() or any("catgt" in f.name.lower() for f in rec_candidate.glob("*.bin"))

    if cfg.catgt.enabled and catgt_exe.exists() and not is_already_catgt:
        dest_scratch = Path(cfg.catgt.dest_dir) if cfg.catgt.dest_dir else Path(r"C:\SCRATCH_neuropixels\catgt_output")
        dest_scratch.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 80)
        logger.info(f" EXECUTING CATGT PREPROCESSING (Filter: {cfg.catgt.apfilter}, loccar: {cfg.catgt.loccar_um}, duration: {dur_s}s)")
        logger.info("=" * 80)
        catgt_bin, catgt_meta = run_catgt(
            config=cfg.catgt,
            raw_dir=rec_candidate,
            dest_dir=dest_scratch,
            start_secs=start_s,
            max_secs=dur_s
        )
        logger.info(f"Loading CatGT preprocessed recording from: {catgt_bin.parent}")
        recording = read_spikeglx_ap_safe(catgt_bin.parent)
    else:
        logger.info(f"Loading SpikeGLX recording directly from: {rec_candidate}")
        recording = si.read_spikeglx(rec_candidate, stream_id="imec0.ap")
        if not is_already_catgt and (dur_s is not None or start_s > 0):
            fs = recording.sampling_frequency
            start_frame = int(start_s * fs)
            end_frame = int((start_s + dur_s) * fs) if dur_s is not None else recording.get_num_samples()
            end_frame = min(end_frame, recording.get_num_samples())
            recording = recording.frame_slice(start_frame=start_frame, end_frame=end_frame)
            logger.info(f"Sliced recording to {start_s:.2f}s - {start_s + recording.get_total_duration():.2f}s ({recording.get_total_duration():.2f}s total duration)")

    logger.info(f"Active recording for spike sorting: {recording.get_num_channels()} channels, {recording.sampling_frequency:.1f} Hz, {recording.get_total_duration():.2f}s")

    # 3. Output directories
    base_out = Path(args.output_dir) if args.output_dir else (cfg.output_dir / "sliding_window_pipeline")
    out_dir = base_out / target_sorter
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pipeline output directory set to: {out_dir.resolve()}")

    # 4. Execute Sliding Window Sorting
    window_results = run_sliding_window_sorting(
        recording=recording,
        config=cfg,
        out_dir=out_dir / "windows",
        sorter_name=target_sorter,
        max_windows=args.max_windows
    )

    if not window_results:
        logger.error("No windows completed sorting successfully.")
        return

    # 5. Compute Cross-Window Consensus Matching & Master Unit Assembly
    consensus_res = compute_sliding_window_consensus(
        window_results=window_results,
        recording=recording,
        out_dir=out_dir / "consensus_master",
        agreement_threshold=cfg.sliding_window.agreement_threshold,
        sorter_name=target_sorter,
        pipeline_config=cfg
    )

    # 6. Generate Synchronized 5-Panel Interactive HTML Raster & PNG Preview
    master_dir = out_dir / "consensus_master"
    try:
        build_interactive_sorter_raster(
            sorter_dir=master_dir,
            sorter_name=f"{target_sorter.capitalize()} (Sliding-Window Consensus Master)"
        )
    except Exception as e:
        logger.warning(f"Could not generate master interactive plots automatically: {e}")

    print("\n" + "=" * 85)
    print(f" [SUCCESS] SLIDING-WINDOW SPIKE SORTING & CONSENSUS COMPLETE ({target_sorter.upper()})")
    print(f" Windows Processed       : {len(window_results)}")
    print(f" Master Units Unified    : {consensus_res['n_total']}")
    print(f" Passing Single Units    : {consensus_res['n_sua']}")
    print(f" Cross-Window Confirmed  : {consensus_res['n_consensus']}")
    print(f" Run Journal Saved to    : {consensus_res.get('journal_path', master_dir)}")
    print(f" Master Output Folder    : {master_dir.resolve()}")
    print("=" * 85)

if __name__ == "__main__":
    main()
