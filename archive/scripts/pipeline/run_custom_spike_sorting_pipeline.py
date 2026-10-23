#!/usr/bin/env python3
r"""
Master Customized Neuropixels Spike Sorting Pipeline
Incorporating CatGT Preprocessing, SpikeInterface Motion Estimation/Interpolation,
and Multi-Sorter Benchmarking (Kilosort 4, MountainSort 5, SpyKING-CIRCUS 2, Lupin)
Based on Garcia et al. 2026 (bioRxiv 2026.01.23.701239).
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.config import PipelineConfig, CatGTConfig, MotionConfig, SorterConfig, LupinConfig
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.io.io_catgt import run_catgt
from src.pipeline.motion.motion_corrector import run_stepwise_motion_pipeline
from src.pipeline.sorting.sorter_runner import run_multi_sorter_suite
from src.pipeline.sorting.comparator import compute_pairwise_agreement_matrix
from src.pipeline.diagnostics.preproc_inspector import plot_preprocessing_diagnostics
from src.pipeline.diagnostics.motion_inspector import plot_motion_diagnostics
from src.pipeline.diagnostics.sorting_inspector import plot_multi_sorter_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MASTER_SPIKE_PIPELINE")

def main():
    parser = argparse.ArgumentParser(
        description="Customized Neuropixels Spike Sorting Pipeline & Multi-Sorter Benchmarking"
    )
    # Directory & Duration Flags
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_imec0", help="Path to raw AP SpikeGLX directory")
    parser.add_argument("--out-dir", type=str, default=r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1", help="Path to final output directory")
    parser.add_argument("--scratch-dir", type=str, default=r"C:\SCRATCH_neuropixels", help="Path to fast local NVMe scratch directory")
    parser.add_argument("--duration", type=float, default=600.0, help="Duration in seconds to process (default: 600s = 10 min)")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time offset in seconds (default: 0.0s)")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "catgt", "motion", "sort", "inspect"], help="Pipeline stage to execute")
    
    # Sorter Selection
    parser.add_argument("--sorters", type=str, default="kilosort4,mountainsort5,lupin", help="Comma-separated list of sorters to run (e.g. kilosort4,mountainsort5,lupin,spykingcircus2,tridesclous2)")
    
    # CatGT Parameters
    parser.add_argument("--catgt-path", type=str, default=r"C:\Users\lpascu\SpikeGLX\CatGT-win\CatGT.exe", help="Path to CatGT executable")
    parser.add_argument("--no-catgt", action="store_true", help="Bypass CatGT and use SpikeInterface internal filtering")
    parser.add_argument("--apfilter", type=str, default="butter,12,300,9000", help="CatGT bandpass filter")
    parser.add_argument("--loccar", type=str, default="75,300", help="CatGT loccar_um inner,outer exclusion/inclusion radii in um")
    parser.add_argument("--no-t-shift", action="store_true", help="Disable CatGT ADC time-shift correction")
    
    # Motion Correction Parameters
    parser.add_argument("--motion-preset", type=str, default="kilosort_like", choices=["kilosort_like", "dredge", "nonrigid_fast_and_accurate", "nonrigid_accurate"], help="Motion estimation algorithm")
    parser.add_argument("--bin-size", type=float, default=2.0, help="Motion estimation temporal bin size in seconds (default: 2.0s)")
    parser.add_argument("--peak-thresh", type=float, default=5.5, help="Peak detection threshold (in SD units)")
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of CPU worker jobs")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    scratch_dir = Path(args.scratch_dir)
    diag_dir = out_dir / "diagnostics"

    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    loc_inner, loc_outer = map(float, args.loccar.split(","))
    sorter_list = [s.strip().lower() for s in args.sorters.split(",") if s.strip()]

    config = PipelineConfig(
        raw_data_dir=raw_dir,
        output_dir=out_dir,
        duration_s=args.duration,
        start_time_s=args.start_time,
        catgt=CatGTConfig(
            enabled=not args.no_catgt,
            catgt_path=args.catgt_path,
            t_shift=not args.no_t_shift,
            apfilter=args.apfilter,
            loccar_um=(loc_inner, loc_outer),
            start_secs=args.start_time,
            max_secs=args.duration,
            dest_dir=str(scratch_dir / "catgt_output")
        ),
        motion=MotionConfig(
            enabled=True,
            preset=args.motion_preset,
            peak_detect_threshold=args.peak_thresh,
            bin_duration_s=args.bin_size,
            scratch_dir=str(scratch_dir)
        ),
        sorters=SorterConfig(
            enabled_sorters=sorter_list,
            n_jobs=args.n_jobs
        ),
        lupin=LupinConfig(enabled=True)
    )

    # Save pipeline run manifest
    config.save_json(out_dir / "pipeline_config.json")

    print("\n" + "=" * 95)
    print(" CUSTOMIZED NEUROPIXELS SPIKE SORTING PIPELINE")
    print(f" Raw Dataset    : {raw_dir}")
    print(f" Target Duration: {args.duration:.1f}s ({args.duration/60:.1f} min)")
    print(f" Scratch NVMe   : {scratch_dir}")
    print(f" Output Folder  : {out_dir}")
    print(f" Target Sorters : {', '.join(sorter_list)}")
    print("=" * 95)

    # =========================================================================
    # STAGE 1: LOAD RAW RECORDING & PREPROCESSING (CatGT)
    # =========================================================================
    logger.info("Loading Raw SpikeGLX AP stream...")
    rec_raw = read_spikeglx_ap_safe(raw_dir)
    fs = rec_raw.get_sampling_frequency()
    total_dur_s = rec_raw.get_total_duration()

    # Slice recording if duration specified
    slice_dur = min(args.duration, total_dur_s) if args.duration else total_dur_s
    f0 = int(args.start_time * fs)
    f1 = int((args.start_time + slice_dur) * fs)
    rec_raw_sliced = rec_raw.frame_slice(start_frame=f0, end_frame=f1)

    if config.catgt.enabled and Path(config.catgt.catgt_path).exists():
        logger.info(f"Running CatGT Preprocessing on {slice_dur:.1f}s segment...")
        t0_catgt = time.time()
        catgt_bin, catgt_meta = run_catgt(
            config.catgt,
            raw_dir,
            dest_dir=Path(config.catgt.dest_dir),
            start_secs=args.start_time,
            max_secs=slice_dur
        )
        t_catgt_elapsed = time.time() - t0_catgt
        logger.info(f"CatGT completed in {t_catgt_elapsed:.1f}s.")
        rec_pre = read_spikeglx_ap_safe(catgt_bin.parent)
    else:
        logger.info("Applying internal SpikeInterface preprocessing (Phase Shift -> Bandpass -> Median CAR)...")
        import spikeinterface.preprocessing as spre
        rec_phase = spre.phase_shift(rec_raw_sliced)
        rec_filt = spre.bandpass_filter(rec_phase, freq_min=300.0, freq_max=9000.0)
        rec_pre = spre.common_reference(rec_filt, reference='global', operator='median')

    # Stage 1 Diagnostics
    fig_preproc = diag_dir / "stage1_catgt_preproc_diagnostics.png"
    plot_preprocessing_diagnostics(rec_raw_sliced, rec_pre, fig_preproc)
    print(f" [Stage 1] Preprocessing Diagnostic Saved: {fig_preproc}")

    if args.stage == "catgt":
        print("\n Target stage 'catgt' reached. Exiting.")
        return

    # =========================================================================
    # STAGE 2-4: SPIKEINTERFACE MOTION ESTIMATION & KRIGING INTERPOLATION
    # =========================================================================
    print("\n" + "=" * 90)
    print(" STAGE 2: MODULAR MOTION ESTIMATION & KRIGING INTERPOLATION")
    print(f" Preset: {config.motion.preset} | Bin Duration: {config.motion.bin_duration_s}s | Peak Thresh: {config.motion.peak_detect_threshold} SD")
    print("=" * 90)

    motion_scratch = scratch_dir / "motion_working"
    t0_motion = time.time()
    corrected_rec, motion_info = run_stepwise_motion_pipeline(
        recording=rec_pre,
        config=config.motion,
        working_dir=motion_scratch,
        n_jobs=args.n_jobs
    )
    t_motion_elapsed = time.time() - t0_motion
    logger.info(f"Motion correction pipeline finished in {t_motion_elapsed:.1f}s.")

    # Stage 2-4 Diagnostics
    fig_motion = diag_dir / "stage2_motion_drift_diagnostics.png"
    plot_motion_diagnostics(motion_info, rec_pre, fig_motion, corrected_rec=corrected_rec)
    print(f" [Stage 2] Motion & Drift Diagnostic Saved: {fig_motion}")

    if args.stage == "motion":
        print("\n Target stage 'motion' reached. Exiting.")
        return

    # =========================================================================
    # STAGE 5: PROGRAMMATIC MULTI-SORTER EXECUTION & LUPIN BENCHMARKING
    # =========================================================================
    print("\n" + "=" * 90)
    print(f" STAGE 3: MULTI-SORTER BENCHMARKING SUITE: {', '.join(sorter_list).upper()}")
    print("=" * 90)

    t0_sorting_all = time.time()
    sorter_results = run_multi_sorter_suite(
        sorter_names=sorter_list,
        recording=corrected_rec,
        base_out_dir=out_dir,
        scratch_dir=scratch_dir,
        config=config.sorters,
        lupin_config=config.lupin
    )
    t_sorting_elapsed = time.time() - t0_sorting_all

    # =========================================================================
    # STAGE 6: CROSS-SORTER COMPARISON, AGREEMENT MATRIX & FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 90)
    print(" STAGE 4: CROSS-SORTER AGREEMENT MATRIX & BENCHMARK DASHBOARD")
    print("=" * 90)

    qc_dataframes = {}
    alf_dirs = {}
    for s_name, res in sorter_results.items():
        qc_dataframes[s_name.upper()] = res["df_qc"]
        alf_dirs[s_name.upper()] = out_dir / s_name.lower() / "alf"

    agreement_matrix = None
    if len(alf_dirs) > 1:
        agreement_matrix = compute_pairwise_agreement_matrix(alf_dirs)
        agreement_matrix.to_csv(out_dir / "cross_sorter_pairwise_agreement_matrix.csv")
        print("\n Cross-Sorter Pairwise Agreement Matrix (Jaccard):")
        print(agreement_matrix)

    fig_benchmark = diag_dir / "stage3_multi_sorter_benchmark_dashboard.png"
    plot_multi_sorter_comparison(qc_dataframes, fig_benchmark, duration_s=slice_dur, agreement_matrix=agreement_matrix)
    print(f"\n [Stage 3] Multi-Sorter Benchmark Dashboard Saved: {fig_benchmark}")

    # Summary Report
    print("\n" + "=" * 95)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 95)
    for s_name, res in sorter_results.items():
        n_tot = res['n_units']
        n_qc = res['n_qc_passed']
        rt = res['runtime_s']
        print(f" * {s_name.upper():<16}: {n_tot:>3} units found | {n_qc:>3} passed Allen QC | Runtime: {rt:>5.1f}s")
    print(f" Total Sorting Runtime: {t_sorting_elapsed:.1f}s")
    print(f" Diagnostic Figures   : {diag_dir}")
    print("=" * 95)

if __name__ == "__main__":
    main()
