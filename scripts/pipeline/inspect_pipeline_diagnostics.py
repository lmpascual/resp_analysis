#!/usr/bin/env python3
"""
CLI Diagnostic Inspector:
Inspect the impact of key pipeline parameters across every processing stage:
1. Preprocessing & CatGT Filtering / Loccar (Traces, PSD, Channel Correlation)
2. Peak Detection & Motion Estimation (Peak Scatter, Displacement, Density)
3. Multi-Sorter Benchmarks & Agreement (Yields, SNR, ISI, Jaccard Matrix)
"""
import sys
import argparse
import pickle
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.diagnostics.preproc_inspector import plot_preprocessing_diagnostics
from src.pipeline.diagnostics.motion_inspector import plot_motion_diagnostics
from src.pipeline.diagnostics.sorting_inspector import plot_multi_sorter_comparison
from src.pipeline.sorting.comparator import compute_pairwise_agreement_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DIAGNOSTIC_INSPECTOR")

def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Pipeline Diagnostic Inspector")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "preproc", "motion", "sorting"], help="Diagnostic stage to generate")
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw SpikeGLX recording")
    parser.add_argument("--preproc-dir", type=str, default=None, help="Path to CatGT or preprocessed recording")
    parser.add_argument("--motion-pkl", type=str, default=None, help="Path to motion_info.pkl")
    parser.add_argument("--sorting-dir", type=str, default=None, help="Path to base sorting output directory containing sorter subfolders")
    parser.add_argument("--out-dir", type=str, default="./diagnostic_figures", help="Output directory for diagnostic plots")
    parser.add_argument("--duration", type=float, default=300.0, help="Recording duration in seconds")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessing Diagnostics
    if args.stage in ["all", "preproc"]:
        if args.raw_dir and args.preproc_dir:
            logger.info("Generating Preprocessing Diagnostics...")
            rec_raw = read_spikeglx_ap_safe(Path(args.raw_dir))
            rec_pre = read_spikeglx_ap_safe(Path(args.preproc_dir))
            p_fig = out_dir / "stage1_preprocessing_diagnostics.png"
            plot_preprocessing_diagnostics(rec_raw, rec_pre, p_fig)
            print(f" Preprocessing Diagnostic: {p_fig}")
        else:
            logger.warning("Skipping Preproc stage (requires both --raw-dir and --preproc-dir).")

    # 2. Motion Diagnostics
    if args.stage in ["all", "motion"]:
        if args.motion_pkl and Path(args.motion_pkl).exists():
            logger.info("Generating Motion & Drift Diagnostics...")
            with open(args.motion_pkl, "rb") as f:
                motion_info = pickle.load(f)
            rec_ref = read_spikeglx_ap_safe(Path(args.preproc_dir or args.raw_dir))
            m_fig = out_dir / "stage2_motion_drift_diagnostics.png"
            plot_motion_diagnostics(motion_info, rec_ref, m_fig)
            print(f" Motion Diagnostic: {m_fig}")
        else:
            logger.warning("Skipping Motion stage (requires valid --motion-pkl).")

    # 3. Sorting Diagnostics
    if args.stage in ["all", "sorting"]:
        if args.sorting_dir and Path(args.sorting_dir).exists():
            logger.info("Generating Multi-Sorter Benchmark & Agreement Diagnostics...")
            sort_root = Path(args.sorting_dir)
            qc_dfs = {}
            alf_dirs = {}
            for sub in sort_root.iterdir():
                if sub.is_dir() and (sub / "cluster_info.tsv").exists():
                    s_name = sub.name.upper()
                    qc_dfs[s_name] = pd.read_csv(sub / "cluster_info.tsv", sep="\t")
                    if (sub / "alf").exists():
                        alf_dirs[s_name] = sub / "alf"

            if qc_dfs:
                agree_mat = compute_pairwise_agreement_matrix(alf_dirs) if len(alf_dirs) > 1 else None
                s_fig = out_dir / "stage3_multi_sorter_benchmark_dashboard.png"
                plot_multi_sorter_comparison(qc_dfs, s_fig, duration_s=args.duration, agreement_matrix=agree_mat)
                print(f" Sorting Benchmark Dashboard: {s_fig}")
            else:
                logger.warning(f"No cluster_info.tsv found in subfolders of {sort_root}")
        else:
            logger.warning("Skipping Sorting stage (requires valid --sorting-dir).")

if __name__ == "__main__":
    main()
