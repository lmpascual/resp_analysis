"""
Master script to generate all figures for e097_g2 60s pre-laser window:
1. Breath-onset aligned rasters sorted by inspiration duration (T_I) + HTML gallery
2. Publication Figure 2 (Single-cell respiratory phase dynamics)
3. Publication Figure 3 (Low-dimensional 3D PCA neural manifold & interactive HTML)
4. Unit respiratory phase classification table (.csv)
5. All neurons PETH summary panel (mean + SD)
6. Breath detection diagnostic dashboard (EMG + Flow + Markers)
"""
import sys
import argparse
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("E097_FIGURES_RUNNER")

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

def main():
    parser = argparse.ArgumentParser(description="Generate Figures for e097_g2 60s pre-laser window")
    parser.add_argument("--sorter", type=str, default="kilosort4", help="Sorter name (e.g. kilosort4 or lupin)")
    parser.add_argument("--consensus-dir", type=str, default=None, help="Path to consensus_master directory containing alf/")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\budde\npx\animal_files\e097_g2", help="Path to raw animal recording folder")
    parser.add_argument("--out-rasters", type=str, default=None, help="Output directory for onset-aligned rasters & gallery")
    parser.add_argument("--out-pub", type=str, default=None, help="Output directory for publication figures & diagnostics")
    parser.add_argument("--start-time", type=float, default=796.827, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()

    sorter = args.sorter.lower().strip()
    raw_dir = Path(args.raw_dir)
    start_time = args.start_time
    duration = args.duration

    if args.consensus_dir:
        consensus_dir = Path(args.consensus_dir)
    else:
        cand1 = REPO_ROOT / "outputs" / "e097" / "output_e097_prelaser_60s" / sorter / "consensus_master"
        cand2 = REPO_ROOT / "output_e097_prelaser_60s" / sorter / "consensus_master"
        consensus_dir = cand1 if cand1.exists() else cand2

    if args.out_rasters:
        out_onset_rasters = Path(args.out_rasters)
    else:
        out_onset_rasters = REPO_ROOT / "outputs" / "e097" / f"output_individual_neuron_rasters_onset_sorted_insp_duration_e097_60s_{sorter}"

    if args.out_pub:
        out_pub_figs = Path(args.out_pub)
    else:
        out_pub_figs = REPO_ROOT / "outputs" / "e097" / f"e097_prelaser_respiratory_neurodynamics_{sorter}"

    out_onset_rasters.mkdir(parents=True, exist_ok=True)
    out_pub_figs.mkdir(parents=True, exist_ok=True)

    if not (consensus_dir / "alf" / "spike_times.npy").exists():
        raise FileNotFoundError(f"Consensus spike sorting output not found at: {consensus_dir / 'alf'}")

    logger.info("=" * 80)
    logger.info(f" STARTING FIGURE GENERATION PIPELINE FOR E097_G2 (60s PRE-LASER, SORTER: {sorter.upper()})")
    logger.info(f" Consensus Dir: {consensus_dir}")
    logger.info("=" * 80)

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # 1. Generate Breath-Onset Aligned Rasters Sorted by Inspiration Duration (T_I)
    gallery_file = out_onset_rasters / "gallery_all_37_neurons.html"
    if gallery_file.exists() and len(list(out_onset_rasters.glob("*.png"))) > 0:
        logger.info(f">>> 1. Breath-Onset Aligned Rasters already generated ({len(list(out_onset_rasters.glob('*.png')))} PNGs in {out_onset_rasters}). Skipping...")
    else:
        logger.info(">>> 1. Generating Breath-Onset Aligned Rasters Sorted by Inspiration Duration...")
        cmd1 = [
            str(PYTHON_EXE),
            str(REPO_ROOT / "scripts" / "visualization" / "plot_onset_rasters_sorted_by_insp_duration.py"),
            "--consensus-dir", str(consensus_dir),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out_onset_rasters),
            "--start-time", str(start_time),
            "--duration", str(duration)
        ]
        subprocess.run(cmd1, env=env, check=True)

    # 2. Replicate Publication Figures 2 & 3
    logger.info(">>> 2. Replicating Publication Figures 2 & 3...")
    cmd2 = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "visualization" / "replicate_publication_figures.py"),
        "--consensus-dir", str(consensus_dir),
        "--raw-dir", str(raw_dir),
        "--out-dir", str(out_pub_figs),
        "--start-time", str(start_time)
    ]
    subprocess.run(cmd2, env=env, check=True)

    # 3. All Neurons PETH Summary Panel
    logger.info(">>> 3. Generating All Neurons PETH Summary Panel...")
    cmd3 = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "visualization" / "plot_all_37_neurons_peth_with_sd.py"),
        "--consensus-dir", str(consensus_dir),
        "--raw-dir", str(raw_dir),
        "--out-dir", str(out_pub_figs),
        "--start-time", str(start_time),
        "--duration", str(duration)
    ]
    subprocess.run(cmd3, env=env, check=True)

    # 4. Breath Detection Diagnostic Dashboard
    logger.info(">>> 4. Generating Breath Detection Diagnostic Dashboard...")
    cmd4 = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "visualization" / "plot_breath_detection_diagnostics.py"),
        "--raw-dir", str(raw_dir),
        "--out-dir", str(out_pub_figs),
        "--start-time", str(start_time),
        "--duration", str(duration)
    ]
    subprocess.run(cmd4, env=env, check=True)

    logger.info("=" * 80)
    logger.info(f" [SUCCESS] ALL FIGURES & DASHBOARDS GENERATED FOR E097_G2 (60s PRE-LASER, {sorter.upper()})")
    logger.info(f" - Onset Rasters & Gallery: {out_onset_rasters}")
    logger.info(f" - Publication Figures & Diagnostics: {out_pub_figs}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
