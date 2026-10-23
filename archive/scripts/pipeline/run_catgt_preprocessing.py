#!/usr/bin/env python3
"""
CLI entrypoint for standalone CatGT Preprocessing:
- ADC Phase Time-Shift Correction (-t_shift)
- Butterworth Bandpass Filter (default: butter,12,300,9000)
- Local Common Average Referencing (default: loccar_um 75,300)
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.config import CatGTConfig
from src.pipeline.io.io_catgt import run_catgt
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.diagnostics.preproc_inspector import plot_preprocessing_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RUN_CATGT")

def main():
    parser = argparse.ArgumentParser(description="Standalone CatGT Preprocessing & Artifact Inspector")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw SpikeGLX recording or imec0 folder")
    parser.add_argument("--dest-dir", type=str, default=None, help="Output destination folder for CatGT processed data")
    parser.add_argument("--catgt-path", type=str, default=r"C:\Users\lpascu\SpikeGLX\CatGT-win\CatGT.exe", help="Path to CatGT.exe")
    parser.add_argument("--apfilter", type=str, default="butter,12,300,9000", help="CatGT apfilter string (e.g. butter,12,300,9000)")
    parser.add_argument("--loccar", type=str, default="75,300", help="CatGT loccar_um inner,outer radii in um (e.g. 75,300)")
    parser.add_argument("--no-t-shift", action="store_true", help="Disable ADC time shift correction")
    parser.add_argument("--startsecs", type=float, default=None, help="Start time in seconds for snippet processing")
    parser.add_argument("--maxsecs", type=float, default=None, help="Max duration in seconds for snippet processing")
    parser.add_argument("--inspect", action="store_true", help="Generate diagnostic comparison plots (Raw vs CatGT)")
    args = parser.parse_args()

    loc_inner, loc_outer = map(float, args.loccar.split(","))
    config = CatGTConfig(
        enabled=True,
        catgt_path=args.catgt_path,
        t_shift=not args.no_t_shift,
        apfilter=args.apfilter,
        loccar_um=(loc_inner, loc_outer),
        start_secs=args.startsecs,
        max_secs=args.maxsecs,
        dest_dir=args.dest_dir
    )

    raw_path = Path(args.raw_dir)
    dest_path = Path(args.dest_dir) if args.dest_dir else None

    logger.info("=" * 80)
    logger.info(" CATGT PREPROCESSING RUNNER")
    logger.info(f" Input Raw Path : {raw_path}")
    logger.info(f" Filter         : {args.apfilter}")
    logger.info(f" Loccar radii   : {loc_inner}um exclude, {loc_outer}um include")
    logger.info(f" Time Shift     : {not args.no_t_shift}")
    logger.info("=" * 80)

    out_bin, out_meta = run_catgt(config, raw_path, dest_dir=dest_path, start_secs=args.startsecs, max_secs=args.maxsecs)
    print(f"\n CatGT Processing Complete!\n Processed Binary: {out_bin}\n Processed Meta  : {out_meta}")

    if args.inspect:
        logger.info("Generating Preprocessing & Filter Artifact Diagnostics...")
        rec_raw = read_spikeglx_ap_safe(raw_path)
        rec_pre = read_spikeglx_ap_safe(out_bin.parent)
        fig_path = out_bin.parent / "catgt_preprocessing_diagnostics.png"
        plot_preprocessing_diagnostics(rec_raw, rec_pre, fig_path)
        print(f" Diagnostic Figure: {fig_path}")

if __name__ == "__main__":
    main()
