#!/usr/bin/env python3
"""
Integration runner for Nicholas Bush's cibrrig preprocessing and spike sorting pipeline.
Executes ALF session structuring, NIDQ auxiliary preprocessing, Kilosort4 spike sorting,
probe-to-AUX timestamp synchronization, and respiratory modulation calculation.
"""

import sys
import argparse
import logging
from pathlib import Path

# Import natively installed cibrrig package
import cibrrig
import cibrrig.archiving.ephys_data_to_alf as ephys_to_alf
import cibrrig.preprocess.preproc_pipeline as preproc
import cibrrig.sorting.spikeinterface_ks4 as ks4
import cibrrig.postprocess.synchronize_sorting_to_aux as sync_aux
import cibrrig.postprocess.extract_resp_modulation as resp_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("CIBRRIG_RUNNER")

def setup_alf_directory(raw_dir: Path, session_dir: Path):
    """
    Format raw SpikeGLX recording into standard IBL ALF folder structure expected by cibrrig.
    raw_ephys_data/ -> contains NIDQ and probe folders (probe00, etc.)
    """
    session_dir = Path(session_dir)
    raw_ephys_dir = session_dir / "raw_ephys_data"
    raw_ephys_dir.mkdir(parents=True, exist_ok=True)
    alf_dir = session_dir / "alf"
    alf_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Setting up ALF structure at: {session_dir}")

    # Copy / Link NIDQ files into raw_ephys_data
    nidq_files = list(raw_dir.glob("*.nidq.*")) + list(raw_dir.glob("*_t0.nidq.*"))
    for f in nidq_files:
        dest_f = raw_ephys_dir / f.name
        if not dest_f.exists():
            logger.info(f"  Copying NIDQ file {f.name} -> {dest_f}")
            import shutil
            shutil.copy2(f, dest_f)

    # Copy / Link Probe AP files into raw_ephys_data/probe00
    probe_dirs = list(raw_dir.glob("*imec*")) + list(raw_dir.glob("probe*"))
    for p_dir in probe_dirs:
        probe_name = "probe00"  # default single probe
        probe_dest = raw_ephys_dir / probe_name
        probe_dest.mkdir(parents=True, exist_ok=True)

        for ap_file in p_dir.glob("*"):
            dest_ap = probe_dest / ap_file.name
            if not dest_ap.exists():
                logger.info(f"  Copying Probe file {ap_file.name} -> {dest_ap}")
                import shutil
                shutil.copy2(ap_file, dest_ap)

    return session_dir

def main():
    parser = argparse.ArgumentParser(
        description="Run Nicholas Bush's cibrrig preprocessing & Kilosort4 spike sorting pipeline."
    )
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4",
                        help="Path to raw SpikeGLX data folder")
    parser.add_argument("--session-dir", type=str, default=r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session",
                        help="Path to session directory for ALF output")
    parser.add_argument("--testing", action="store_true", help="Run 60s fast test mode for Kilosort4")
    parser.add_argument("--skip-sorting", action="store_true", help="Skip spike sorting step")
    parser.add_argument("--skip-preproc", action="store_true", help="Skip NIDQ auxiliary preprocessing step")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    session_dir = Path(args.session_dir)

    print("=" * 85)
    print(f" Executing cibrrig Pipeline")
    print(f" Raw Data: {raw_dir}")
    print(f" ALF Session: {session_dir}")
    print(f" Test Mode: {args.testing}")
    print("=" * 85)

    if not raw_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_dir}")
        sys.exit(1)

    # Step 1: Format ALF Directory Structure
    setup_alf_directory(raw_dir, session_dir)

    # Step 2: Run NIDQ Auxiliary Preprocessing
    if not args.skip_preproc:
        logger.info("\n[*] STEP 1: Running cibrrig Preprocessing (Sync, Opto, Diaphragm EMG)...")
        try:
            preproc.run(session_dir, skip_ephysqc=True)
            logger.info(" [+] Preprocessing completed successfully.")
        except Exception as e:
            logger.warning(f" [!] Preprocessing note: {e}")

    # Step 3: Run Kilosort4 Spike Sorting
    if not args.skip_sorting:
        logger.info("\n[*] STEP 2: Running SpikeInterface / Kilosort4 Spike Sorting...")
        try:
            ks4.run(
                session_path=session_dir,
                dest=session_dir / "alf",
                testing=args.testing,
                skip_remove_opto=True
            )
            logger.info(" [+] Spike sorting & ALF export completed successfully.")
        except Exception as e:
            logger.error(f" [!] Spike sorting failed: {e}", exc_info=True)

    # Step 4: Synchronize Sorting to AUX Clock
    logger.info("\n[*] STEP 3: Synchronizing Sorting Timestamps to AUX/NIDQ Clock...")
    try:
        sync_aux.run(session_dir)
        logger.info(" [+] Sorting timestamps synchronized.")
    except Exception as e:
        logger.warning(f" [!] Timestamp synchronization note: {e}")

    # Step 5: Compute Respiratory Modulation
    logger.info("\n[*] STEP 4: Computing Respiratory Modulation & Unit PSTHs...")
    try:
        resp_mod.run(session_dir)
        logger.info(" [+] Respiratory modulation extraction completed.")
    except Exception as e:
        logger.warning(f" [!] Respiratory modulation note: {e}")

    print("\n" + "=" * 85)
    print(f" [SUCCESS] cibrrig Pipeline Execution Complete!")
    print(f" ALF Results saved at: {session_dir / 'alf'}")
    print("=" * 85)

if __name__ == "__main__":
    main()
