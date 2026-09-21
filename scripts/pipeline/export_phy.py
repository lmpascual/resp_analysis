#!/usr/bin/env python3
"""
Export SpikeInterface SortingAnalyzer to Complete Phy GUI Format.
Fixes missing spike_templates.npy, templates.npy, amplitudes.npy, etc.
Handles directory locks cleanly.
"""

import sys
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import spikeinterface.full as si
import spikeinterface.exporters as se_exp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("PHY_EXPORTER")

def main():
    scratch_parent = Path(r"C:\SCRATCH_dredge_shanks")
    waveforms_folder = scratch_parent / "combined_waveforms"

    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    raw_ap_bin = session_dir / "raw_ephys_data" / "probe00" / "TP_Tacr1_1_npx2_g4_t0.imec0.ap.bin"
    phy_dir = session_dir / "alf" / "probe00"
    temp_export_dir = session_dir / "alf" / "phy_temp"

    print("=" * 85)
    print(" Exporting Kilosort4 Units to Complete Phy GUI Format")
    print(f" Source Analyzer: {waveforms_folder}")
    print(f" Output Directory: {phy_dir}")
    print("=" * 85)

    # 1. Load analyzer
    logger.info("Loading SortingAnalyzer from disk...")
    analyzer = si.load_sorting_analyzer(waveforms_folder)
    logger.info(f"Loaded analyzer: {len(analyzer.sorting.unit_ids)} units, {analyzer.get_num_channels()} channels.")

    # 2. Run export_to_phy into temp_export_dir
    if temp_export_dir.exists():
        shutil.rmtree(temp_export_dir, ignore_errors=True)

    logger.info(f"Running SpikeInterface export_to_phy to temporary folder: {temp_export_dir}...")
    se_exp.export_to_phy(
        sorting_analyzer=analyzer,
        output_folder=temp_export_dir,
        compute_pc_features=False,
        compute_amplitudes=True,
        copy_binary=False,
        remove_if_exists=True,
        add_quality_metrics=False,
        add_template_metrics=False,
        verbose=True
    )

    # 3. Copy files from temp_export_dir to phy_dir
    logger.info(f"Copying generated Phy files into target directory: {phy_dir}...")
    phy_dir.mkdir(parents=True, exist_ok=True)
    for f in temp_export_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, phy_dir / f.name)

    # Clean up temp folder
    shutil.rmtree(temp_export_dir, ignore_errors=True)

    # 4. Ensure params.py points to the local AP binary
    params_file = phy_dir / "params.py"
    with open(params_file, "w") as f:
        f.write(f"dat_path = r'{raw_ap_bin}'\n")
        f.write("n_channels_dat = 385\n")
        f.write("dtype = 'int16'\n")
        f.write("offset = 0\n")
        f.write(f"sample_rate = {analyzer.sampling_frequency}\n")
        f.write("hp_filtered = False\n")

    # Create aliases if phy or ALF needs specific naming conventions
    aliases = {
        "spike_times.npy": "spikes.times.npy",
        "spike_clusters.npy": "spikes.clusters.npy",
        "spike_templates.npy": "spikes.templates.npy",
        "channel_positions.npy": "channels.localCoordinates.npy"
    }

    for src_name, target_name in aliases.items():
        src_path = phy_dir / src_name
        target_path = phy_dir / target_name
        if src_path.exists() and not target_path.exists():
            shutil.copy2(src_path, target_path)

    # Check cluster_info.tsv
    info_file = phy_dir / "cluster_info.tsv"
    if info_file.exists():
        info_df = pd.read_csv(info_file, sep="\t")
        logger.info(f"Generated cluster_info.tsv with {len(info_df)} units.")

    print("\n" + "=" * 85)
    print(f" [SUCCESS] Phy GUI Export Complete!")
    print(f" Directory: {phy_dir}")
    print(" You can now run:")
    print(f"   cd {phy_dir}")
    print("   phy template-gui params.py")
    print("=" * 85)

if __name__ == "__main__":
    main()
