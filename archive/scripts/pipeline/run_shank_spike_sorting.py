#!/usr/bin/env python3
"""
Dedicated Shank-by-Shank Neuropixels 2.0 Kilosort4 Spike Sorting Pipeline.
Guarantees 100% physical spatial isolation per shank (Shanks 0, 1, 2, 3).
"""

import sys
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Import spikeinterface & cibrrig
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("SHANK_SORTER")

def run_shank_sorting(
    raw_dir: Path,
    session_dir: Path,
    testing: bool = False
):
    session_dir = Path(session_dir)
    alf_dir = session_dir / "alf" / "probe00"
    raw_ephys_dir = session_dir / "raw_ephys_data" / "probe00"

    print("=" * 85)
    print(" Running 4-Shank Neuropixels 2.0 Kilosort 4 Spike Sorting")
    print(f" Raw Ephys Folder: {raw_ephys_dir}")
    print(f" Output ALF Folder: {alf_dir}")
    print(f" Testing Mode (60s clip): {testing}")
    print("=" * 85)

    # 1. Clean previous ALF probe folder if restarting
    if alf_dir.exists():
        logger.info(f"Cleaning previous sorting output at: {alf_dir}")
        shutil.rmtree(alf_dir, ignore_errors=True)
    alf_dir.mkdir(parents=True, exist_ok=True)

    # 2. Read SpikeGLX Recording
    rec = se.read_spikeglx(raw_ephys_dir, stream_id='imec0.ap')
    probe = rec.get_probe()
    logger.info(f"Loaded SpikeGLX Recording: {rec.get_num_channels()} channels | Fs = {rec.get_sampling_frequency()} Hz | Duration = {rec.get_total_duration():.1f} s")

    # If testing mode, slice first 60s (1,800,000 samples)
    if testing:
        logger.info("Testing mode active: slicing first 60 seconds (1,800,000 samples)...")
        rec = rec.frame_slice(start_frame=0, end_frame=int(60.0 * rec.get_sampling_frequency()))

    # 3. Preprocessing: Phase shift correction, bandpass 300-6000Hz, bad channel interpolation
    logger.info("Preprocessing AP recording (Phase Shift, Bandpass 300-6000Hz, Bad Channel Interp)...")
    rec_pre = spre.phase_shift(rec)
    rec_pre = spre.bandpass_filter(rec_pre, freq_min=300.0, freq_max=6000.0)
    try:
        bad_channel_ids, _ = spre.detect_bad_channels(rec_pre)
        if len(bad_channel_ids) > 0:
            logger.info(f"  Detected {len(bad_channel_ids)} bad channels: {bad_channel_ids}")
            rec_pre = spre.interpolate_bad_channels(rec_pre, bad_channel_ids)
    except Exception as e:
        logger.warning(f"  Bad channel detection note: {e}")

    # 4. Split by Shank (Shanks 0, 1, 2, 3)
    rec_pre.set_property('shank', [int(s) for s in probe.shank_ids])
    rec_shanks = rec_pre.split_by('shank')
    logger.info(f"Split probe into {len(rec_shanks)} isolated shanks: {list(rec_shanks.keys())}")

    # 5. Run Kilosort 4 per Shank
    shank_sortings = []
    scratch_parent = Path(r"C:\SCRATCH_dredge_shanks")
    scratch_parent.mkdir(parents=True, exist_ok=True)

    for shank_id, rec_s in rec_shanks.items():
        logger.info(f"\n---> Sorting Shank {shank_id} ({rec_s.get_num_channels()} channels)...")
        scratch_s = scratch_parent / f"shank_{shank_id}"
        if scratch_s.exists():
            shutil.rmtree(scratch_s, ignore_errors=True)

        sorter_params = dict(do_CAR=True, do_correction=True)
        sort_s = ss.run_sorter(
            sorter_name='kilosort4',
            recording=rec_s,
            output_folder=scratch_s,
            verbose=True,
            **sorter_params
        )
        logger.info(f"  [+] Shank {shank_id} sorting finished: {len(sort_s.unit_ids)} units detected.")
        shank_sortings.append(sort_s)

    # 6. Aggregate Sortings into Unified Recording & Sorting
    logger.info("\nAggregating shank sortings into unified probe dataset...")
    combined_sorting = si.aggregate_units(shank_sortings)
    logger.info(f"Total aggregated units across all 4 shanks: {len(combined_sorting.unit_ids)}")

    # 7. Extract Waveforms & Postprocessing Metrics
    logger.info("Computing unit waveforms, templates, amplitudes, and quality metrics...")
    waveforms_folder = scratch_parent / "combined_waveforms"
    if waveforms_folder.exists():
        shutil.rmtree(waveforms_folder, ignore_errors=True)

    analyzer = si.create_sorting_analyzer(
        sorting=combined_sorting,
        recording=rec_pre,
        format="binary_folder",
        folder=waveforms_folder,
        overwrite=True
    )
    analyzer.compute("random_spikes")
    analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("spike_amplitudes")
    analyzer.compute("noise_levels")
    analyzer.compute("quality_metrics")

    # 8. Export to ALF Format
    logger.info("\nExporting sorted units and metrics to ALF directory...")
    spikes_times = combined_sorting.get_all_spike_trains()[0][0] / rec.get_sampling_frequency()
    spikes_clusters = combined_sorting.get_all_spike_trains()[0][1]

    # Channel & Shank Mapping
    channel_coords = rec_pre.get_channel_locations()
    channel_shanks = np.array([int(s) for s in probe.shank_ids], dtype=int)

    qm_ext = analyzer.get_extension("quality_metrics")
    qm_df = qm_ext.get_data() if qm_ext is not None else pd.DataFrame()

    unit_ids = combined_sorting.unit_ids
    cluster_channels = []
    cluster_depths = []
    cluster_shanks = []

    templates_ext = analyzer.get_extension("templates")
    templates_data = templates_ext.get_data() if templates_ext is not None else None

    for idx, uid in enumerate(unit_ids):
        if templates_data is not None:
            tmpl = templates_data[idx] # (samples, channels)
            ptp = np.ptp(tmpl, axis=0)
            best_ch = int(np.argmax(ptp))
        else:
            best_ch = 0
        cluster_channels.append(best_ch)
        cluster_depths.append(float(channel_coords[best_ch, 1]))
        cluster_shanks.append(int(channel_shanks[best_ch]))

    cluster_channels = np.array(cluster_channels)
    cluster_depths = np.array(cluster_depths)
    cluster_shanks = np.array(cluster_shanks)

    # Save ALF numpy files
    np.save(alf_dir / "spikes.times.npy", spikes_times)
    np.save(alf_dir / "spikes.clusters.npy", spikes_clusters)
    np.save(alf_dir / "channels.localCoordinates.npy", channel_coords)
    np.save(alf_dir / "channel_positions.npy", channel_coords)
    np.save(alf_dir / "channel_shanks.npy", channel_shanks)
    np.save(alf_dir / "channel_map.npy", np.arange(rec.get_num_channels()))
    np.save(alf_dir / "clusters.channels.npy", cluster_channels)
    np.save(alf_dir / "clusters.depths.npy", cluster_depths)

    # Spike depths
    spike_depths = np.array([cluster_depths[c] for c in spikes_clusters])
    np.save(alf_dir / "spikes.depths.npy", spike_depths)

    # Save cluster_info.tsv & cluster_group.tsv
    info_df = pd.DataFrame({
        "cluster_id": np.arange(len(unit_ids)),
        "ch": cluster_channels,
        "shank": cluster_shanks,
        "depth": np.round(cluster_depths, 1),
        "n_spikes": [np.sum(spikes_clusters == i) for i in range(len(unit_ids))],
        "group": ["good" if (qm_df.get("snr", [10])[i] >= 5.0 and qm_df.get("isi_violations_ratio", [0])[i] < 0.5) else "mua" for i in range(len(unit_ids))],
        "pos_x": [channel_coords[ch, 0] for ch in cluster_channels],
        "pos_y": [channel_coords[ch, 1] for ch in cluster_channels]
    })

    if not qm_df.empty:
        for col in ["firing_rate", "snr", "isi_violations_ratio"]:
            if col in qm_df.columns:
                info_df[col] = np.round(qm_df[col].values, 3)

    info_df.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)
    info_df[["cluster_id", "group"]].to_csv(alf_dir / "cluster_group.tsv", sep="\t", index=False)

    # Write Phy params.py
    ap_bin_file = list(raw_ephys_dir.glob("*.ap.bin"))[0]
    with open(alf_dir / "params.py", "w") as f:
        f.write(f"dat_path = r'{ap_bin_file}'\n")
        f.write(f"n_channels_dat = 385\n")
        f.write(f"dtype = 'int16'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {rec.get_sampling_frequency()}\n")
        f.write(f"hp_filtered = False\n")

    print("\n" + "=" * 85)
    print(f" [SUCCESS] 4-Shank Kilosort 4 Spike Sorting Complete!")
    print(f" Total Units: {len(unit_ids)} across 4 isolated shanks.")
    print(f" Saved to ALF: {alf_dir}")
    print("=" * 85)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 4-Shank Neuropixels 2.0 Kilosort4 Spike Sorting")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    parser.add_argument("--session-dir", type=str, default=r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    parser.add_argument("--testing", action="store_true", help="Run 60s test mode")
    args = parser.parse_args()

    run_shank_sorting(Path(args.raw_dir), Path(args.session_dir), testing=args.testing)
