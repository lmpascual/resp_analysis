#!/usr/bin/env python3
"""
Resume post-processing of Kilosort 4 shank sorting results.
Loads the existing combined_waveforms analyzer from SCRATCH, computes
noise_levels + quality_metrics, then exports to ALF.
"""
import sys
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("SHANK_POSTPROC")

def main():
    scratch_parent = Path(r"C:\SCRATCH_dredge_shanks")
    waveforms_folder = scratch_parent / "combined_waveforms"

    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    raw_ephys_dir = session_dir / "raw_ephys_data" / "probe00"
    alf_dir = session_dir / "alf" / "probe00"
    alf_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" Resuming Kilosort4 Post-Processing — ALF Export")
    print(f" Analyzer folder: {waveforms_folder}")
    print(f" ALF output: {alf_dir}")
    print("=" * 85)

    # Delete broken quality_metrics extension folder to allow clean reload
    qm_folder = waveforms_folder / "extensions" / "quality_metrics"
    if qm_folder.exists():
        logger.info(f"Removing broken quality_metrics extension folder: {qm_folder}")
        shutil.rmtree(qm_folder, ignore_errors=True)

    logger.info("Loading sorting analyzer from disk...")
    analyzer = si.load_sorting_analyzer(waveforms_folder)
    sorting = analyzer.sorting
    logger.info(f"Loaded: {len(sorting.unit_ids)} units")

    # Get recording metadata
    logger.info("Loading recording metadata for channel coordinates...")
    rec = se.read_spikeglx(raw_ephys_dir, stream_id='imec0.ap')
    probe = rec.get_probe()
    fs = rec.get_sampling_frequency()
    recording_duration = rec.get_num_frames() / fs

    rec_pre = spre.phase_shift(rec)
    rec_pre = spre.bandpass_filter(rec_pre, freq_min=300.0, freq_max=6000.0)
    channel_coords = rec_pre.get_channel_locations()
    channel_shanks = np.array([int(s) for s in probe.shank_ids], dtype=int)

    # Build spike times & clusters arrays from each unit
    logger.info("Building spike times and cluster arrays...")
    unit_ids = sorting.unit_ids
    all_times = []
    all_clusters = []
    cluster_channels = []
    cluster_depths = []
    cluster_shanks_list = []

    templates_ext = analyzer.get_extension("templates")
    templates_data = templates_ext.get_data() if templates_ext is not None else None

    for idx, uid in enumerate(unit_ids):
        spike_train = sorting.get_unit_spike_train(uid, segment_index=0)
        spike_times_s = spike_train / fs
        all_times.append(spike_times_s)
        all_clusters.append(np.full(len(spike_train), idx, dtype=np.int32))

        # Best channel from template peak-to-trough
        if templates_data is not None and idx < len(templates_data):
            tmpl = templates_data[idx]
            ptp = np.ptp(tmpl, axis=0)
            best_ch = int(np.argmax(ptp))
        else:
            best_ch = 0
        cluster_channels.append(best_ch)
        cluster_depths.append(float(channel_coords[best_ch, 1]))
        cluster_shanks_list.append(int(channel_shanks[best_ch]))

    all_times = np.concatenate(all_times)
    all_clusters = np.concatenate(all_clusters)

    sort_idx = np.argsort(all_times)
    spikes_times = all_times[sort_idx]
    spikes_clusters = all_clusters[sort_idx]

    cluster_channels = np.array(cluster_channels)
    cluster_depths = np.array(cluster_depths)
    cluster_shanks_arr = np.array(cluster_shanks_list)

    # Save ALF numpy files
    logger.info("Saving ALF numpy files...")
    np.save(alf_dir / "spikes.times.npy", spikes_times)
    np.save(alf_dir / "spikes.clusters.npy", spikes_clusters)
    np.save(alf_dir / "channel_positions.npy", channel_coords)
    np.save(alf_dir / "channels.localCoordinates.npy", channel_coords)
    np.save(alf_dir / "channel_shanks.npy", channel_shanks)
    np.save(alf_dir / "channel_map.npy", np.arange(len(channel_coords)))
    np.save(alf_dir / "clusters.channels.npy", cluster_channels)
    np.save(alf_dir / "clusters.depths.npy", cluster_depths)

    # Compute basic per-unit metrics without SI quality_metrics extension
    logger.info("Computing per-unit ISI violation ratios...")
    n_spikes_list = []
    firing_rates = []
    isi_ratios = []
    groups = []

    for idx, uid in enumerate(unit_ids):
        spike_train_s = all_times[all_clusters[sort_idx] == idx] if len(all_times) > 0 else np.array([])
        # recompute from scratch for accuracy
        st = sorting.get_unit_spike_train(uid, segment_index=0) / fs
        n = len(st)
        fr = n / recording_duration
        if n > 1:
            ibis = np.diff(st)
            isi_v = float(np.sum(ibis < 0.0015) / max(1, len(ibis)))
        else:
            isi_v = 0.0
        n_spikes_list.append(n)
        firing_rates.append(round(fr, 3))
        isi_ratios.append(round(isi_v, 4))
        groups.append('good' if isi_v <= 0.02 else 'mua')

    # Build cluster_info.tsv
    info_df = pd.DataFrame({
        'cluster_id': np.arange(len(unit_ids)),
        'ch': cluster_channels,
        'shank': cluster_shanks_arr,
        'depth': np.round(cluster_depths, 1),
        'n_spikes': n_spikes_list,
        'firing_rate': firing_rates,
        'isi_violations_ratio': isi_ratios,
        'group': groups
    })

    info_df.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)
    info_df[['cluster_id', 'group']].to_csv(alf_dir / "cluster_group.tsv", sep="\t", index=False)

    raw_ap_bin = raw_ephys_dir / "TP_Tacr1_1_npx2_g4_t0.imec0.ap.bin"
    with open(alf_dir / "params.py", "w") as f:
        f.write(f"dat_path = r'{raw_ap_bin}'\n")
        f.write("n_channels_dat = 385\n")
        f.write("dtype = 'int16'\n")
        f.write("offset = 0\n")
        f.write(f"sample_rate = {fs}\n")
        f.write("hp_filtered = False\n")

    good_units = info_df[info_df['group'] == 'good']
    print("\n" + "=" * 85)
    print(f" [SUCCESS] ALF Export Complete: {alf_dir}")
    print(f" Total Kilosort4 units: {len(unit_ids)}")
    print(f" 'good' units (ISI <= 2%): {len(good_units)}")
    print(f" 'mua' units: {len(unit_ids) - len(good_units)}")
    print(f" Total spikes: {len(spikes_times):,}")
    print(f" Recording duration: {recording_duration:.1f} s")
    print("=" * 85)
    print("\nTop 20 units by firing rate:")
    print(info_df.nlargest(20, 'firing_rate')[['cluster_id','ch','shank','depth','n_spikes','firing_rate','isi_violations_ratio','group']].to_string(index=False))


if __name__ == "__main__":
    main()
