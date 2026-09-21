#!/usr/bin/env python3
"""
Full 384-Channel Conservative Single-Channel blech_clust Spike Sorter.
Sorts all 384 Neuropixels channels, performs sub-millisecond spike coincidence check,
and cross-checks extracted high-SNR units against user's manually identified channels.
"""

import sys
import shutil
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spikeinterface.extractors as se
from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig
from src.pipeline.processors.proc_blech_single_channel import (
    make_laser_blank_mask,
    filter_and_blank,
    detect_spike_snippets,
    cluster_single_channel_gmm,
    cross_check_spike_coincidence
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("BLECH_384_SORTER")

# User's manually identified channels
MANUAL_CHANNELS = [
    # Shank 1 (Shank 0)
    4, 12, 28, 32, 96, 108, 202, 210, 222, 224, 226, 290, 316,
    # Shank 2 (Shank 1)
    48, 58, 66, 82, 90, 154, 178, 186, 190, 242, 346, 364, 374,
    # Shank 4 (Shank 3)
    279, 283, 287, 343, 349, 91, 167
]

YELLOW_HIGHLIGHTED_CHANNELS = [32, 48, 222, 226]

def main():
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    session_dir = out_dir / "alf_session"
    blech_alf_dir = out_dir / "blech_alf_384"
    blech_alf_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(" Running Full 384-Channel blech_clust Spike Sorting & Manual Channel Alignment")
    print(f" Raw Data: {raw_dir}")
    print(f" Output ALF Directory: {blech_alf_dir}")
    print("=" * 85)

    # Step 1: Read Neuropixels AP Recording
    ap_bin_file = session_dir / "raw_ephys_data" / "probe00" / "TP_Tacr1_1_npx2_g4_t0.imec0.ap.bin"
    n_channels = 385
    n_samples = ap_bin_file.stat().st_size // (2 * n_channels)

    # Read SpikeGLX Metadata for probe geometry
    ap_path = session_dir / "raw_ephys_data" / "probe00"
    rec = se.read_spikeglx(ap_path, stream_id='imec0.ap')
    fs_ap = rec.get_sampling_frequency()
    probe = rec.get_probe()
    channel_coords = probe.contact_positions

    def get_shank(x):
        if x < 125: return 0
        elif x < 391: return 1
        elif x < 657: return 2
        else: return 3

    channel_shanks = np.array([get_shank(x) for x in channel_coords[:, 0]], dtype=int)

    logger.info(f"Loaded AP Binary: {n_samples} samples ({n_samples/fs_ap:.1f} s) across {n_channels} channels.")

    # Step 2: Direct Chunked Memmap Read of ALL 384 AP Electrodes
    target_channels = list(range(384))
    logger.info(f"Direct memmap loading raw voltage traces for all {len(target_channels)} probe channels...")
    
    mm = np.memmap(ap_bin_file, dtype=np.int16, mode='r', shape=(n_samples, n_channels))
    chunk_size = 2000000
    all_traces = np.zeros((n_samples, 384), dtype=np.float32)

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        all_traces[start_idx:end_idx] = mm[start_idx:end_idx, :384].astype(np.float32)

    # Load NIDQ Laser TTL onsets/offsets for photoelectric artifact blanking
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)
    t_nidq = nidq_data.time_vector
    ch6_laser = nidq_data.all_channels[6]

    laser_onsets = t_nidq[np.where(np.diff((ch6_laser > 0.8).astype(int)) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff((ch6_laser > 0.8).astype(int)) == -1)[0]]
    logger.info(f"Loaded {len(laser_onsets)} optogenetic laser TTL pulses for photoelectric artifact blanking.")

    # Step 3: Build shared laser blanking mask (one per recording, shared across all channels)
    # Strategy: linearly interpolate across each laser window before filtering to prevent
    # filtfilt edge-ringing, then zero the filtered output inside expanded blank windows.
    MIN_FIRING_RATE_HZ = 5.0  # Laser pulses fire at 0.107 Hz; real neurons >= 5 Hz
    recording_duration_s = n_samples / fs_ap

    logger.info("Building laser blank mask (interpolation strategy)...")
    clean_mask = make_laser_blank_mask(
        n_samples, fs_ap, laser_onsets, laser_offsets,
        pre_pad_ms=1.0, post_pad_ms=2.0
    )
    logger.info(f"Blank mask: {np.sum(~clean_mask)} samples blanked ({np.sum(~clean_mask)/fs_ap*1000:.0f} ms total across {len(laser_onsets)} pulses).")

    single_channel_spiketrains = {}
    single_channel_amplitudes = {}
    single_channel_meta = []

    for ch in range(384):
        sh = channel_shanks[ch]
        depth = channel_coords[ch, 1]

        raw_trace = all_traces[:, ch]

        # Filter with interpolation-based blanking (prevents filtfilt ringing artifacts)
        filt_trace, filt_clean_mask = filter_and_blank(raw_trace, fs_ap, clean_mask, ringing_pad_ms=3.0)

        # Spike detection with threshold estimated only from clean signal segments
        spike_samples, snippets = detect_spike_snippets(
            filt_trace, fs=fs_ap, thresh_sd=4.0,
            clean_mask=filt_clean_mask
        )

        if len(spike_samples) < 20:
            continue

        # GMM PCA Feature Clustering
        labels, best_mask_gmm = cluster_single_channel_gmm(snippets, n_components=2)
        unit_spike_samples = spike_samples[best_mask_gmm]
        unit_snippets = snippets[best_mask_gmm]

        if len(unit_spike_samples) < 20:
            continue

        unit_spiketimes = unit_spike_samples / fs_ap
        firing_rate = len(unit_spiketimes) / recording_duration_s

        # Reject non-neuronal detections: require >= 5 Hz mean firing rate
        if firing_rate < MIN_FIRING_RATE_HZ:
            continue

        mean_waveform = np.mean(unit_snippets, axis=0)
        peak_amp = float(np.ptp(mean_waveform))

        # ISI refractory violation check (< 1.5 ms): allow up to 2%
        ibis = np.diff(unit_spiketimes)
        isi_violations = np.sum(ibis < 0.0015)
        isi_ratio = float(isi_violations / max(1, len(ibis)))

        if isi_ratio <= 0.02:
            single_channel_spiketrains[ch] = unit_spiketimes
            single_channel_amplitudes[ch] = peak_amp

            single_channel_meta.append({
                'channel': ch,
                'shank': sh,
                'depth': depth,
                'n_spikes': len(unit_spiketimes),
                'peak_amp_uv': peak_amp,
                'isi_violations_ratio': isi_ratio,
                'firing_rate': float(firing_rate),
                'is_manual_candidate': bool(ch in MANUAL_CHANNELS),
                'is_yellow_highlight': bool(ch in YELLOW_HIGHLIGHTED_CHANNELS)
            })

    df_meta_384 = pd.DataFrame(single_channel_meta)
    logger.info(f"Extracted {len(single_channel_spiketrains)} single-channel units across 384 channels prior to deduplication.")

    # Step 4: Sub-Millisecond Cross-Channel Spike Coincidence Deduplication
    logger.info("\n[*] Executing Sub-Millisecond Cross-Channel Spike Coincidence Check (|tA - tB| <= 0.5 ms)...")
    dedup_spiketrains, coincidence_report = cross_check_spike_coincidence(
        single_channel_spiketrains,
        single_channel_amplitudes,
        window_ms=0.5,
        match_thresh=0.80
    )

    df_coincidence = pd.DataFrame(coincidence_report)
    coinc_csv = out_dir / "cross_channel_unit_coincidence_384.csv"
    df_coincidence.to_csv(coinc_csv, index=False)

    # Step 5: Compare Extracted Units against User's Manual Channels
    final_unit_channels = sorted(list(dedup_spiketrains.keys()))
    logger.info(f"\nFinal Deduplicated Single Units Count: {len(final_unit_channels)} (from 384 channels)")

    alignment_records = []
    for ch in final_unit_channels:
        spks = dedup_spiketrains[ch]
        amp = single_channel_amplitudes[ch]
        sh = channel_shanks[ch]
        depth = channel_coords[ch, 1]
        
        is_manual = ch in MANUAL_CHANNELS
        is_yellow = ch in YELLOW_HIGHLIGHTED_CHANNELS

        alignment_records.append({
            'channel': ch,
            'shank': sh,
            'depth': depth,
            'n_spikes': len(spks),
            'peak_amp_uv': amp,
            'firing_rate': np.round(len(spks) / (n_samples / fs_ap), 3),
            'is_manual_channel': is_manual,
            'is_yellow_highlight': is_yellow
        })

    df_alignment = pd.DataFrame(alignment_records)
    align_csv = out_dir / "manual_vs_blech_channel_alignment.csv"
    df_alignment.to_csv(align_csv, index=False)

    # Save ALF numpy files for Phy
    all_times = []
    all_clusters = []
    for uid, ch in enumerate(final_unit_channels):
        spks = dedup_spiketrains[ch]
        all_times.extend(spks)
        all_clusters.extend([uid] * len(spks))

    all_times = np.array(all_times)
    all_clusters = np.array(all_clusters)

    sort_idx = np.argsort(all_times)
    spikes_times = all_times[sort_idx]
    spikes_clusters = all_clusters[sort_idx]

    cluster_channels = np.array(final_unit_channels)
    cluster_depths = np.array([channel_coords[ch, 1] for ch in final_unit_channels])
    cluster_shanks = np.array([channel_shanks[ch] for ch in final_unit_channels])

    np.save(blech_alf_dir / "spikes.times.npy", spikes_times)
    np.save(blech_alf_dir / "spikes.clusters.npy", spikes_clusters)
    np.save(blech_alf_dir / "channels.localCoordinates.npy", channel_coords)
    np.save(blech_alf_dir / "channel_positions.npy", channel_coords)
    np.save(blech_alf_dir / "channel_shanks.npy", channel_shanks)
    np.save(blech_alf_dir / "channel_map.npy", np.arange(384))
    np.save(blech_alf_dir / "clusters.channels.npy", cluster_channels)
    np.save(blech_alf_dir / "clusters.depths.npy", cluster_depths)

    info_df = pd.DataFrame({
        'cluster_id': np.arange(len(final_unit_channels)),
        'ch': cluster_channels,
        'shank': cluster_shanks,
        'depth': np.round(cluster_depths, 1),
        'n_spikes': [np.sum(spikes_clusters == i) for i in range(len(final_unit_channels))],
        'group': ['good' for _ in final_unit_channels],
        'is_manual_pick': [ch in MANUAL_CHANNELS for ch in cluster_channels],
        'peak_amp_uv': [np.round(single_channel_amplitudes[ch], 1) for ch in cluster_channels],
        'firing_rate': [np.round(np.sum(spikes_clusters == i) / (n_samples / fs_ap), 3) for i in range(len(final_unit_channels))]
    })

    info_df.to_csv(blech_alf_dir / "cluster_info.tsv", sep="\t", index=False)
    info_df[['cluster_id', 'group']].to_csv(blech_alf_dir / "cluster_group.tsv", sep="\t", index=False)

    # Write Phy params.py
    raw_ap_bin = session_dir / "raw_ephys_data" / "probe00" / "TP_Tacr1_1_npx2_g4_t0.imec0.ap.bin"
    with open(blech_alf_dir / "params.py", "w") as f:
        f.write(f"dat_path = r'{raw_ap_bin}'\n")
        f.write(f"n_channels_dat = 385\n")
        f.write(f"dtype = 'int16'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {fs_ap}\n")
        f.write(f"hp_filtered = False\n")

    # Step 6: Alignment Summary Statistics
    manual_matched = df_alignment[df_alignment['is_manual_channel'] == True]
    yellow_matched = df_alignment[df_alignment['is_yellow_highlight'] == True]

    print("\n" + "=" * 85)
    print(" [ALIGNMENT REPORT] Full 384-Channel blech_clust vs. Manual Channel Selection")
    print("=" * 85)
    print(f" Total 384-Channel Deduplicated Units Extracted: {len(final_unit_channels)}")
    print(f" Manual Channels Matched: {len(manual_matched)} / {len(MANUAL_CHANNELS)} ({len(manual_matched)/len(MANUAL_CHANNELS)*100:.1f}%)")
    print(f" Yellow Highlighted Channels Matched: {len(yellow_matched)} / {len(YELLOW_HIGHLIGHTED_CHANNELS)} ({len(yellow_matched)/len(YELLOW_HIGHLIGHTED_CHANNELS)*100:.1f}%)")
    print("=" * 85)
    print("\nMatched Units Detail:")
    print(df_alignment.to_string(index=False))

if __name__ == "__main__":
    main()
