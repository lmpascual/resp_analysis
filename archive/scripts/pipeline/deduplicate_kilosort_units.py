#!/usr/bin/env python3
"""
Automated Sub-Millisecond Spike Coincidence Deduplication Processor.
Performs pairwise spike time overlap check across all Kilosort4 units:
- Identifies unit pairs with >= 20% mutual overlap or >= 50% single-unit containment.
- Retains the dominant unit with larger peak-to-trough waveform amplitude (uV).
- Marks secondary duplicate units as 'duplicate' in cluster_info.tsv and cluster_group.tsv.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("DEDUPLICATOR")

def get_shank(x):
    if x < 125: return 0
    elif x < 391: return 1
    elif x < 657: return 2
    else: return 3

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    session_dir = out_dir / "alf_session"
    alf_dir = session_dir / "alf" / "probe00"

    print("=" * 85)
    print(" Running Automated Spike Coincidence Deduplication on Kilosort4 Units")
    print(f" ALF Directory: {alf_dir}")
    print("=" * 85)

    # 1. Load spike data & channel geometry
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    templates = np.load(alf_dir / "templates.npy")
    channel_coords = np.load(alf_dir / "channel_positions.npy")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    rec_duration = spikes_times.max()

    # Load or compute cluster info
    info_file = alf_dir / "cluster_info.tsv"
    if info_file.exists():
        info_df = pd.read_csv(info_file, sep="\t")
    else:
        unit_ids = np.unique(spikes_clusters)
        rows = []
        for uid in unit_ids:
            u_spks = spikes_times[spikes_clusters == uid]
            n_spk = len(u_spks)
            fr = n_spk / rec_duration
            
            if uid < len(templates):
                tmpl = templates[uid]
                ptp = np.ptp(tmpl, axis=0)
                best_ch = int(np.argmax(ptp))
                amp_uv = float(np.max(ptp))
            else:
                best_ch = 0
                amp_uv = 0.0

            depth = float(channel_coords[best_ch, 1])
            shank = get_shank(channel_coords[best_ch, 0])

            if n_spk > 1:
                ibis = np.diff(u_spks)
                isi_v = float(np.sum(ibis < 0.0015) / max(1, len(ibis)))
            else:
                isi_v = 0.0

            group = 'good' if isi_v <= 0.02 else 'mua'
            rows.append({
                'cluster_id': uid, 'ch': best_ch, 'shank': shank, 'depth': depth,
                'n_spikes': n_spk, 'firing_rate': fr, 'peak_amp_uv': amp_uv,
                'isi_violations_ratio': isi_v, 'group': group
            })
        info_df = pd.DataFrame(rows)

    # Compute peak waveform amplitude for all units if not in info_df
    if 'peak_amp_uv' not in info_df.columns:
        amps = []
        for idx, row in info_df.iterrows():
            uid = int(row['cluster_id'])
            if uid < len(templates):
                tmpl = templates[uid]
                amp = float(np.max(np.ptp(tmpl, axis=0)))
            else:
                amp = 0.0
            amps.append(round(amp, 2))
        info_df['peak_amp_uv'] = amps

    # 2. Sub-Millisecond Spike Coincidence Deduplication
    spiketrains = {}
    amplitudes = {}
    for idx, row in info_df.iterrows():
        uid = int(row['cluster_id'])
        u_spks = spikes_times[spikes_clusters == uid]
        if len(u_spks) > 0:
            spiketrains[uid] = np.sort(u_spks)
            amplitudes[uid] = float(row['peak_amp_uv'])

    all_uids = sorted(list(spiketrains.keys()))
    win_s = 0.0005 # 0.5 ms window

    duplicate_map = {} # duplicate_uid -> dominant_uid
    dedup_report = []

    logger.info(f"Cross-checking {len(all_uids)} units for sub-millisecond spike coincidence...")

    for i in range(len(all_uids)):
        u1 = all_uids[i]
        if u1 in duplicate_map:
            continue
        st1 = spiketrains[u1]
        n1 = len(st1)

        for j in range(i + 1, len(all_uids)):
            u2 = all_uids[j]
            if u2 in duplicate_map:
                continue
            st2 = spiketrains[u2]
            n2 = len(st2)

            # Binary search coincidence check
            left = np.searchsorted(st2, st1 - win_s)
            right = np.searchsorted(st2, st1 + win_s)
            n_coinc = int(np.sum(right > left))

            if n_coinc > 0:
                ratio1 = n_coinc / n1
                ratio2 = n_coinc / n2
                overlap = (2.0 * n_coinc) / (n1 + n2)

                # Flag as duplicate if mutual overlap >= 20% OR single-unit containment >= 50%
                is_dup = (overlap >= 0.20) or (ratio1 >= 0.50) or (ratio2 >= 0.50)

                if is_dup:
                    amp1 = amplitudes.get(u1, 0.0)
                    amp2 = amplitudes.get(u2, 0.0)

                    # Retain dominant unit with larger peak amplitude / higher spike count
                    if amp1 > amp2 or (amp1 == amp2 and n1 >= n2):
                        dominant, duplicate = u1, u2
                    else:
                        dominant, duplicate = u2, u1

                    duplicate_map[duplicate] = dominant
                    dedup_report.append({
                        'dominant_unit': dominant,
                        'duplicate_unit': duplicate,
                        'n_spikes_dom': max(n1, n2),
                        'n_spikes_dup': min(n1, n2),
                        'amp_dom_uv': max(amp1, amp2),
                        'amp_dup_uv': min(amp1, amp2),
                        'coinc_overlap': float(overlap)
                    })

                    logger.info(
                        f"  [Duplicate Found] Unit {duplicate} (amp {min(amp1,amp2):.1f} uV) "
                        f"is duplicate of Unit {dominant} (amp {max(amp1,amp2):.1f} uV) | Overlap: {overlap*100:.1f}%"
                    )

                    if duplicate == u1:
                        break

    # 3. Update cluster_info.tsv and cluster_group.tsv
    new_groups = []
    duplicate_of_list = []

    for idx, row in info_df.iterrows():
        uid = int(row['cluster_id'])
        orig_group = str(row['group'])
        if uid in duplicate_map:
            new_groups.append('duplicate')
            duplicate_of_list.append(duplicate_map[uid])
        else:
            new_groups.append(orig_group)
            duplicate_of_list.append(-1)

    info_df['group'] = new_groups
    info_df['duplicate_of'] = duplicate_of_list

    info_df.to_csv(info_file, sep="\t", index=False)
    info_df[['cluster_id', 'group']].to_csv(alf_dir / "cluster_group.tsv", sep="\t", index=False)

    df_dedup = pd.DataFrame(dedup_report)
    df_dedup.to_csv(out_dir / "kilosort_deduplication_report.csv", index=False)

    # Summary Statistics
    total_units = len(info_df)
    good_clean = len(info_df[info_df['group'] == 'good'])
    duplicates_count = len(info_df[info_df['group'] == 'duplicate'])
    mua_count = len(info_df[info_df['group'] == 'mua'])

    print("\n" + "=" * 85)
    print(" [DEDUPLICATION REPORT] Kilosort4 Unit Coincidence Filtering Results")
    print("=" * 85)
    print(f" Total Units Evaluated:       {total_units}")
    print(f" Clean 'good' Single Units:    {good_clean}  (Retained Dominant Units)")
    print(f" Marked 'duplicate' Units:     {duplicates_count}  (Discarded Minor Sub-Clusters)")
    print(f" 'mua' Multi-Units:           {mua_count}")
    print("=" * 85)

    if len(df_dedup) > 0:
        print("\nTop 15 Deduplication Actions (Dominant vs Discarded Duplicate):")
        print(df_dedup.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
