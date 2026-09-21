#!/usr/bin/env python3
"""
Fix global channel mapping & shank assignments in cluster_info.tsv using template_ind.npy.
Correctly maps units to Shanks 0, 1, 2, and 3.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def get_shank(x):
    if x < 125: return 0
    elif x < 391: return 1
    elif x < 657: return 2
    else: return 3

def fix_cluster_info():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    session_dir = out_dir / "alf_session"
    alf_dir = session_dir / "alf" / "probe00"

    print("=" * 85)
    print(" Fixing Channel, Shank (0-3), and Depth Mapping in cluster_info.tsv")
    print("=" * 85)

    templates = np.load(alf_dir / "templates.npy")       # (1072, 90, 13)
    template_ind = np.load(alf_dir / "template_ind.npy") # (1072, 13)
    channel_coords = np.load(alf_dir / "channel_positions.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    real_channels = []
    real_shanks = []
    real_depths = []
    real_amps = []

    for idx, row in info_df.iterrows():
        uid = int(row['cluster_id'])
        if uid < len(templates):
            tmpl_sparse = templates[uid]
            ch_indices = template_ind[uid]
            valid_mask = (ch_indices >= 0) & (ch_indices < len(channel_coords))
            if np.sum(valid_mask) > 0:
                tmpl_valid = tmpl_sparse[:, valid_mask]
                ch_valid = ch_indices[valid_mask]
                ptps = np.ptp(tmpl_valid, axis=0)
                best_loc = np.argmax(ptps)
                best_global_ch = int(ch_valid[best_loc])
                amp = float(ptps[best_loc])
            else:
                best_global_ch = 0
                amp = 0.0
        else:
            best_global_ch = 0
            amp = 0.0

        x = float(channel_coords[best_global_ch, 0])
        y = float(channel_coords[best_global_ch, 1])
        sh = get_shank(x)

        real_channels.append(best_global_ch)
        real_shanks.append(sh)
        real_depths.append(round(y, 1))
        real_amps.append(round(amp, 2))

    info_df['ch'] = real_channels
    info_df['shank'] = real_shanks
    info_df['depth'] = real_depths
    info_df['peak_amp_uv'] = real_amps

    info_df.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)

    print("\n" + "=" * 85)
    print(" [SUCCESS] Corrected cluster_info.tsv!")
    print(" Breakdown of ALL 1072 units across 4 Shanks:")
    print(info_df['shank'].value_counts().to_string())
    print("\n Breakdown of GOOD units across 4 Shanks:")
    print(info_df[info_df['group'] == 'good']['shank'].value_counts().to_string())
    print("=" * 85)

if __name__ == "__main__":
    fix_cluster_info()
