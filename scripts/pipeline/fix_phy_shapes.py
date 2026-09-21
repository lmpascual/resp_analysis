#!/usr/bin/env python3
"""
Fix Phy file shapes & remove conflicting duplicate files for phylib compatibility:
- Converts 2D (N, 1) arrays to 1D (N,) for spike_times, spike_clusters, spike_templates, amplitudes
- Fixes off-by-one length mismatch in amplitudes.npy (3383855 vs 3383856)
- Removes duplicate ALF-style files (spikes.times.npy, etc.) to prevent phylib conflict errors
"""

import os
import numpy as np
from pathlib import Path

def fix_phy_folder():
    phy_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session\alf\probe00")
    print("=" * 85)
    print(f" Fixing Phy file shapes & removing conflicts in: {phy_dir}")
    print("=" * 85)

    # 1. Remove conflicting duplicate ALF files if present
    conflicting_alf_files = [
        "spikes.times.npy",
        "spikes.clusters.npy",
        "spikes.templates.npy",
        "channels.localCoordinates.npy"
    ]
    for f in conflicting_alf_files:
        p = phy_dir / f
        if p.exists():
            print(f" -> Removing conflicting file: {f}")
            p.unlink()

    # 2. Load spike_times
    st = np.load(phy_dir / "spike_times.npy").squeeze()
    n_spikes = len(st)
    np.save(phy_dir / "spike_times.npy", st.astype(np.int64))
    print(f" -> Saved spike_times.npy: shape = {st.shape}, dtype = {st.dtype}")

    # 3. Load spike_clusters
    sc = np.load(phy_dir / "spike_clusters.npy").squeeze()
    if len(sc) != n_spikes:
        sc = sc[:n_spikes]
    np.save(phy_dir / "spike_clusters.npy", sc.astype(np.int64))
    print(f" -> Saved spike_clusters.npy: shape = {sc.shape}, dtype = {sc.dtype}")

    # 4. Load spike_templates
    stemp = np.load(phy_dir / "spike_templates.npy").squeeze()
    if len(stemp) != n_spikes:
        stemp = stemp[:n_spikes]
    np.save(phy_dir / "spike_templates.npy", stemp.astype(np.int64))
    print(f" -> Saved spike_templates.npy: shape = {stemp.shape}, dtype = {stemp.dtype}")

    # 5. Load amplitudes & pad to n_spikes if off-by-one
    amp = np.load(phy_dir / "amplitudes.npy").squeeze()
    if len(amp) < n_spikes:
        amp = np.pad(amp, (0, n_spikes - len(amp)), mode='edge')
    elif len(amp) > n_spikes:
        amp = amp[:n_spikes]
    np.save(phy_dir / "amplitudes.npy", amp.astype(np.float32))
    print(f" -> Saved amplitudes.npy: shape = {amp.shape}, dtype = {amp.dtype}")

    # 6. Fix channel_map and channel_positions
    ch_map = np.load(phy_dir / "channel_map.npy").squeeze()
    np.save(phy_dir / "channel_map.npy", ch_map.astype(np.int32))

    ch_pos = np.load(phy_dir / "channel_positions.npy")
    np.save(phy_dir / "channel_positions.npy", ch_pos.astype(np.float32))

    print("\n" + "=" * 85)
    print(" [SUCCESS] All Phy files clean, reshaped to 1D, and conflicts removed!")
    print("=" * 85)

if __name__ == "__main__":
    fix_phy_folder()
