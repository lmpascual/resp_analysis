#!/usr/bin/env python3
r"""
Ultra-Fast Signal Dropout & Cutout Scanner for Neuropixels AP Recording
Uses Direct Binary Memory Mapping (np.memmap) for sub-second scanning across 87.6 minutes.
Dataset: LP22_fent_npx_EXPT_g0
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def format_min_sec(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

def main():
    bin_path = Path(r"Z:\ramirezlab\pascual\rec_data\LP22_fent_npx_EXPT_g0\LP22_fent_npx_EXPT_g0_imec0\LP22_fent_npx_EXPT_g0_t0.imec0.ap.bin")
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_LP22_fent_EXPT_g0")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fs = 29999.8
    n_chans = 384
    file_size_bytes = bin_path.stat().st_size
    n_samples = file_size_bytes // (n_chans * 2)
    total_dur_s = n_samples / fs
    
    print("=" * 85)
    print(" FAST SIGNAL DROPOUT & ARTIFACT SCANNER (np.memmap)")
    print(f" Binary Path: {bin_path}")
    print(f" Samples: {n_samples:,} | Duration: {total_dur_s:.2f} s ({total_dur_s/60:.2f} min)")
    print("=" * 85)
    
    mm = np.memmap(bin_path, dtype=np.int16, mode='r', shape=(n_samples, n_chans))
    
    # 0.2s bin analysis across 32 representative channels
    bin_dur_s = 0.2
    bin_samples = int(bin_dur_s * fs)
    n_bins = n_samples // bin_samples
    
    scan_chans = np.linspace(0, n_chans - 1, 32, dtype=int)
    
    print(f" Scanning {n_bins:,} temporal bins ({bin_dur_s}s resolution) across {len(scan_chans)} channels...")
    
    t0 = time.time()
    # Read chunk by chunk (10,000 bins = 2000s per chunk) to avoid memory overhead
    chunk_bins = 5000
    n_chunks = int(np.ceil(n_bins / chunk_bins))
    
    bin_stds = []
    bin_times = []
    
    for c in range(n_chunks):
        b0 = c * chunk_bins
        b1 = min(n_bins, (c + 1) * chunk_bins)
        
        f0 = b0 * bin_samples
        f1 = b1 * bin_samples
        
        cur_bins = b1 - b0
        chunk_data = mm[f0:f1, scan_chans] # (cur_bins * bin_samples, 32)
        
        # Reshape to (cur_bins, bin_samples, 32)
        chunk_reshaped = chunk_data.reshape(cur_bins, bin_samples, len(scan_chans))
        # Compute std per bin across time for each channel, then mean across channels
        stds = np.mean(np.std(chunk_reshaped, axis=1), axis=1) # in raw int16 counts (~0.6 uV/LSB)
        
        for bi in range(cur_bins):
            bin_times.append((b0 + bi) * bin_dur_s)
            bin_stds.append(stds[bi] * 0.6) # convert LSB to uV
            
    bin_times = np.array(bin_times)
    bin_stds = np.array(bin_stds)
    print(f" Scan completed in {time.time() - t0:.2f} seconds!")
    
    # Signal dropout threshold: std < 2.0 uV (normal noise floor is 8-20 uV)
    is_flat = bin_stds < 2.0
    
    dropouts = []
    in_dropout = False
    start_t = 0.0
    
    for i, (t, flat) in enumerate(zip(bin_times, is_flat)):
        if flat and not in_dropout:
            in_dropout = True
            start_t = t
        elif not flat and in_dropout:
            in_dropout = False
            end_t = t
            dur = end_t - start_t
            if dur >= 0.4:
                dropouts.append((start_t, end_t, dur))
                
    if in_dropout:
        end_t = bin_times[-1] + bin_dur_s
        dropouts.append((start_t, end_t, end_t - start_t))
        
    epochs_def = [
        ("Epoch 1: First 5 min (0-5m)", 0.0, 300.0),
        ("Epoch 2: Pre-Inj Baseline (13-18m)", 780.0, 1080.0),
        ("Epoch 3: Post-Injection 1 (18-38m)", 1080.0, 2280.0),
        ("Pre-Inj 2 Gap (38-54m)", 2280.0, 3240.0),
        ("Epoch 4: Post-Injection 2 (54-74m)", 3240.0, 4440.0),
        ("Late Recording (74-88m)", 4440.0, total_dur_s)
    ]
    
    dropout_records = []
    for d_idx, (d_t0, d_t1, dur) in enumerate(dropouts):
        affected = []
        for ep_name, ep_t0, ep_t1 in epochs_def:
            if not (d_t1 < ep_t0 or d_t0 > ep_t1):
                affected.append(ep_name)
                
        dropout_records.append({
            'dropout_id': d_idx + 1,
            'start_time_s': round(d_t0, 2),
            'end_time_s': round(d_t1, 2),
            'start_time_min_sec': format_min_sec(d_t0),
            'end_time_min_sec': format_min_sec(d_t1),
            'duration_s': round(dur, 2),
            'affected_epoch': "; ".join(affected) if affected else "Inter-Epoch Window",
            'dropout_type': 'Complete Signal Cutout (Flatline / Near-Zero Noise)'
        })
        
    df_dropouts = pd.DataFrame(dropout_records)
    out_csv = out_dir / "signal_dropout_windows.csv"
    df_dropouts.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 80)
    print(f" DETECTED {len(df_dropouts)} SIGNAL DROPOUT / CUTOUT WINDOWS")
    print("=" * 80)
    print(df_dropouts[['dropout_id', 'start_time_min_sec', 'end_time_min_sec', 'duration_s', 'affected_epoch']].to_string(index=False))
    print(f"\n Saved full cutout table to: {out_csv}")
    
    # Generate Timeline Figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Neuropixels AP Signal Quality & Dropout Windows (LP22_fent_npx_EXPT_g0)", fontsize=14, fontweight='bold')
    
    t_min = bin_times / 60.0
    axes[0].plot(t_min, bin_stds, color='#2c3e50', linewidth=0.7, label='Channel RMS / Noise Floor (µV)')
    axes[0].axhline(2.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Cutout Threshold (<2.0 µV)')
    
    epoch_colors = {
        'Epoch 1: First 5 min (0-5m)': '#3498db',
        'Epoch 2: Pre-Inj Baseline (13-18m)': '#2ecc71',
        'Epoch 3: Post-Injection 1 (18-38m)': '#e67e22',
        'Epoch 4: Post-Injection 2 (54-74m)': '#9b59b6'
    }
    
    for ep_name, ep_t0, ep_t1 in epochs_def:
        if ep_name in epoch_colors:
            axes[0].axvspan(ep_t0/60.0, ep_t1/60.0, alpha=0.15, color=epoch_colors[ep_name], label=ep_name)
            
    for _, row in df_dropouts.iterrows():
        axes[0].axvspan(row['start_time_s']/60.0, row['end_time_s']/60.0, color='#e74c3c', alpha=0.5)
        axes[1].axvspan(row['start_time_s']/60.0, row['end_time_s']/60.0, color='#e74c3c', alpha=0.9)
        
    axes[0].set_ylabel("Signal RMS (µV)", fontsize=11)
    axes[0].set_title("AP Band Signal Amplitude Profile Across Entire Recording Session", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].set_yticks([])
    axes[1].set_ylabel("Dropouts", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Recording Time (Minutes)", fontsize=12)
    axes[1].grid(axis='x', linestyle=':', alpha=0.6)
    axes[1].set_xlim(0, total_dur_s / 60.0)
    
    out_fig = out_dir / "lp22_signal_dropout_timeline.png"
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f" Saved dropout timeline figure to: {out_fig}")

if __name__ == '__main__':
    main()
