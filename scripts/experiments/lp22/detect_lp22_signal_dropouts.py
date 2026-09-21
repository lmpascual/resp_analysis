#!/usr/bin/env python3
r"""
Signal Dropout and Artifact Scanner for Neuropixels AP Recording
Dataset: LP22_fent_npx_EXPT_g0 (Duration: ~87.6 minutes)

Scans the 384-channel recording at high temporal resolution (0.1s - 0.5s bins)
to identify periods where the signal cut out completely (flatline / near-zero variance / sync lost).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spikeinterface.extractors as se

def format_min_sec(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

def detect_dropouts(raw_dir, out_dir, bin_dur_s=0.2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(" SCANNING FOR SIGNAL DROPOUTS & CUTOUT WINDOWS")
    print(f" Recording: {raw_dir}")
    print("=" * 80)
    
    rec = se.read_spikeglx(raw_dir, stream_id='imec0.ap')
    fs = rec.get_sampling_frequency()
    total_samples = rec.get_num_samples()
    total_dur_s = rec.get_total_duration()
    n_chans = rec.get_num_channels()
    
    bin_samples = int(bin_dur_s * fs)
    n_bins = total_samples // bin_samples
    
    # Subsample subset of channels across shank for fast scanning (e.g. 32 evenly spaced channels)
    scan_chans = np.linspace(0, n_chans - 1, 32, dtype=int)
    
    print(f" Scanning {n_bins:,} time bins ({bin_dur_s}s per bin) across {len(scan_chans)} channels...")
    
    bin_times_s = []
    bin_std_uv = []
    bin_is_flat = []
    
    # Read in chunks of 500 bins (100 seconds) for speed
    chunk_size_bins = 500
    n_chunks = int(np.ceil(n_bins / chunk_size_bins))
    
    for c in range(n_chunks):
        b_start = c * chunk_size_bins
        b_end = min(n_bins, (c + 1) * chunk_size_bins)
        
        f_start = b_start * bin_samples
        f_end = b_end * bin_samples
        
        traces = rec.get_traces(start_frame=f_start, end_frame=f_end, channel_ids=rec.channel_ids[scan_chans])
        # Convert to uV using gain
        gains = rec.get_channel_gains()[scan_chans]
        traces_uv = traces * gains
        
        # Reshape into bins
        cur_bins = b_end - b_start
        actual_len = cur_bins * bin_samples
        traces_reshaped = traces_uv[:actual_len].reshape(cur_bins, bin_samples, len(scan_chans))
        
        # Compute std per bin across time and mean across channels
        stds = np.mean(np.std(traces_reshaped, axis=1), axis=1)
        
        for bi in range(cur_bins):
            t_bin = (b_start + bi) * bin_dur_s
            bin_times_s.append(t_bin)
            bin_std_uv.append(stds[bi])
            # Dropout threshold: std < 1.0 uV (normal noise floor is ~6-15 uV)
            bin_is_flat.append(stds[bi] < 1.5)
            
        if (c + 1) % 10 == 0 or (c + 1) == n_chunks:
            print(f"  Processed {min(b_end * bin_dur_s, total_dur_s):.1f} / {total_dur_s:.1f} s ({min(b_end * bin_dur_s, total_dur_s)/60:.1f} min)...")
            
    bin_times_s = np.array(bin_times_s)
    bin_std_uv = np.array(bin_std_uv)
    bin_is_flat = np.array(bin_is_flat)
    
    # Identify contiguous dropout segments
    dropouts = []
    in_dropout = False
    start_t = 0.0
    
    for i, (t, flat) in enumerate(zip(bin_times_s, bin_is_flat)):
        if flat and not in_dropout:
            in_dropout = True
            start_t = t
        elif not flat and in_dropout:
            in_dropout = False
            end_t = t
            dur = end_t - start_t
            if dur >= 0.4: # Filter micro-glitches < 400ms
                dropouts.append((start_t, end_t, dur))
                
    if in_dropout:
        end_t = bin_times_s[-1] + bin_dur_s
        dropouts.append((start_t, end_t, end_t - start_t))
        
    # Classify affected epoch
    epochs_def = [
        ("First 5 min (0-5m)", 0.0, 300.0),
        ("Pre-Injection Baseline (13-18m)", 780.0, 1080.0),
        ("Post-Injection 1 (18-38m)", 1080.0, 2280.0),
        ("Pre-Inj 2 Gap (38-54m)", 2280.0, 3240.0),
        ("Post-Injection 2 (54-74m)", 3240.0, 4440.0),
        ("Late Recording (74-88m)", 4440.0, total_dur_s)
    ]
    
    dropout_records = []
    for d_idx, (t0, t1, dur) in enumerate(dropouts):
        affected = []
        for ep_name, ep_t0, ep_t1 in epochs_def:
            if not (t1 < ep_t0 or t0 > ep_t1):
                affected.append(ep_name)
        
        dropout_records.append({
            'dropout_id': d_idx + 1,
            'start_time_s': round(t0, 2),
            'end_time_s': round(t1, 2),
            'start_time_min_sec': format_min_sec(t0),
            'end_time_min_sec': format_min_sec(t1),
            'duration_s': round(dur, 2),
            'affected_epoch': "; ".join(affected) if affected else "Inter-Epoch Window",
            'dropout_type': 'Complete Signal Cutout (Flatline / Loss of Signal)'
        })
        
    df_dropouts = pd.DataFrame(dropout_records)
    out_csv = out_dir / "signal_dropout_windows.csv"
    df_dropouts.to_csv(out_csv, index=False)
    print(f"\n Found {len(df_dropouts)} signal dropout windows! Saved table to: {out_csv}")
    
    # 3. Generate Diagnostic Dropout Plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Neuropixels AP Signal Quality & Dropout Windows (LP22_fent_npx_EXPT_g0)", fontsize=14, fontweight='bold')
    
    # Timeline
    t_min = bin_times_s / 60.0
    axes[0].plot(t_min, bin_std_uv, color='#2c3e50', linewidth=0.8, label='Channel Bandpass RMS / Noise Floor (µV)')
    axes[0].axhline(1.5, color='#e74c3c', linestyle='--', linewidth=1.5, label='Cutout Threshold (<1.5 µV)')
    
    # Highlight Epochs
    epoch_colors = {'First 5 min (0-5m)': '#3498db', 'Pre-Injection Baseline (13-18m)': '#2ecc71', 
                    'Post-Injection 1 (18-38m)': '#e67e22', 'Post-Injection 2 (54-74m)': '#9b59b6'}
    
    for ep_name, ep_t0, ep_t1 in epochs_def:
        if ep_name in epoch_colors:
            axes[0].axvspan(ep_t0/60.0, ep_t1/60.0, alpha=0.12, color=epoch_colors[ep_name], label=f'Epoch: {ep_name}')
            
    for _, row in df_dropouts.iterrows():
        axes[0].axvspan(row['start_time_s']/60.0, row['end_time_s']/60.0, color='#e74c3c', alpha=0.45)
        axes[1].axvspan(row['start_time_s']/60.0, row['end_time_s']/60.0, color='#e74c3c', alpha=0.8)
        
    axes[0].set_ylabel("Signal RMS (µV)", fontsize=11)
    axes[0].set_title("AP Band Signal Amplitude Profile Across Session", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Binary Dropout mask
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
    
    return df_dropouts

if __name__ == '__main__':
    raw_dir = r"Z:\ramirezlab\pascual\rec_data\LP22_fent_npx_EXPT_g0\LP22_fent_npx_EXPT_g0_imec0"
    out_dir = r"c:\Users\lpascu\analysis_src\output_LP22_fent_EXPT_g0"
    detect_dropouts(raw_dir, out_dir)
