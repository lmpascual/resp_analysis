#!/usr/bin/env python3
"""
Generate 3-Trial Laser Train PETHs (-10s Pre-Laser to +10s Post-Laser Offset) for e097:
- Phase 1: 10 seconds Pre-Laser Eupnea [-10.0s, 0.0s]
- Phase 2: During Laser Photostimulation [0.0s, 1.955s] (20 Hz, 40 pulses, 5 ms each, central apnea)
- Phase 3: 10 seconds Post-Laser Train Offset [1.955s, 11.955s]
  * Sub-phase 3a: Post-apneic transition / sigh / delayed resumption [1.955s, 5.955s] (~4s post-offset)
  * Sub-phase 3b: Established post-resumption normal breathing [5.955s, 11.955s] (6s)

Outputs:
1. Multi-panel publication figures (PNG + SVG) for:
   - Recruited Rostral Units (Cluster 315, 13, 343, 366, 61, 269)
   - Baseline Archetypes & Surge Units (Cluster 430, 298, 448, 18, 93, 82)
   - Recruited Caudal Units (Cluster 316, 354, 10, 19, 387, 74)
   - Population 3-Trial Side-by-Side Heatmaps (All 162 Units)
2. Comprehensive quantitative summary CSV: laser_train_3trial_peth_summary.csv
3. Upgraded Interactive HTML Workbench with dedicated 3-trial raster, rate overlays, and bar charts.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    print("[*] Starting 3-Trial Laser Train PETH analysis (-10s Pre-Onset to +10s Post-Offset)...")
    base_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    alf_dir = base_dir / "alf"
    out_dir = base_dir / "peth_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Units & Spikes
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    pr_df = pd.read_csv(base_dir / "detailed_presence_ratio_comparison.csv")

    target_cids = list(pr_df["cluster_id"].values)
    if 448 not in target_cids and 448 in spk_clusters:
        target_cids.append(448)

    print(f"Total units to analyze: {len(target_cids)} units.")

    # 2. Load NIDQ Analog Data
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(laser_v)) / fs
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)

    # Process EMG envelope
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    emg_env = emg_proc.envelope

    # 3. Detect Laser Trains
    above = laser_v > 0.5
    diff = np.diff(above.astype(np.int32))
    pulse_starts = t_vec[np.where(diff == 1)[0] + 1]

    train_break_indices = np.where(np.diff(pulse_starts) > 1.0)[0]
    train_start_indices = [0] + [b + 1 for b in train_break_indices]
    train_starts = [pulse_starts[idx] for idx in train_start_indices]
    train_dur = 1.955
    post_offset_dur = 10.0
    t_pre = -10.0
    t_post = train_dur + post_offset_dur  # 11.955 s

    print(f"Detected 3 laser trains at:")
    for i, ts in enumerate(train_starts, 1):
        print(f"  Train {i}: start = {ts:.4f}s")
    print(f"Window: [{t_pre:.1f}s, +{t_post:.3f}s] (Total duration = {t_post - t_pre:.3f}s)")

    # 4. Extract Physiological Traces for each Train (-10.0s to +11.955s)
    # Downsample to 1000 Hz for efficient plotting
    ds_factor = 10
    t_rel = np.arange(t_pre, t_post, 1.0 / (fs / ds_factor))
    flow_trials = []
    emg_trials = []
    laser_trials = []

    for ts in train_starts:
        idx_start = int((ts + t_pre) * fs)
        idx_end = int((ts + t_post) * fs)
        flow_trials.append(flow_v[idx_start:idx_end:ds_factor][:len(t_rel)])
        emg_trials.append(emg_env[idx_start:idx_end:ds_factor][:len(t_rel)])
        laser_trials.append(laser_v[idx_start:idx_end:ds_factor][:len(t_rel)])

    # =========================================================================
    # 5. COMPUTE 3-TRAIN PETH FOR EACH UNIT
    # =========================================================================
    # Window: -10.0s to +11.955s, 50 ms bins
    bin_dt = 0.050
    t_bins = np.arange(t_pre, t_post + bin_dt, bin_dt)
    bin_centers = (t_bins[:-1] + t_bins[1:]) / 2.0
    n_bins = len(bin_centers)

    # Smooth kernel for firing rates (75 ms Gaussian)
    g_kernel = signal.windows.gaussian(5, std=1.2)
    g_kernel /= np.sum(g_kernel)

    unit_3trial_dict = {}
    summary_records = []

    for cid in target_cids:
        row_match = pr_df[pr_df["cluster_id"] == cid]
        if len(row_match) > 0:
            row = row_match.iloc[0]
            depth = float(row["depth_um"])
            is_rostral = bool(row.get("is_rostral_vrc", depth <= 1080.0))
            is_recruited = bool(row.get("is_newly_recruited", False))
        else:
            depth = 1380.0
            is_rostral = False
            is_recruited = False

        u_spks = spk_times[spk_clusters == cid]

        # Firing rates and rasters per trial
        trials_fr = np.zeros((3, n_bins), dtype=np.float32)
        trial_rasters = {0: [], 1: [], 2: []}
        
        pre_rates = []
        laser_rates = []
        post_rates = []
        post_trans_rates = []
        post_resumed_rates = []

        for tr_i, ts in enumerate(train_starts):
            rel = u_spks[(u_spks >= ts + t_pre) & (u_spks <= ts + t_post)] - ts
            trial_rasters[tr_i] = rel
            
            counts, _ = np.histogram(rel, bins=t_bins)
            smooth_fr = np.convolve(counts / bin_dt, g_kernel, mode="same")
            trials_fr[tr_i, :] = smooth_fr

            # Quantifications for the discrete physiological windows:
            # 1. Pre-Laser (10s baseline): [-10.0, 0.0]
            n_pre = np.sum((rel >= -10.0) & (rel < 0.0))
            fr_pre = n_pre / 10.0
            pre_rates.append(fr_pre)

            # 2. During Laser (1.955s apnea): [0.0, 1.955]
            n_laser = np.sum((rel >= 0.0) & (rel < train_dur))
            fr_laser = n_laser / train_dur
            laser_rates.append(fr_laser)

            # 3. Post-Laser (10s post offset): [1.955, 11.955]
            n_post = np.sum((rel >= train_dur) & (rel < t_post))
            fr_post = n_post / post_offset_dur
            post_rates.append(fr_post)

            # 3a. Post-Laser Transition (4s post offset, sigh/delayed recovery): [1.955, 5.955]
            n_trans = np.sum((rel >= train_dur) & (rel < train_dur + 4.0))
            fr_trans = n_trans / 4.0
            post_trans_rates.append(fr_trans)

            # 3b. Post-Laser Resumption (subsequent 6s normal breathing): [5.955, 11.955]
            n_resumed = np.sum((rel >= train_dur + 4.0) & (rel < t_post))
            fr_resumed = n_resumed / 6.0
            post_resumed_rates.append(fr_resumed)

        mean_fr = np.mean(trials_fr, axis=0)
        sem_fr = np.std(trials_fr, axis=0) / np.sqrt(3)

        unit_3trial_dict[cid] = {
            "depth_um": depth,
            "is_rostral": is_rostral,
            "is_recruited": is_recruited,
            "bin_centers": bin_centers,
            "trials_fr": trials_fr,
            "mean_fr": mean_fr,
            "sem_fr": sem_fr,
            "trial_rasters": trial_rasters,
            "pre_rates": pre_rates,
            "laser_rates": laser_rates,
            "post_rates": post_rates,
            "post_trans_rates": post_trans_rates,
            "post_resumed_rates": post_resumed_rates,
            "mean_pre": np.mean(pre_rates),
            "mean_laser": np.mean(laser_rates),
            "mean_post": np.mean(post_rates),
            "mean_trans": np.mean(post_trans_rates),
            "mean_resumed": np.mean(post_resumed_rates),
        }

        summary_records.append({
            "cluster_id": cid,
            "depth_um": depth,
            "is_rostral": is_rostral,
            "is_recruited": is_recruited,
            
            # Train 1
            "t1_pre_10s_fr": round(pre_rates[0], 2),
            "t1_laser_fr": round(laser_rates[0], 2),
            "t1_post_10s_fr": round(post_rates[0], 2),
            "t1_post_trans_4s_fr": round(post_trans_rates[0], 2),
            "t1_post_resumed_6s_fr": round(post_resumed_rates[0], 2),
            "t1_laser_ratio": round(laser_rates[0] / (pre_rates[0] + 0.05), 2),
            "t1_post_ratio": round(post_rates[0] / (pre_rates[0] + 0.05), 2),
            
            # Train 2
            "t2_pre_10s_fr": round(pre_rates[1], 2),
            "t2_laser_fr": round(laser_rates[1], 2),
            "t2_post_10s_fr": round(post_rates[1], 2),
            "t2_post_trans_4s_fr": round(post_trans_rates[1], 2),
            "t2_post_resumed_6s_fr": round(post_resumed_rates[1], 2),
            "t2_laser_ratio": round(laser_rates[1] / (pre_rates[1] + 0.05), 2),
            "t2_post_ratio": round(post_rates[1] / (pre_rates[1] + 0.05), 2),
            
            # Train 3
            "t3_pre_10s_fr": round(pre_rates[2], 2),
            "t3_laser_fr": round(laser_rates[2], 2),
            "t3_post_10s_fr": round(post_rates[2], 2),
            "t3_post_trans_4s_fr": round(post_trans_rates[2], 2),
            "t3_post_resumed_6s_fr": round(post_resumed_rates[2], 2),
            "t3_laser_ratio": round(laser_rates[2] / (pre_rates[2] + 0.05), 2),
            "t3_post_ratio": round(post_rates[2] / (pre_rates[2] + 0.05), 2),
            
            # Means across trials
            "mean_pre_10s_fr": round(float(np.mean(pre_rates)), 2),
            "mean_laser_fr": round(float(np.mean(laser_rates)), 2),
            "mean_post_10s_fr": round(float(np.mean(post_rates)), 2),
            "mean_post_trans_4s_fr": round(float(np.mean(post_trans_rates)), 2),
            "mean_post_resumed_6s_fr": round(float(np.mean(post_resumed_rates)), 2),
            "mean_laser_ratio": round(float(np.mean(laser_rates)) / (float(np.mean(pre_rates)) + 0.05), 2),
        })

    # Save summary CSV
    df_summary = pd.DataFrame(summary_records).sort_values(by="depth_um").reset_index(drop=True)
    csv_path = out_dir / "laser_train_3trial_peth_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"[+] Exported 3-trial summary CSV to: {csv_path}")

    # =========================================================================
    # 6. RENDER PUBLICATION-QUALITY 3-TRIAL FIGURES
    # =========================================================================
    print("[*] Rendering multi-panel 3-trial PETH figures (-10s to +12s)...")

    # Group 1: Recruited Rostral Units
    render_3trial_panel_figure(
        unit_ids=[315, 13, 343, 366, 61, 269],
        unit_dict=unit_3trial_dict,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur,
        title="RECRUITED ROSTRAL UNITS: 3-Trial Laser Train PETHs (-10s Pre-Onset to +10s Post-Offset)",
        out_png=out_dir / "peth_3trials_recruited_rostral_units.png",
        out_svg=out_dir / "peth_3trials_recruited_rostral_units.svg"
    )

    # Group 2: Baseline Archetypes & Laser Surge Units
    render_3trial_panel_figure(
        unit_ids=[430, 298, 448, 18, 93, 82],
        unit_dict=unit_3trial_dict,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur,
        title="BASELINE ARCHETYPES & LASER SURGE: 3-Trial Laser Train PETHs (-10s Pre-Onset to +10s Post-Offset)",
        out_png=out_dir / "peth_3trials_baseline_archetypes.png",
        out_svg=out_dir / "peth_3trials_baseline_archetypes.svg"
    )

    # Group 3: Recruited Caudal Units
    render_3trial_panel_figure(
        unit_ids=[316, 354, 10, 19, 387, 74],
        unit_dict=unit_3trial_dict,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur,
        title="RECRUITED CAUDAL UNITS (cVRG): 3-Trial Laser Train PETHs (-10s Pre-Onset to +10s Post-Offset)",
        out_png=out_dir / "peth_3trials_recruited_caudal_units.png",
        out_svg=out_dir / "peth_3trials_recruited_caudal_units.svg"
    )

    # Group 4: Population Heatmaps across Train 1, Train 2, Train 3
    render_population_3trial_heatmaps(
        unit_dict=unit_3trial_dict,
        bin_centers=bin_centers,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur,
        out_png=out_dir / "population_3trials_heatmaps.png",
        out_svg=out_dir / "population_3trials_heatmaps.svg"
    )

    # =========================================================================
    # 7. UPDATE INTERACTIVE HTML WORKBENCH
    # =========================================================================
    print("[*] Generating comprehensive interactive HTML workbench with -10s to +10s post-offset view...")
    generate_upgraded_interactive_workbench(
        unit_3trial_dict=unit_3trial_dict,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur,
        out_html=out_dir / "e097_laser_train_3trial_peth_interactive_workbench.html"
    )

    print("[DONE] 3-Trial Laser Train PETH analysis (-10s Pre to +10s Post-Offset) successfully generated and exported!")

def render_3trial_panel_figure(unit_ids, unit_dict, t_rel, flow_trials, emg_trials, t_pre, t_post, train_dur, title, out_png, out_svg):
    """
    Renders a publication figure for a group of units:
    For each unit (row):
    - Col 1: 3-Trial Spike Raster (Trial 1 blue, Trial 2 green, Trial 3 red) across [-10s, +12s]
             with overlaid Diaphragm EMG / Airflow in background and phase markers.
    - Col 2: Continuous Firing Rate Overlays (T1, T2, T3, and Mean ± SEM) across [-10s, +12s]
             with shaded phases: Pre-Laser [-10s, 0s], Laser [0s, 1.955s], Transition [1.955s, 5.955s], Resumption [5.955s, 12s].
    - Col 3: Grouped Bar Chart comparing Pre (10s), During Laser (2s), Post Trans (4s), and Resumed (6s) rates
             for Train 1, Train 2, and Train 3!
    """
    n_units = len(unit_ids)
    fig, axes = plt.subplots(n_units, 3, figsize=(20, 2.9 * n_units), gridspec_kw={'width_ratios': [1.35, 1.35, 0.9]})
    fig.patch.set_facecolor('white')
    if n_units == 1:
        axes = np.array([axes])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    trial_colors = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red
    t_resump = train_dur + 4.0 # 5.955 s

    for row_idx, uid in enumerate(unit_ids):
        u = unit_dict[uid]
        depth = u["depth_um"]

        name_str = f"Cluster {uid} ({depth:.0f} μm)"
        if uid == 430: name_str += " [Unit 2 Pacemaker]"
        elif uid == 298: name_str += " [Unit 255 Ramping]"
        elif uid == 448: name_str += " [Unit 163 Pre-I]"
        elif uid == 18: name_str += " [Unit 18 BötC Exp]"
        elif uid == 93: name_str += " [Unit 70 Vanguard]"
        elif uid == 315: name_str += " [Recruited BötC]"
        elif uid == 82: name_str += " [Laser Surge]"

        # --- Col 1: 3-Trial Raster & Flow Overlay ---
        ax1 = axes[row_idx, 0]
        ax1.set_facecolor('#fafafa')
        
        # Phase shadings
        ax1.axvspan(t_pre, 0.0, color='#e9ecef', alpha=0.45, label='Pre-Laser (10s)')
        ax1.axvspan(0.0, train_dur, color='#ffe8cc', alpha=0.6, label='Laser ON (2s)')
        ax1.axvspan(train_dur, t_resump, color='#fff3bf', alpha=0.45, label='Transition (~4s)')
        ax1.axvspan(t_resump, t_post, color='#e6fcf5', alpha=0.45, label='Resumed Normal (6s)')
        ax1.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.2, alpha=0.8)

        # Background Airflow (Train 3 flow in light gray)
        norm_flow = (flow_trials[2] - np.min(flow_trials[2])) / (np.max(flow_trials[2]) - np.min(flow_trials[2]) + 1e-5) * 2.8 + 0.6
        ax1.plot(t_rel, norm_flow, color='#adb5bd', alpha=0.4, linewidth=0.8, label='Airflow (T3)')

        # Plot spikes for each trial on rows y=1, 2, 3
        for tr_i in range(3):
            spks = u["trial_rasters"][tr_i]
            if len(spks) > 0:
                ax1.scatter(spks, np.full_like(spks, tr_i + 1), s=10, color=trial_colors[tr_i], marker='|', linewidth=1.2, alpha=0.9)

        ax1.set_xlim(t_pre, t_post)
        ax1.set_ylim(0.4, 3.6)
        ax1.set_yticks([1, 2, 3])
        ax1.set_yticklabels(['Train 1', 'Train 2', 'Train 3'], fontsize=8, fontweight='bold')
        ax1.set_ylabel(f"{name_str}", fontsize=8, fontweight='bold')
        if row_idx == 0:
            ax1.set_title("3-Trial Spike Raster (-10s Pre to +10s Post-Offset)\nBlue: T1 | Green: T2 | Red: T3 | Green Line: Breathing Resumption (~4s post-offset)", fontsize=9, fontweight='bold')
        if row_idx == n_units - 1:
            ax1.set_xlabel("Time from Laser Train Onset (s)", fontsize=9, fontweight='bold')

        # --- Col 2: Firing Rate Overlays ---
        ax2 = axes[row_idx, 1]
        ax2.set_facecolor('#fafafa')
        ax2.axvspan(t_pre, 0.0, color='#e9ecef', alpha=0.45)
        ax2.axvspan(0.0, train_dur, color='#ffe8cc', alpha=0.6)
        ax2.axvspan(train_dur, t_resump, color='#fff3bf', alpha=0.45)
        ax2.axvspan(t_resump, t_post, color='#e6fcf5', alpha=0.45)
        ax2.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.2, alpha=0.8)

        # Plot individual trials
        for tr_i in range(3):
            ax2.plot(u["bin_centers"], u["trials_fr"][tr_i], color=trial_colors[tr_i], linewidth=1.0, alpha=0.75, label=f'Train {tr_i+1}')
        
        # Plot mean
        ax2.plot(u["bin_centers"], u["mean_fr"], color='black', linewidth=1.8, label='Mean FR')
        ax2.fill_between(
            u["bin_centers"],
            np.maximum(0, u["mean_fr"] - u["sem_fr"]),
            u["mean_fr"] + u["sem_fr"],
            color='black', alpha=0.15
        )

        ax2.set_xlim(t_pre, t_post)
        max_y2 = max(5.0, np.max(u["trials_fr"]) * 1.15)
        ax2.set_ylim(0, max_y2)
        ax2.set_ylabel("Firing Rate (Hz)", fontsize=8)
        if row_idx == 0:
            ax2.set_title("Firing Rate Dynamics (-10s to +12s)\nGray: Pre (10s) | Orange: Laser (2s) | Yellow: Trans (4s) | Teal: Resumed (6s)", fontsize=9, fontweight='bold')
        if row_idx == n_units - 1:
            ax2.set_xlabel("Time from Laser Train Onset (s)", fontsize=9, fontweight='bold')

        # Annotation badge
        ax2.text(0.97, 0.90, f"T3: Pre={u['pre_rates'][2]:.1f}, Laser={u['laser_rates'][2]:.1f}, Trans={u['post_trans_rates'][2]:.1f}, Resumed={u['post_resumed_rates'][2]:.1f} Hz",
                 transform=ax2.transAxes, ha='right', va='top', fontsize=7.5,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#ccc'))

        # --- Col 3: 4-Phase Bar Chart (Pre vs Laser vs Trans vs Resumed) ---
        ax3 = axes[row_idx, 2]
        ax3.set_facecolor('#fafafa')
        
        bar_width = 0.19
        x_indices = np.array([0, 1, 2]) # Train 1, Train 2, Train 3

        pre_vals = u["pre_rates"]
        laser_vals = u["laser_rates"]
        trans_vals = u["post_trans_rates"]
        resumed_vals = u["post_resumed_rates"]

        b1 = ax3.bar(x_indices - 1.5 * bar_width, pre_vals, width=bar_width, color='#adb5bd', label='Pre (10s)')
        b2 = ax3.bar(x_indices - 0.5 * bar_width, laser_vals, width=bar_width, color='#ff922b', label='Laser (2s)')
        b3 = ax3.bar(x_indices + 0.5 * bar_width, trans_vals, width=bar_width, color='#fcc419', label='Trans (4s)')
        b4 = ax3.bar(x_indices + 1.5 * bar_width, resumed_vals, width=bar_width, color='#20c997', label='Resumed (6s)')

        ax3.set_xticks(x_indices)
        ax3.set_xticklabels(['Train 1', 'Train 2', 'Train 3'], fontsize=8, fontweight='bold')
        ax3.set_ylabel("Avg Rate (Hz)", fontsize=8)
        max_y3 = max(5.0, max(pre_vals + laser_vals + trans_vals + resumed_vals) * 1.2)
        ax3.set_ylim(0, max_y3)
        if row_idx == 0:
            ax3.set_title("Phase Comparison (Hz)\nPre (10s) | Laser (2s) | Trans (4s) | Resumed (6s)", fontsize=8.5, fontweight='bold')
            ax3.legend(loc='upper left', fontsize=6.5, framealpha=0.85)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved figure to {out_png.name} and {out_svg.name}")

def render_population_3trial_heatmaps(unit_dict, bin_centers, t_rel, flow_trials, emg_trials, t_pre, t_post, train_dur, out_png, out_svg):
    """
    Renders 3 side-by-side population heatmaps for Train 1, Train 2, and Train 3
    across [-10s to +12s], sorted by probe depth.
    """
    sorted_uids = sorted(unit_dict.keys(), key=lambda uid: unit_dict[uid]["depth_um"])
    depths = [unit_dict[uid]["depth_um"] for uid in sorted_uids]

    n_units = len(sorted_uids)
    t_resump = train_dur + 4.0
    
    # Construct rate matrices (n_units x n_bins) for each train
    matrix_t1 = np.zeros((n_units, len(bin_centers)))
    matrix_t2 = np.zeros((n_units, len(bin_centers)))
    matrix_t3 = np.zeros((n_units, len(bin_centers)))

    for i, uid in enumerate(sorted_uids):
        # Normalize each unit to its max rate across the 3 trains to visualize relative modulation
        t1_r = unit_dict[uid]["trials_fr"][0]
        t2_r = unit_dict[uid]["trials_fr"][1]
        t3_r = unit_dict[uid]["trials_fr"][2]
        max_r = max(1.0, np.max([t1_r, t2_r, t3_r]))
        matrix_t1[i, :] = t1_r / max_r
        matrix_t2[i, :] = t2_r / max_r
        matrix_t3[i, :] = t3_r / max_r

    fig = plt.figure(figsize=(20, 9.5))
    fig.patch.set_facecolor('white')

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.25, 1.0], hspace=0.15, wspace=0.12)

    titles = ["TRAIN 1 (t = 60.0s)", "TRAIN 2 (t = 119.7s)", "TRAIN 3 (t = 164.3s)"]
    matrices = [matrix_t1, matrix_t2, matrix_t3]

    for col in range(3):
        # Top panel: Airflow and EMG
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.set_facecolor('#fafafa')
        ax_top.axvspan(t_pre, 0.0, color='#e9ecef', alpha=0.45)
        ax_top.axvspan(0.0, train_dur, color='#ffe8cc', alpha=0.6)
        ax_top.axvspan(train_dur, t_resump, color='#fff3bf', alpha=0.45)
        ax_top.axvspan(t_resump, t_post, color='#e6fcf5', alpha=0.45)
        ax_top.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.2, alpha=0.8)
        
        ax_top.plot(t_rel, flow_trials[col], color='#009688', linewidth=1.1, label='Airflow')
        ax_top.plot(t_rel, emg_trials[col] * 4 - 2.5, color='#3f51b5', linewidth=0.9, alpha=0.8, label='Diaphragm EMG')
        ax_top.set_xlim(t_pre, t_post)
        ax_top.set_title(titles[col], fontsize=11, fontweight='bold')
        if col == 0:
            ax_top.set_ylabel("Airflow / EMG (V)", fontsize=9, fontweight='bold')
            ax_top.legend(loc='upper right', fontsize=7.5)
        ax_top.set_xticklabels([])

        # Bottom panel: Heatmap
        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.axvspan(0.0, train_dur, color='orange', alpha=0.15)
        ax_bot.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.2, alpha=0.8)
        im = ax_bot.imshow(
            matrices[col], aspect='auto', origin='upper', cmap='viridis',
            extent=[t_pre, t_post, depths[-1], depths[0]], vmin=0, vmax=1.0
        )
        ax_bot.set_xlim(t_pre, t_post)
        ax_bot.set_xlabel("Time from Train Onset (s)", fontsize=10, fontweight='bold')
        if col == 0:
            ax_bot.set_ylabel("Probe Depth (μm)\n(Rostral / BötC → Caudal / cVRG)", fontsize=10, fontweight='bold')
            ax_bot.axhline(600, color='white', linestyle='--', linewidth=0.8)
            ax_bot.text(t_pre + 0.3, 580, "BötC", color='white', fontsize=8, fontweight='bold')
            ax_bot.axhline(1400, color='white', linestyle='--', linewidth=0.8)
            ax_bot.text(t_pre + 0.3, 1380, "preBötC", color='white', fontsize=8, fontweight='bold')
            ax_bot.axhline(2200, color='white', linestyle='--', linewidth=0.8)
            ax_bot.text(t_pre + 0.3, 2180, "cVRG", color='white', fontsize=8, fontweight='bold')
        else:
            ax_bot.set_yticklabels([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Firing Rate", fontsize=9, fontweight='bold')

    plt.suptitle("POPULATION 3-TRAIN HEATMAP DYNAMICS: Pre (10s), Laser Apnea (2s), Transition (4s), Resumed Eupnea (6s)", fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved population 3-trial heatmaps to {out_png.name}")

def generate_upgraded_interactive_workbench(unit_3trial_dict, t_rel, flow_trials, emg_trials, t_pre, t_post, train_dur, out_html):
    """
    Generates an interactive HTML workbench allowing the user to inspect
    any unit's 3-trial rasters, continuous firing rates, and pre/intra/post phase metrics across [-10s, +12s].
    """
    units_data = {}
    for uid, u in unit_3trial_dict.items():
        units_data[str(uid)] = {
            "cid": int(uid),
            "depth": round(u["depth_um"], 1),
            "is_rostral": bool(u["is_rostral"]),
            "is_recruited": bool(u["is_recruited"]),
            
            "t1_spks": [round(float(x), 3) for x in u["trial_rasters"][0]],
            "t2_spks": [round(float(x), 3) for x in u["trial_rasters"][1]],
            "t3_spks": [round(float(x), 3) for x in u["trial_rasters"][2]],
            
            "t1_fr": [round(float(x), 1) for x in u["trials_fr"][0]],
            "t2_fr": [round(float(x), 1) for x in u["trials_fr"][1]],
            "t3_fr": [round(float(x), 1) for x in u["trials_fr"][2]],
            "mean_fr": [round(float(x), 1) for x in u["mean_fr"]],
            
            "pre_rates": [round(float(x), 1) for x in u["pre_rates"]],
            "laser_rates": [round(float(x), 1) for x in u["laser_rates"]],
            "post_rates": [round(float(x), 1) for x in u["post_rates"]],
            "post_trans_rates": [round(float(x), 1) for x in u["post_trans_rates"]],
            "post_resumed_rates": [round(float(x), 1) for x in u["post_resumed_rates"]],
        }

    first_u = list(unit_3trial_dict.values())[0]
    time_x = [round(float(x), 3) for x in first_u["bin_centers"]]
    
    # Subsampled flow and emg traces for lightweight HTML embedding (every 10th sample = 100 Hz)
    flow_sub = [round(float(x), 2) for x in flow_trials[2][::10]]
    emg_sub = [round(float(x), 3) for x in emg_trials[2][::10]]
    physio_x = [round(float(x), 3) for x in t_rel[::10]]

    json_payload = json.dumps({
        "time_x": time_x,
        "physio_x": physio_x,
        "flow_t3": flow_sub,
        "emg_t3": emg_sub,
        "units": units_data
    })

    t_resump = round(train_dur + 4.0, 3)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>e097 3-Trial Laser Train PETH Interactive Workbench (-10s Pre, Laser, +10s Post-Offset)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 16px 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #30363d;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        h1 {{
            font-size: 20px;
            font-weight: 700;
            color: #58a6ff;
            margin: 0 0 4px 0;
        }}
        .subtitle {{
            font-size: 12px;
            color: #8b949e;
            margin: 0;
        }}
        .controls {{
            display: flex;
            gap: 14px;
            align-items: center;
            background: #161b22;
            padding: 10px 16px;
            border-radius: 8px;
            border: 1px solid #30363d;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}
        select {{
            background: #21262d;
            color: #f0f6fc;
            border: 1px solid #388bfd;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }}
        .stat-badge {{
            background: #21262d;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #30363d;
            font-size: 12px;
        }}
        .stat-badge b {{
            color: #58a6ff;
        }}
        #plot-container {{
            background: #161b22;
            border-radius: 8px;
            border: 1px solid #30363d;
            padding: 8px;
            min-height: 860px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>e097 3-Trial Laser Train PETH Interactive Workbench (-10s Pre, Laser 2s, +10s Post-Offset)</h1>
            <p class="subtitle">Complete 22s Epochs: 10s Pre-Laser Eupnea | 2s Laser Apnea | ~4s Post-Apneic Transition & Sigh | 6s Resumed Normal Breathing</p>
        </div>
        <div>
            <span class="stat-badge">Animal: <b>e097</b> | Total Sorter Duration: <b>226.24s</b> | Epochs: <b>3 Trains</b></span>
        </div>
    </div>

    <div class="controls">
        <label for="unit-select"><b>Select Single Unit:</b></label>
        <select id="unit-select" onchange="updateUnit(this.value)">
            <optgroup label="⭐ Recruited Rostral Units (d ≤ 1080 μm)">
                <option value="315" selected>Cluster 315 (435 μm, BötC) — Surges from 0 Hz → 60 Hz in Train 3</option>
                <option value="13">Cluster 13 (15 μm, far rostral) — Surges to 12.3 Hz in Train 3</option>
                <option value="343">Cluster 343 (15 μm, far rostral) — Surges to 10.7 Hz in Train 3</option>
                <option value="366">Cluster 366 (30 μm, far rostral) — Surges to 10.2 Hz in Train 3</option>
                <option value="61">Cluster 61 (450 μm, BötC) — Recruited in Train 3</option>
                <option value="269">Cluster 269 (690 μm, rVRC) — Recruited across Trains</option>
            </optgroup>
            <optgroup label="⭐ Recruited Caudal Units (cVRG, d > 1800 μm)">
                <option value="316">Cluster 316 (2010 μm) — Surges to 80.8 Hz in Train 3</option>
                <option value="354">Cluster 354 (1905 μm) — Surges to 78.8 Hz in Train 3</option>
                <option value="10">Cluster 10 (2115 μm) — Surges to 58.8 Hz in Train 3</option>
                <option value="19">Cluster 19 (2505 μm) — Surges to 49.6 Hz in Train 3</option>
                <option value="387">Cluster 387 (2385 μm) — Surges to 46.0 Hz in Train 3</option>
                <option value="74">Cluster 74 (1875 μm) — Surges to 36.8 Hz in Train 3</option>
            </optgroup>
            <optgroup label="⭐ Tracked Baseline Archetypes & Surges">
                <option value="430">Cluster 430 (1410 μm) [Unit 2 Pacemaker] — Apnea Quenched (17 Hz → 1 Hz)</option>
                <option value="298">Cluster 298 (1114 μm) [Unit 255 Ramping] — Apnea Quenched (62 Hz → 16 Hz)</option>
                <option value="448">Cluster 448 (1380 μm) [Unit 163 Pre-I] — Pulse-Synchronized Surges</option>
                <option value="18">Cluster 18 (645 μm) [Unit 18 BötC Exp] — Suppressed during Apnea</option>
                <option value="93">Cluster 93 (1470 μm) [Unit 70 Vanguard] — High-Rate Burster</option>
                <option value="82">Cluster 82 (1913 μm) [Laser Surge] — Surges 0.5 Hz → 33 Hz across all 3 Trains</option>
            </optgroup>
            <optgroup label="All Single Units (Sorted by Depth)" id="all-units-group">
            </optgroup>
        </select>
        <span class="stat-badge" id="badge-depth">Depth: <b>435 μm</b></span>
        <span class="stat-badge" id="badge-t1">Train 1: Pre=<b>0.0</b>, Laser=<b>3.1</b>, Post=<b>0.0 Hz</b></span>
        <span class="stat-badge" id="badge-t2">Train 2: Pre=<b>0.0</b>, Laser=<b>3.1</b>, Post=<b>0.0 Hz</b></span>
        <span class="stat-badge" id="badge-t3">Train 3: Pre=<b>0.0</b>, Laser=<b>59.8</b>, Post=<b>67.0 Hz</b></span>
    </div>

    <div id="plot-container"></div>

    <script>
        const DATA = {json_payload};
        const allGroup = document.getElementById("all-units-group");
        
        const uList = Object.values(DATA.units).sort((a, b) => a.depth - b.depth);
        uList.forEach(u => {{
            const opt = document.createElement("option");
            opt.value = u.cid;
            opt.textContent = `Cluster ${{u.cid}} (${{u.depth}} μm, T3 Laser=${{u.laser_rates[2]}}Hz)`;
            allGroup.appendChild(opt);
        }});

        function updateUnit(uid) {{
            const u = DATA.units[uid];
            if (!u) return;

            document.getElementById("badge-depth").innerHTML = `Depth: <b>${{u.depth}} μm</b>`;
            document.getElementById("badge-t1").innerHTML = `Train 1: Pre=<b>${{u.pre_rates[0]}}</b>, Laser=<b>${{u.laser_rates[0]}}</b>, Post(10s)=<b>${{u.post_rates[0]}} Hz</b>`;
            document.getElementById("badge-t2").innerHTML = `Train 2: Pre=<b>${{u.pre_rates[1]}}</b>, Laser=<b>${{u.laser_rates[1]}}</b>, Post(10s)=<b>${{u.post_rates[1]}} Hz</b>`;
            document.getElementById("badge-t3").innerHTML = `Train 3: Pre=<b>${{u.pre_rates[2]}}</b>, Laser=<b>${{u.laser_rates[2]}}</b>, Post(10s)=<b>${{u.post_rates[2]}} Hz</b>`;

            const traces = [
                // 1. Spikes Raster T1 (row 1, col 1)
                {{
                    x: u.t1_spks, y: u.t1_spks.map(() => 1),
                    mode: 'markers', marker: {{ symbol: 'line-ns', size: 12, color: '#1f77b4', line: {{ width: 1.5 }} }},
                    name: 'Train 1', xaxis: 'x', yaxis: 'y',
                    hovertemplate: "Train 1<br>Time: %{{x:.3f}} s<extra></extra>"
                }},
                // Spikes Raster T2
                {{
                    x: u.t2_spks, y: u.t2_spks.map(() => 2),
                    mode: 'markers', marker: {{ symbol: 'line-ns', size: 12, color: '#2ca02c', line: {{ width: 1.5 }} }},
                    name: 'Train 2', xaxis: 'x', yaxis: 'y',
                    hovertemplate: "Train 2<br>Time: %{{x:.3f}} s<extra></extra>"
                }},
                // Spikes Raster T3
                {{
                    x: u.t3_spks, y: u.t3_spks.map(() => 3),
                    mode: 'markers', marker: {{ symbol: 'line-ns', size: 12, color: '#d62728', line: {{ width: 1.5 }} }},
                    name: 'Train 3', xaxis: 'x', yaxis: 'y',
                    hovertemplate: "Train 3<br>Time: %{{x:.3f}} s<extra></extra>"
                }},
                // Airflow overlay on raster
                {{
                    x: DATA.physio_x, y: DATA.flow_t3.map(v => (v + 3.0) / 6.0 * 2.5 + 0.5),
                    mode: 'lines', line: {{ color: '#8b949e', width: 1.0 }}, opacity: 0.35,
                    name: 'Airflow (T3)', xaxis: 'x', yaxis: 'y', hoverinfo: 'skip'
                }},

                // 2. Firing Rate Overlays (row 1, col 2)
                {{
                    x: DATA.time_x, y: u.t1_fr,
                    mode: 'lines', line: {{ color: '#1f77b4', width: 1.4 }},
                    name: 'Train 1 (Blue)', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 1: %{{y:.1f}} Hz<extra></extra>"
                }},
                {{
                    x: DATA.time_x, y: u.t2_fr,
                    mode: 'lines', line: {{ color: '#2ca02c', width: 1.4 }},
                    name: 'Train 2 (Green)', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 2: %{{y:.1f}} Hz<extra></extra>"
                }},
                {{
                    x: DATA.time_x, y: u.t3_fr,
                    mode: 'lines', line: {{ color: '#d62728', width: 2.0 }},
                    name: 'Train 3 (Red)', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 3: %{{y:.1f}} Hz<extra></extra>"
                }},
                {{
                    x: DATA.time_x, y: u.mean_fr,
                    mode: 'lines', line: {{ color: '#f0f6fc', width: 2.3 }},
                    name: 'Mean FR (White)', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Mean: %{{y:.1f}} Hz<extra></extra>"
                }},

                // 3. Phase Bar Chart (row 2, full width)
                {{
                    x: ['Train 1', 'Train 2', 'Train 3'],
                    y: u.pre_rates,
                    name: 'Pre-Laser (10s)',
                    type: 'bar', marker: {{ color: '#8b949e' }},
                    xaxis: 'x3', yaxis: 'y3'
                }},
                {{
                    x: ['Train 1', 'Train 2', 'Train 3'],
                    y: u.laser_rates,
                    name: 'Laser Apnea (2s)',
                    type: 'bar', marker: {{ color: '#ff9800' }},
                    xaxis: 'x3', yaxis: 'y3'
                }},
                {{
                    x: ['Train 1', 'Train 2', 'Train 3'],
                    y: u.post_trans_rates,
                    name: 'Transition / Sigh (4s)',
                    type: 'bar', marker: {{ color: '#fcc419' }},
                    xaxis: 'x3', yaxis: 'y3'
                }},
                {{
                    x: ['Train 1', 'Train 2', 'Train 3'],
                    y: u.post_resumed_rates,
                    name: 'Resumed Normal (6s)',
                    type: 'bar', marker: {{ color: '#20c997' }},
                    xaxis: 'x3', yaxis: 'y3'
                }}
            ];

            const layout = {{
                template: 'plotly_dark',
                paper_bgcolor: '#161b22',
                plot_bgcolor: '#0d1117',
                font: {{ family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color: '#c9d1d9' }},
                height: 840,
                showlegend: true,
                legend: {{ orientation: 'h', x: 0.1, y: 1.08 }},
                margin: {{ l: 60, r: 40, t: 70, b: 60 }},
                grid: {{ rows: 2, columns: 2, pattern: 'independent' }},
                annotations: [
                    {{ text: "<b>1. 3-Trial Spike Raster (-10s to +12s) with Airflow overlay</b>", xref: "paper", yref: "paper", x: 0.20, y: 1.02, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "<b>2. Continuous Firing Rate Overlays (50ms bins, Gaussian smooth)</b>", xref: "paper", yref: "paper", x: 0.80, y: 1.02, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "<b>3. Phase-by-Phase Rate Quantification (Pre 10s vs Laser 2s vs Trans 4s vs Resumed 6s)</b>", xref: "paper", yref: "paper", x: 0.50, y: 0.44, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }}
                ],
                shapes: [
                    // Subplot 1: 0-1.955s laser shading
                    {{ type: 'rect', xref: 'x', yref: 'y', x0: 0, x1: 1.955, y0: 0.4, y1: 3.6, fillcolor: '#ff9800', opacity: 0.22, line: {{ width: 0 }} }},
                    // Subplot 1: 1.955-5.955 transition shading
                    {{ type: 'rect', xref: 'x', yref: 'y', x0: 1.955, x1: {t_resump}, y0: 0.4, y1: 3.6, fillcolor: '#fcc419', opacity: 0.15, line: {{ width: 0 }} }},
                    // Subplot 1: Resumption line
                    {{ type: 'line', xref: 'x', yref: 'y', x0: {t_resump}, x1: {t_resump}, y0: 0.4, y1: 3.6, line: {{ color: '#20c997', width: 1.5, dash: 'dash' }} }},
                    
                    // Subplot 2: 0-1.955s laser shading
                    {{ type: 'rect', xref: 'x2', yref: 'paper', x0: 0, x1: 1.955, y0: 0.54, y1: 1.0, fillcolor: '#ff9800', opacity: 0.22, line: {{ width: 0 }} }},
                    // Subplot 2: 1.955-5.955 transition shading
                    {{ type: 'rect', xref: 'x2', yref: 'paper', x0: 1.955, x1: {t_resump}, y0: 0.54, y1: 1.0, fillcolor: '#fcc419', opacity: 0.15, line: {{ width: 0 }} }},
                    // Subplot 2: Resumption line
                    {{ type: 'line', xref: 'x2', yref: 'paper', x0: {t_resump}, x1: {t_resump}, y0: 0.54, y1: 1.0, line: {{ color: '#20c997', width: 1.5, dash: 'dash' }} }}
                ],
                xaxis: {{ title: "Time from Train Onset (s)", range: [{t_pre}, {t_post}], gridcolor: "#21262d" }},
                yaxis: {{ title: "Trial #", range: [0.4, 3.6], tickvals: [1, 2, 3], ticktext: ['Train 1', 'Train 2', 'Train 3'], gridcolor: "#21262d" }},
                xaxis2: {{ title: "Time from Train Onset (s)", range: [{t_pre}, {t_post}], gridcolor: "#21262d" }},
                yaxis2: {{ title: "Firing Rate (Hz)", gridcolor: "#21262d" }},
                xaxis3: {{ domain: [0.0, 1.0], title: "Photostimulation Train Trial", gridcolor: "#21262d" }},
                yaxis3: {{ domain: [0.0, 0.38], title: "Average Firing Rate (Hz)", gridcolor: "#21262d" }}
            }};

            Plotly.newPlot('plot-container', traces, layout, {{ responsive: true }});
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            updateUnit("315");
        }});
    </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Exported upgraded 3-trial interactive workbench to: {out_html}")

if __name__ == "__main__":
    main()
