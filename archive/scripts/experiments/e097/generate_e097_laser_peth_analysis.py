#!/usr/bin/env python3
"""
Generate Comprehensive Peri-Event Time Histograms (PETH / PSTH) for e097 Units Across:
1. Laser Pulse-Triggered PETH (-15 ms to +45 ms, 1 ms bins, 120 pulses @ 20 Hz, 5 ms pulse width)
2. Laser Train-Triggered PETH (-1.5 s to +4.5 s, 50 ms bins, 3 trains of 2 s each)
3. Inspiratory Breath-Triggered PETH (-150 ms to +350 ms, 5 ms bins, 199 eupnea breaths)

Covers:
- Recruited Rostral Units (Cluster 315, 13, 343, 366, 61, 269)
- Recruited Caudal Units (Cluster 316, 354, 10, 19, 387, 74)
- Core Baseline Archetypes (Cluster 430 / Unit 2, Cluster 298 / Unit 255, Cluster 448 / Unit 163, Cluster 18 / Unit 18, Cluster 93 / Unit 70)
- Laser-Surging Units (Cluster 82, Cluster 209, Cluster 320)
- Laser-Suppressed Units (Cluster 103, 220, 450, 437, 447)
- All 161 QC-passing single units in the consensus master dataset + Cluster 448.
"""

import os
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
    print("[*] Starting comprehensive PETH analysis for e097...")
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

    print(f"Total units to analyze: {len(target_cids)} (161 QC SUA + Cluster 448).")

    # 2. Load NIDQ Analog Data for Laser, Flow, and EMG
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    
    # Laser channel (Ch 6)
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(laser_v)) / fs

    # EMG channel (Ch 0)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    
    # Airflow channel (Ch 5)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)

    # 3. Detect Laser Pulses and Trains
    above = laser_v > 0.5
    diff = np.diff(above.astype(np.int32))
    pulse_starts = t_vec[np.where(diff == 1)[0] + 1]

    # Train definitions
    train_break_indices = np.where(np.diff(pulse_starts) > 1.0)[0]
    train_start_indices = [0] + [b + 1 for b in train_break_indices]
    train_starts = [pulse_starts[idx] for idx in train_start_indices]
    train_dur = 1.955

    print(f"Detected {len(pulse_starts)} laser pulses across {len(train_starts)} trains:")
    for i, ts in enumerate(train_starts, 1):
        t_pulses = pulse_starts[(pulse_starts >= ts - 0.01) & (pulse_starts < ts + 3.0)]
        print(f"  Train {i}: Start={ts:.4f}s, Pulses={len(t_pulses)}, Duration={t_pulses[-1] - ts + 0.005:.3f}s")

    # 4. Detect Breaths via EMG
    print("[*] Detecting breath onsets from Diaphragm EMG...")
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    all_breath_onsets = emg_proc.breath_onsets
    eupnea_breath_onsets = all_breath_onsets[(all_breath_onsets >= 1.0) & (all_breath_onsets <= 59.0)]
    print(f"Detected {len(eupnea_breath_onsets)} eupneic breath onsets in baseline window (t=1..59s).")

    # =========================================================================
    # 5. PETH COMPUTATIONS
    # =========================================================================
    pulse_pre_s = 0.015
    pulse_post_s = 0.045
    pulse_dt = 0.001
    pulse_bins = np.arange(-pulse_pre_s, pulse_post_s + pulse_dt, pulse_dt)
    pulse_centers_ms = (pulse_bins[:-1] + pulse_bins[1:]) / 2.0 * 1000.0
    n_pulse_bins = len(pulse_centers_ms)

    train_pre_s = 1.5
    train_post_s = 4.5
    train_dt = 0.050
    train_bins = np.arange(-train_pre_s, train_post_s + train_dt, train_dt)
    train_centers_s = (train_bins[:-1] + train_bins[1:]) / 2.0
    n_train_bins = len(train_centers_s)

    breath_pre_s = 0.150
    breath_post_s = 0.350
    breath_dt = 0.005
    breath_bins = np.arange(-breath_pre_s, breath_post_s + breath_dt, breath_dt)
    breath_centers_ms = (breath_bins[:-1] + breath_bins[1:]) / 2.0 * 1000.0
    n_breath_bins = len(breath_centers_ms)

    pulse_results = []
    unit_peth_dict = {}

    print(f"[*] Computing PETH matrices for {len(target_cids)} single units...")

    for cid in target_cids:
        # Check metadata
        row_match = pr_df[pr_df["cluster_id"] == cid]
        if len(row_match) > 0:
            row = row_match.iloc[0]
            depth = float(row["depth_um"])
            is_rostral = bool(row.get("is_rostral_vrc", depth <= 1080.0))
            is_recruited = bool(row.get("is_newly_recruited", False))
            fr_pre = float(row.get("fr_pre_laser", 0.0))
            max_laser_fr = float(row.get("max_laser_fr", 0.0))
            pr_eupnea = float(row.get("pr_eupnea_100", 1.0))
            pr_laser = float(row.get("pr_all_120_pulses", 0.5))
        else:
            depth = 1380.0
            is_rostral = False
            is_recruited = False
            fr_pre = 8.8
            max_laser_fr = 29.2
            pr_eupnea = 1.0
            pr_laser = 0.5

        u_spks = spk_times[spk_clusters == cid]

        # --- A. Pulse-level matrix (120 trials x n_pulse_bins) ---
        pulse_trials = np.zeros((len(pulse_starts), n_pulse_bins), dtype=np.float32)
        pulse_raster_times = []
        pulse_raster_trials = []

        for p_idx, p_t in enumerate(pulse_starts):
            rel = u_spks[(u_spks >= p_t - pulse_pre_s) & (u_spks <= p_t + pulse_post_s)] - p_t
            counts, _ = np.histogram(rel, bins=pulse_bins)
            pulse_trials[p_idx, :] = counts / pulse_dt
            if len(rel) > 0:
                pulse_raster_times.extend(rel * 1000.0) # in ms
                pulse_raster_trials.extend([p_idx] * len(rel))

        t1_trials = pulse_trials[0:40, :]
        t2_trials = pulse_trials[40:80, :]
        t3_trials = pulse_trials[80:120, :]

        mean_pulse_fr = np.mean(pulse_trials, axis=0)
        sem_pulse_fr = np.std(pulse_trials, axis=0) / np.sqrt(len(pulse_starts))

        baseline_mask = pulse_centers_ms < 0.0
        in_pulse_mask = (pulse_centers_ms >= 0.0) & (pulse_centers_ms <= 5.0)
        post_pulse_mask = (pulse_centers_ms > 5.0) & (pulse_centers_ms <= 40.0)

        base_rate = float(np.mean(mean_pulse_fr[baseline_mask]))
        in_pulse_rate = float(np.mean(mean_pulse_fr[in_pulse_mask]))
        post_pulse_rate = float(np.mean(mean_pulse_fr[post_pulse_mask]))
        
        peak_idx = int(np.argmax(mean_pulse_fr))
        peak_latency_ms = float(pulse_centers_ms[peak_idx])
        peak_rate_hz = float(mean_pulse_fr[peak_idx])
        mod_idx = float((peak_rate_hz - base_rate) / (peak_rate_hz + base_rate + 1e-5))

        t3_mean = np.mean(t3_trials, axis=0)
        t3_peak_idx = int(np.argmax(t3_mean))
        t3_peak_lat_ms = float(pulse_centers_ms[t3_peak_idx])
        t3_peak_hz = float(t3_mean[t3_peak_idx])

        pulse_results.append({
            "cluster_id": cid,
            "depth_um": depth,
            "baseline_fr_hz": base_rate,
            "in_pulse_fr_hz": in_pulse_rate,
            "post_pulse_fr_hz": post_pulse_rate,
            "peak_latency_ms": peak_latency_ms,
            "peak_rate_hz": peak_rate_hz,
            "modulation_index": mod_idx,
            "t3_peak_latency_ms": t3_peak_lat_ms,
            "t3_peak_rate_hz": t3_peak_hz,
            "is_rostral": is_rostral,
            "is_recruited": is_recruited,
            "fr_pre": fr_pre,
            "max_laser_fr": max_laser_fr,
            "pr_eupnea": pr_eupnea,
            "pr_laser": pr_laser,
        })

        # --- B. Train-level matrix (3 trials x n_train_bins) ---
        train_trials = np.zeros((len(train_starts), n_train_bins), dtype=np.float32)
        train_raster_times = []
        train_raster_trials = []

        for tr_idx, tr_t in enumerate(train_starts):
            rel = u_spks[(u_spks >= tr_t - train_pre_s) & (u_spks <= tr_t + train_post_s)] - tr_t
            counts, _ = np.histogram(rel, bins=train_bins)
            train_trials[tr_idx, :] = counts / train_dt
            if len(rel) > 0:
                train_raster_times.extend(rel)
                train_raster_trials.extend([tr_idx] * len(rel))

        mean_train_fr = np.mean(train_trials, axis=0)
        sem_train_fr = np.std(train_trials, axis=0) / np.sqrt(len(train_starts))

        # --- C. Breath-level matrix (n_breaths x n_breath_bins) ---
        breath_trials = np.zeros((len(eupnea_breath_onsets), n_breath_bins), dtype=np.float32)
        breath_raster_times = []
        breath_raster_trials = []

        for b_idx, b_t in enumerate(eupnea_breath_onsets):
            rel = u_spks[(u_spks >= b_t - breath_pre_s) & (u_spks <= b_t + breath_post_s)] - b_t
            counts, _ = np.histogram(rel, bins=breath_bins)
            breath_trials[b_idx, :] = counts / breath_dt
            if len(rel) > 0:
                breath_raster_times.extend(rel * 1000.0)
                breath_raster_trials.extend([b_idx] * len(rel))

        g_kernel = signal.windows.gaussian(7, std=1.5)
        g_kernel /= np.sum(g_kernel)
        mean_breath_raw = np.mean(breath_trials, axis=0)
        mean_breath_fr = np.convolve(mean_breath_raw, g_kernel, mode="same")
        std_breath_fr = np.std(breath_trials, axis=0)

        unit_peth_dict[cid] = {
            "depth_um": depth,
            "pulse_centers_ms": pulse_centers_ms,
            "mean_pulse_fr": mean_pulse_fr,
            "sem_pulse_fr": sem_pulse_fr,
            "t1_mean_pulse": np.mean(t1_trials, axis=0),
            "t2_mean_pulse": np.mean(t2_trials, axis=0),
            "t3_mean_pulse": t3_mean,
            "pulse_raster_times": np.array(pulse_raster_times),
            "pulse_raster_trials": np.array(pulse_raster_trials),
            
            "train_centers_s": train_centers_s,
            "mean_train_fr": mean_train_fr,
            "sem_train_fr": sem_train_fr,
            "train_trials": train_trials,
            "train_raster_times": np.array(train_raster_times),
            "train_raster_trials": np.array(train_raster_trials),
            
            "breath_centers_ms": breath_centers_ms,
            "mean_breath_fr": mean_breath_fr,
            "std_breath_fr": std_breath_fr,
            "breath_raster_times": np.array(breath_raster_times),
            "breath_raster_trials": np.array(breath_raster_trials),
            
            "peak_latency_ms": peak_latency_ms,
            "peak_rate_hz": peak_rate_hz,
            "mod_idx": mod_idx,
            "base_rate": base_rate,
            "in_pulse_rate": in_pulse_rate,
            "post_pulse_rate": post_pulse_rate,
            "is_rostral": is_rostral,
            "is_recruited": is_recruited,
            "fr_pre": fr_pre,
            "max_laser_fr": max_laser_fr,
            "pr_eupnea": pr_eupnea,
            "pr_laser": pr_laser,
        }

    # Save summary CSV
    df_pulse_res = pd.DataFrame(pulse_results).sort_values(by="depth_um").reset_index(drop=True)
    csv_out = out_dir / "peth_laser_pulse_summary.csv"
    df_pulse_res.to_csv(csv_out, index=False)
    print(f"[+] Exported pulse PETH summary CSV to: {csv_out}")

    # =========================================================================
    # 6. GENERATE PUBLICATION-QUALITY STATIC FIGURE PANELS
    # =========================================================================
    print("[*] Rendering publication-quality multi-panel PETH figures...")
    
    # Figure 1: Recruited Rostral Units (Cluster 315, 13, 343, 366, 61, 269)
    render_group_figure(
        unit_ids=[315, 13, 343, 366, 61, 269],
        unit_peth_dict=unit_peth_dict,
        title="RECRUITED ROSTRAL UNITS: Laser-Pulse, Laser-Train & Breath PETHs",
        out_png=out_dir / "peth_recruited_rostral_units.png",
        out_svg=out_dir / "peth_recruited_rostral_units.svg"
    )

    # Figure 2: Recruited Caudal Units (Cluster 316, 354, 10, 19, 387, 74)
    render_group_figure(
        unit_ids=[316, 354, 10, 19, 387, 74],
        unit_peth_dict=unit_peth_dict,
        title="RECRUITED CAUDAL UNITS (cVRG): Laser-Pulse, Laser-Train & Breath PETHs",
        out_png=out_dir / "peth_recruited_caudal_units.png",
        out_svg=out_dir / "peth_recruited_caudal_units.svg"
    )

    # Figure 3: Tracked Baseline Archetypes & Laser-Surge Units
    render_group_figure(
        unit_ids=[430, 298, 448, 18, 93, 82],
        unit_peth_dict=unit_peth_dict,
        title="BASELINE ARCHETYPES & LASER SURGE UNITS: Laser-Pulse, Laser-Train & Breath PETHs",
        out_png=out_dir / "peth_baseline_archetypes_laser_response.png",
        out_svg=out_dir / "peth_baseline_archetypes_laser_response.svg"
    )

    # Figure 4: Laser-Suppressed Units (Apnea Quenching)
    render_group_figure(
        unit_ids=[103, 220, 450, 437, 447, 333],
        unit_peth_dict=unit_peth_dict,
        title="LASER-SUPPRESSED UNITS (Central Apnea Quenching): Laser-Pulse, Laser-Train & Breath PETHs",
        out_png=out_dir / "peth_laser_suppressed_apnea_units.png",
        out_svg=out_dir / "peth_laser_suppressed_apnea_units.svg"
    )

    # Figure 5: Population Latency & Modulation vs Probe Depth
    render_population_depth_tuning(
        df_pulse_res=df_pulse_res,
        out_png=out_dir / "population_laser_modulation_vs_depth.png",
        out_svg=out_dir / "population_laser_modulation_vs_depth.svg"
    )

    # =========================================================================
    # 7. GENERATE INTERACTIVE HTML WORKBENCH WITH SELECTION
    # =========================================================================
    print("[*] Generating interactive multi-unit PETH dashboard HTML...")
    generate_standalone_interactive_peth_html(
        unit_peth_dict=unit_peth_dict,
        df_summary=df_pulse_res,
        out_html=out_dir / "e097_laser_and_breath_peth_interactive_workbench.html"
    )

    print("[DONE] PETH analysis successfully generated and exported!")

def render_group_figure(unit_ids, unit_peth_dict, title, out_png, out_svg):
    n_units = len(unit_ids)
    fig, axes = plt.subplots(n_units, 4, figsize=(18, 2.7 * n_units), sharex='col')
    fig.patch.set_facecolor('white')
    if n_units == 1:
        axes = np.array([axes])

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    train_colors = ['#1f77b4', '#2ca02c', '#d62728']

    for row_idx, uid in enumerate(unit_ids):
        u_data = unit_peth_dict[uid]
        depth = u_data["depth_um"]
        
        name_str = f"Cluster {uid} ({depth:.0f} μm)"
        if uid == 430: name_str += " [Unit 2 Pacemaker]"
        elif uid == 298: name_str += " [Unit 255 Ramping]"
        elif uid == 448: name_str += " [Unit 163 Pre-I]"
        elif uid == 18: name_str += " [Unit 18 BötC Exp]"
        elif uid == 93: name_str += " [Unit 70 Vanguard]"
        elif uid == 315: name_str += " [Recruited BötC]"
        elif uid == 82: name_str += " [Laser Surge]"

        # --- Col 1: Laser Pulse Raster ---
        ax1 = axes[row_idx, 0]
        ax1.set_facecolor('#fafafa')
        ax1.axvspan(0, 5, color='#ff9800', alpha=0.3, label='Laser (5ms)')
        
        r_times = u_data["pulse_raster_times"]
        r_trials = u_data["pulse_raster_trials"]
        if len(r_times) > 0:
            for tr_num, (tr_min, tr_max) in enumerate([(0, 39), (40, 79), (80, 119)]):
                mask_tr = (r_trials >= tr_min) & (r_trials <= tr_max)
                if np.any(mask_tr):
                    ax1.scatter(r_times[mask_tr], r_trials[mask_tr] + 1, s=4, c=train_colors[tr_num], alpha=0.8, edgecolors='none')
        
        ax1.axhline(40.5, color='gray', linestyle=':', linewidth=0.8)
        ax1.axhline(80.5, color='gray', linestyle=':', linewidth=0.8)
        ax1.set_xlim(-15, 45)
        ax1.set_ylim(0, 121)
        ax1.set_ylabel(f"{name_str}\nPulse #", fontsize=8, fontweight='bold')
        if row_idx == 0:
            ax1.set_title("Laser Pulse Raster (120 Pulses)\nBlue:T1, Green:T2, Red:T3", fontsize=9, fontweight='bold')
        if row_idx == n_units - 1:
            ax1.set_xlabel("Time from Pulse Onset (ms)", fontsize=9, fontweight='bold')

        # --- Col 2: Laser Pulse PETH ---
        ax2 = axes[row_idx, 1]
        ax2.set_facecolor('#fafafa')
        ax2.axvspan(0, 5, color='#ff9800', alpha=0.3)
        ax2.plot(u_data["pulse_centers_ms"], u_data["mean_pulse_fr"], color='#006699', linewidth=1.5, label='All 120')
        ax2.fill_between(
            u_data["pulse_centers_ms"],
            np.maximum(0, u_data["mean_pulse_fr"] - u_data["sem_pulse_fr"]),
            u_data["mean_pulse_fr"] + u_data["sem_pulse_fr"],
            color='#006699', alpha=0.25
        )
        ax2.plot(u_data["pulse_centers_ms"], u_data["t3_mean_pulse"], color='#d62728', linewidth=1.0, linestyle='--', label='Train 3')
        
        ax2.set_xlim(-15, 45)
        max_y2 = max(10, np.max(u_data["mean_pulse_fr"]) * 1.25, np.max(u_data["t3_mean_pulse"]) * 1.1)
        ax2.set_ylim(0, max_y2)
        ax2.set_ylabel("Firing Rate (Hz)", fontsize=8)
        if row_idx == 0:
            ax2.set_title("Pulse PETH (Mean ± SEM)\nBlue: All 120 | Red: Train 3", fontsize=9, fontweight='bold')
        if row_idx == n_units - 1:
            ax2.set_xlabel("Time from Pulse Onset (ms)", fontsize=9, fontweight='bold')
        ax2.text(0.96, 0.90, f"Peak: {u_data['peak_latency_ms']:.1f}ms ({u_data['peak_rate_hz']:.0f}Hz)",
                 transform=ax2.transAxes, ha='right', va='top', fontsize=7.5,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#ccc'))

        # --- Col 3: Laser Train PETH ---
        ax3 = axes[row_idx, 2]
        ax3.set_facecolor('#fafafa')
        ax3.axvspan(0.0, 1.955, color='#ff9800', alpha=0.3, label='Laser Train (2s)')
        
        for tr_i in range(3):
            ax3.plot(u_data["train_centers_s"], u_data["train_trials"][tr_i], color=train_colors[tr_i], linewidth=0.9, alpha=0.7)
        ax3.plot(u_data["train_centers_s"], u_data["mean_train_fr"], color='black', linewidth=1.8, label='Mean')
        ax3.fill_between(
            u_data["train_centers_s"],
            np.maximum(0, u_data["mean_train_fr"] - u_data["sem_train_fr"]),
            u_data["mean_train_fr"] + u_data["sem_train_fr"],
            color='black', alpha=0.15
        )
        ax3.set_xlim(-1.5, 4.5)
        max_y3 = max(5, np.max(u_data["train_trials"]) * 1.15)
        ax3.set_ylim(0, max_y3)
        ax3.set_ylabel("Firing Rate (Hz)", fontsize=8)
        if row_idx == 0:
            ax3.set_title("Laser Train PETH (-1.5s to +4.5s)\nShaded: 2s Photostimulation", fontsize=9, fontweight='bold')
        if row_idx == n_units - 1:
            ax3.set_xlabel("Time from Train Onset (s)", fontsize=9, fontweight='bold')

        # --- Col 4: Inspiratory Breath PETH ---
        ax4 = axes[row_idx, 3]
        ax4.set_facecolor('#fafafa')
        ax4.axvline(0, color='teal', linestyle='--', linewidth=1.2, label='Insp Onset')
        ax4.plot(u_data["breath_centers_ms"], u_data["mean_breath_fr"], color='teal', linewidth=1.6)
        ax4.fill_between(
            u_data["breath_centers_ms"],
            np.maximum(0, u_data["mean_breath_fr"] - u_data["std_breath_fr"] * 0.5),
            u_data["mean_breath_fr"] + u_data["std_breath_fr"] * 0.5,
            color='teal', alpha=0.2
        )
        ax4.set_xlim(-150, 350)
        max_y4 = max(5, np.max(u_data["mean_breath_fr"]) * 1.25)
        ax4.set_ylim(0, max_y4)
        ax4.set_ylabel("Firing Rate (Hz)", fontsize=8)
        if row_idx == 0:
            ax4.set_title("Eupnea Breath PETH (192 Breaths)\nAligned to Insp Onset (t=0)", fontsize=9, fontweight='bold')
        if row_idx == n_units - 1:
            ax4.set_xlabel("Time from Insp Onset (ms)", fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved figure to {out_png.name} and {out_svg.name}")

def render_population_depth_tuning(df_pulse_res, out_png, out_svg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    fig.patch.set_facecolor('white')

    colors = []
    for _, row in df_pulse_res.iterrows():
        if row["is_recruited"] and row["is_rostral"]:
            colors.append('#d62728')
        elif row["is_recruited"]:
            colors.append('#9467bd')
        else:
            colors.append('#1f77b4')

    ax1.set_facecolor('#fafafa')
    ax1.axvspan(0, 5, color='#ff9800', alpha=0.25, label='Laser Pulse (0-5ms)')
    ax1.scatter(df_pulse_res["peak_latency_ms"], df_pulse_res["depth_um"], c=colors, s=35, alpha=0.85, edgecolors='black', linewidth=0.5)
    ax1.set_xlim(-15, 45)
    ax1.set_ylim(3000, -100)
    ax1.set_xlabel("Peak Pulse Latency (ms)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Probe Depth (μm)", fontsize=10, fontweight='bold')
    ax1.set_title("Peak Response Latency vs Depth\n(Red: Rostral Recruited, Blue: Baseline)", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axvline(10.5, color='crimson', linestyle='--', linewidth=1.0, label='Rebound (10.5ms)')
    ax1.legend(loc='lower left', fontsize=8)

    ax2.set_facecolor('#fafafa')
    ax2.axvline(0, color='gray', linestyle='--', linewidth=1.0)
    ax2.scatter(df_pulse_res["modulation_index"], df_pulse_res["depth_um"], c=colors, s=35, alpha=0.85, edgecolors='black', linewidth=0.5)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_xlabel("Laser Pulse Modulation Index", fontsize=10, fontweight='bold')
    ax2.set_title("Laser Pulse Modulation Index\n[(Peak - Base) / (Peak + Base)]", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3.set_facecolor('#fafafa')
    ax3.scatter(df_pulse_res["peak_rate_hz"], df_pulse_res["depth_um"], c=colors, s=35, alpha=0.85, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel("Peak Evoked Firing Rate (Hz)", fontsize=10, fontweight='bold')
    ax3.set_title("Peak Evoked Rate vs Depth\n(Note surge in BötC & cVRG)", fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)

    ax1.axhline(600, color='gray', linestyle=':', linewidth=0.8)
    ax1.text(-14, 580, "BötC / rVRC", color='gray', fontsize=8, fontweight='bold')
    ax1.axhline(1400, color='gray', linestyle=':', linewidth=0.8)
    ax1.text(-14, 1380, "preBötC / Pacemakers", color='gray', fontsize=8, fontweight='bold')
    ax1.axhline(2200, color='gray', linestyle=':', linewidth=0.8)
    ax1.text(-14, 2180, "cVRG / Expiratory Bulbospinal", color='gray', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved population depth tuning figure to {out_png.name}")

def generate_standalone_interactive_peth_html(unit_peth_dict, df_summary, out_html):
    units_data = {}
    for uid, u in unit_peth_dict.items():
        r_t = u["pulse_raster_times"]
        r_p = u["pulse_raster_trials"]
        if len(r_t) > 600:
            sub_idx = np.random.choice(len(r_t), 600, replace=False)
            r_t = r_t[sub_idx]
            r_p = r_p[sub_idx]

        units_data[str(uid)] = {
            "cid": int(uid),
            "depth": round(u["depth_um"], 1),
            "base_fr": round(u["base_rate"], 2),
            "laser_fr": round(u["max_laser_fr"], 2),
            "peak_lat": round(u["peak_latency_ms"], 1),
            "peak_hz": round(u["peak_rate_hz"], 1),
            "mod_idx": round(u["mod_idx"], 3),
            "is_rostral": bool(u["is_rostral"]),
            "is_recruited": bool(u["is_recruited"]),
            "pr_eupnea": round(u["pr_eupnea"], 2),
            "pr_laser": round(u["pr_laser"], 2),
            "r_t": [round(float(x), 2) for x in r_t],
            "r_p": [int(y) for y in r_p],
            "pulse_fr": [round(float(x), 1) for x in u["mean_pulse_fr"]],
            "t3_pulse_fr": [round(float(x), 1) for x in u["t3_mean_pulse"]],
            "train_fr": [round(float(x), 1) for x in u["mean_train_fr"]],
            "t1_tr": [round(float(x), 1) for x in u["train_trials"][0]],
            "t2_tr": [round(float(x), 1) for x in u["train_trials"][1]],
            "t3_tr": [round(float(x), 1) for x in u["train_trials"][2]],
            "breath_fr": [round(float(x), 1) for x in u["mean_breath_fr"]],
        }

    first_u = list(unit_peth_dict.values())[0]
    pulse_x = [round(float(x), 2) for x in first_u["pulse_centers_ms"]]
    train_x = [round(float(x), 3) for x in first_u["train_centers_s"]]
    breath_x = [round(float(x), 2) for x in first_u["breath_centers_ms"]]

    json_payload = json.dumps({
        "pulse_x": pulse_x,
        "train_x": train_x,
        "breath_x": breath_x,
        "units": units_data
    })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>e097 Laser & Breath Peri-Event Time Histograms (PETH) Workbench</title>
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
            gap: 16px;
            align-items: center;
            background: #161b22;
            padding: 10px 16px;
            border-radius: 8px;
            border: 1px solid #30363d;
            margin-bottom: 16px;
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
            min-height: 820px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>e097 Laser & Breath Peri-Event Time Histograms (PETH) Interactive Workbench</h1>
            <p class="subtitle">Single-Cell Photostimulation Latency, Laser Apnea Response, and Respiratory Phase Alignment across 226.24s Continuous Sorting</p>
        </div>
        <div>
            <span class="stat-badge">Animal: <b>e097</b> | Sorter: <b>Lupin 226s Consensus Master</b> | Units: <b>162 Analyzed</b></span>
        </div>
    </div>

    <div class="controls">
        <label for="unit-select"><b>Select Single Unit:</b></label>
        <select id="unit-select" onchange="updateUnit(this.value)">
            <optgroup label="⭐ Recruited Rostral Units (d ≤ 1080 μm)">
                <option value="315" selected>Cluster 315 (435 μm, BötC) — Surges 0 Hz → 60 Hz in Train 3</option>
                <option value="13">Cluster 13 (15 μm, far rostral) — Recruited in Train 3</option>
                <option value="343">Cluster 343 (15 μm, far rostral) — Recruited in Train 3</option>
                <option value="366">Cluster 366 (30 μm, far rostral) — Recruited in Train 3</option>
                <option value="61">Cluster 61 (450 μm, BötC) — Recruited in Train 3</option>
                <option value="269">Cluster 269 (690 μm, rVRC) — Recruited in Train 3</option>
            </optgroup>
            <optgroup label="⭐ Recruited Caudal Units (cVRG, d > 1800 μm)">
                <option value="316">Cluster 316 (2010 μm) — Surges to 82 Hz</option>
                <option value="354">Cluster 354 (1905 μm) — Surges to 80 Hz</option>
                <option value="10">Cluster 10 (2115 μm) — Surges to 59 Hz</option>
                <option value="19">Cluster 19 (2505 μm) — Surges to 50 Hz</option>
                <option value="387">Cluster 387 (2385 μm) — Surges to 46 Hz</option>
                <option value="74">Cluster 74 (1875 μm) — Surges to 37 Hz</option>
            </optgroup>
            <optgroup label="⭐ Tracked Baseline Archetypes & Surges">
                <option value="430">Cluster 430 (1410 μm) [Unit 2 Pacemaker] — Suppressed 16 Hz → 3 Hz</option>
                <option value="298">Cluster 298 (1114 μm) [Unit 255 Ramping] — Suppressed 61 Hz → 17 Hz</option>
                <option value="448">Cluster 448 (1380 μm) [Unit 163 Pre-I] — 10.5ms Rebound Peak</option>
                <option value="18">Cluster 18 (645 μm) [Unit 18 BötC Exp] — Suppressed 84 Hz → 30 Hz</option>
                <option value="93">Cluster 93 (1470 μm) [Unit 70 Vanguard] — 9.5ms Rebound Peak</option>
                <option value="82">Cluster 82 (1913 μm) [Laser Surge] — 1.9 Hz → 33 Hz Surge</option>
            </optgroup>
            <optgroup label="⭐ Laser-Suppressed Units (Apnea Quenched)">
                <option value="103">Cluster 103 (555 μm) — 8.8 Hz → 0 Hz (Complete Quench)</option>
                <option value="220">Cluster 220 (555 μm) — 30.2 Hz → 0 Hz (Complete Quench)</option>
                <option value="450">Cluster 450 (540 μm) — 18.3 Hz → 0 Hz (Complete Quench)</option>
                <option value="437">Cluster 437 (1290 μm) — 30.0 Hz → 0.5 Hz</option>
                <option value="447">Cluster 447 (1125 μm) — 20.8 Hz → 1.0 Hz</option>
            </optgroup>
            <optgroup label="All Single Units (Sorted by Depth)" id="all-units-group">
            </optgroup>
        </select>
        <span class="stat-badge" id="badge-depth">Depth: <b>435 μm</b></span>
        <span class="stat-badge" id="badge-fr">Baseline: <b>0.0 Hz</b> | Laser: <b>59.8 Hz</b></span>
        <span class="stat-badge" id="badge-lat">Peak Latency: <b>8.5 ms</b> (Peak: <b>83 Hz</b>)</span>
        <span class="stat-badge" id="badge-pr">PR Eupnea: <b>0.00</b> | PR Laser: <b>0.85</b></span>
    </div>

    <div id="plot-container"></div>

    <script>
        const DATA = {json_payload};
        const allGroup = document.getElementById("all-units-group");
        
        // Populate all-units group
        const uList = Object.values(DATA.units).sort((a, b) => a.depth - b.depth);
        uList.forEach(u => {{
            const opt = document.createElement("option");
            opt.value = u.cid;
            opt.textContent = `Cluster ${{u.cid}} (${{u.depth}} μm, Base=${{u.base_fr}}Hz, Laser=${{u.laser_fr}}Hz)`;
            allGroup.appendChild(opt);
        }});

        function updateUnit(uid) {{
            const u = DATA.units[uid];
            if (!u) return;

            document.getElementById("badge-depth").innerHTML = `Depth: <b>${{u.depth}} μm</b>`;
            document.getElementById("badge-fr").innerHTML = `Baseline: <b>${{u.base_fr}} Hz</b> | Laser: <b>${{u.laser_fr}} Hz</b>`;
            document.getElementById("badge-lat").innerHTML = `Peak Latency: <b>${{u.peak_lat}} ms</b> (Peak: <b>${{u.peak_hz}} Hz</b>)`;
            document.getElementById("badge-pr").innerHTML = `PR Eupnea: <b>${{u.pr_eupnea}}</b> | PR Laser: <b>${{u.pr_laser}}</b>`;

            const cMap = u.r_p.map(p => p < 40 ? '#1f77b4' : (p < 80 ? '#2ca02c' : '#d62728'));

            const traces = [
                // 1. Raster (row 1, col 1)
                {{
                    x: u.r_t, y: u.r_p.map(p => p + 1),
                    mode: 'markers',
                    marker: {{ size: 4, color: cMap, opacity: 0.85 }},
                    name: 'Pulse Spikes',
                    xaxis: 'x', yaxis: 'y',
                    hovertemplate: "Time: %{{x:.1f}} ms<br>Pulse #: %{{y}}<extra></extra>"
                }},
                // 2. Pulse PETH Mean (row 1, col 2)
                {{
                    x: DATA.pulse_x, y: u.pulse_fr,
                    mode: 'lines', line: {{ color: '#006699', width: 2.5 }},
                    name: 'All 120 Pulses',
                    xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Time: %{{x:.1f}} ms<br>FR: %{{y:.1f}} Hz<extra></extra>"
                }},
                // Pulse PETH Train 3 (row 1, col 2)
                {{
                    x: DATA.pulse_x, y: u.t3_pulse_fr,
                    mode: 'lines', line: {{ color: '#d62728', width: 1.8, dash: 'dash' }},
                    name: 'Train 3 Only',
                    xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 3<br>Time: %{{x:.1f}} ms<br>FR: %{{y:.1f}} Hz<extra></extra>"
                }},
                // 3. Train PETH Mean (row 2, col 1)
                {{
                    x: DATA.train_x, y: u.train_fr,
                    mode: 'lines', line: {{ color: '#f0f6fc', width: 2.5 }},
                    name: 'Train Mean',
                    xaxis: 'x3', yaxis: 'y3',
                    hovertemplate: "Time: %{{x:.2f}} s<br>FR: %{{y:.1f}} Hz<extra></extra>"
                }},
                {{
                    x: DATA.train_x, y: u.t1_tr,
                    mode: 'lines', line: {{ color: '#1f77b4', width: 1.0 }}, opacity: 0.6,
                    name: 'Train 1', xaxis: 'x3', yaxis: 'y3', hoverinfo: 'skip'
                }},
                {{
                    x: DATA.train_x, y: u.t2_tr,
                    mode: 'lines', line: {{ color: '#2ca02c', width: 1.0 }}, opacity: 0.6,
                    name: 'Train 2', xaxis: 'x3', yaxis: 'y3', hoverinfo: 'skip'
                }},
                {{
                    x: DATA.train_x, y: u.t3_tr,
                    mode: 'lines', line: {{ color: '#d62728', width: 1.0 }}, opacity: 0.6,
                    name: 'Train 3', xaxis: 'x3', yaxis: 'y3', hoverinfo: 'skip'
                }},
                // 4. Breath PETH (row 2, col 2)
                {{
                    x: DATA.breath_x, y: u.breath_fr,
                    mode: 'lines', line: {{ color: '#00bc8c', width: 2.5 }},
                    name: 'Eupnea Breaths',
                    xaxis: 'x4', yaxis: 'y4',
                    hovertemplate: "Time: %{{x:.1f}} ms<br>FR: %{{y:.1f}} Hz<extra></extra>"
                }}
            ];

            const layout = {{
                template: 'plotly_dark',
                paper_bgcolor: '#161b22',
                plot_bgcolor: '#0d1117',
                font: {{ family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color: '#c9d1d9' }},
                height: 800,
                showlegend: false,
                margin: {{ l: 60, r: 40, t: 50, b: 60 }},
                grid: {{ rows: 2, columns: 2, pattern: 'independent' }},
                annotations: [
                    {{ text: "<b>1. Laser Pulse-Aligned Spike Raster (120 Pulses @ 20 Hz)</b>", xref: "paper", yref: "paper", x: 0.20, y: 1.04, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "<b>2. Laser Pulse PETH (Blue: All 120 | Red Dash: Train 3)</b>", xref: "paper", yref: "paper", x: 0.80, y: 1.04, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "<b>3. Laser Train PETH (-1.5s to +4.5s, 2s Photostimulation)</b>", xref: "paper", yref: "paper", x: 0.20, y: 0.46, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "<b>4. Inspiratory Breath PETH (-150ms to +350ms, Baseline Eupnea)</b>", xref: "paper", yref: "paper", x: 0.80, y: 0.46, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                ],
                shapes: [
                    {{ type: 'rect', xref: 'x', yref: 'y', x0: 0, x1: 5, y0: 0, y1: 121, fillcolor: '#ff9800', opacity: 0.25, line: {{ width: 0 }} }},
                    {{ type: 'rect', xref: 'x2', yref: 'paper', x0: 0, x1: 5, y0: 0.54, y1: 1.0, fillcolor: '#ff9800', opacity: 0.25, line: {{ width: 0 }} }},
                    {{ type: 'rect', xref: 'x3', yref: 'paper', x0: 0, x1: 1.955, y0: 0, y1: 0.42, fillcolor: '#ff9800', opacity: 0.25, line: {{ width: 0 }} }},
                    {{ type: 'line', xref: 'x4', yref: 'paper', x0: 0, x1: 0, y0: 0, y1: 0.42, line: {{ color: '#00bc8c', width: 1.5, dash: 'dash' }} }}
                ],
                xaxis: {{ title: "Time from Pulse Onset (ms)", range: [-15, 45], gridcolor: "#21262d" }},
                yaxis: {{ title: "Pulse # (1..120)", range: [0, 121], gridcolor: "#21262d" }},
                xaxis2: {{ title: "Time from Pulse Onset (ms)", range: [-15, 45], gridcolor: "#21262d" }},
                yaxis2: {{ title: "Firing Rate (Hz)", gridcolor: "#21262d" }},
                xaxis3: {{ title: "Time from Train Onset (s)", range: [-1.5, 4.5], gridcolor: "#21262d" }},
                yaxis3: {{ title: "Firing Rate (Hz)", gridcolor: "#21262d" }},
                xaxis4: {{ title: "Time from Insp Onset (ms)", range: [-150, 350], gridcolor: "#21262d" }},
                yaxis4: {{ title: "Firing Rate (Hz)", gridcolor: "#21262d" }}
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
    print(f"[+] Exported standalone interactive PETH HTML workbench to: {out_html}")

if __name__ == "__main__":
    main()
