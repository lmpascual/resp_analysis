#!/usr/bin/env python3
"""
Generate Comprehensive Spike Raster Figures and Interactive Raster Workbench
for e097 across the 3 Laser Trains (-10s Pre-Laser to +10s Post-Laser Offset):
- Window: [-10.0s, +12.0s] relative to laser train onset (22 seconds total).
- Phase 1: 10s Pre-Laser Eupnea [-10.0s, 0.0s]
- Phase 2: 2s Laser Apnea [0.0s, 1.955s] (40 pulses @ 20 Hz, 5 ms)
- Phase 3a: ~4s Post-Apneic Transition & Sigh [1.955s, 5.955s]
- Phase 3b: 6s Resumed Normal Breathing [5.955s, 11.955s]

Outputs:
1. population_3trials_spike_rasters.png / .svg:
   - Full population spike raster of all 162 single units sorted by probe depth,
     stacked across Train 1, Train 2, and Train 3, paired with Airflow, EMG, and Laser.
2. exemplary_units_3trials_spike_rasters.png / .svg:
   - Dedicated high-definition spike rasters for key archetypes (Cluster 315, 13, 366, 269, 430, 298, 448, 18, 82, 316).
3. e097_laser_train_interactive_raster_workbench.html:
   - Interactive raster workbench featuring full population raster navigation and single-unit zoomable multi-trial rasters.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    print("[*] Starting dedicated Spike Raster generation for e097 (-10s to +12s)...")
    base_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    alf_dir = base_dir / "alf"
    out_dir = base_dir / "peth_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Spikes & Units
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    pr_df = pd.read_csv(base_dir / "detailed_presence_ratio_comparison.csv")

    target_cids = list(pr_df["cluster_id"].values)
    if 448 not in target_cids and 448 in spk_clusters:
        target_cids.append(448)

    # Map each cluster to its depth and anatomical zone
    cluster_info = {}
    for cid in target_cids:
        row = pr_df[pr_df["cluster_id"] == cid]
        if len(row) > 0:
            d = float(row.iloc[0]["depth_um"])
            is_recruited = bool(row.iloc[0].get("is_newly_recruited", False))
        else:
            d = 1380.0
            is_recruited = False
        cluster_info[cid] = {"depth": d, "is_recruited": is_recruited}

    # Sort target clusters by depth (from rostral/top 0 um to caudal/bottom 2775 um)
    sorted_cids = sorted(target_cids, key=lambda c: cluster_info[c]["depth"])
    cid_to_y = {cid: idx for idx, cid in enumerate(sorted_cids)}

    print(f"Total units to plot in rasters: {len(sorted_cids)} units.")

    # 2. Load Analog Channels (Airflow, EMG, Laser)
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
    t_pre = -10.0
    t_post = train_dur + 10.0  # 11.955 s
    t_resump = train_dur + 4.0 # 5.955 s

    # 4. Extract Spikes for each Train
    train_spikes = {0: [], 1: [], 2: []} # list of (rel_time, y_idx, cluster_id, depth)
    unit_train_spikes = {cid: {0: [], 1: [], 2: []} for cid in sorted_cids}

    target_set = set(sorted_cids)
    mask_targets = np.isin(spk_clusters, sorted_cids)
    target_spk_times = spk_times[mask_targets]
    target_spk_cids = spk_clusters[mask_targets]

    for tr_i, ts in enumerate(train_starts):
        m = (target_spk_times >= ts + t_pre) & (target_spk_times <= ts + t_post)
        sub_times = target_spk_times[m] - ts
        sub_cids = target_spk_cids[m]

        for st, sc in zip(sub_times, sub_cids):
            y = cid_to_y[sc]
            d = cluster_info[sc]["depth"]
            train_spikes[tr_i].append((st, y, sc, d))
            unit_train_spikes[sc][tr_i].append(st)

    print(f"Spike extraction complete. Total ticks per train: T1={len(train_spikes[0])}, T2={len(train_spikes[1])}, T3={len(train_spikes[2])}")

    # Downsample physio traces to 1000 Hz for figures
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
    # FIGURE 1: FULL POPULATION SPIKE RASTERS (TRAIN 1, 2, 3 STACKED)
    # =========================================================================
    print("[*] Rendering Full Population Spike Rasters (All 162 Units across 3 Trains)...")
    render_full_population_raster_figure(
        train_spikes=train_spikes,
        sorted_cids=sorted_cids,
        cluster_info=cluster_info,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials, laser_trials=laser_trials,
        train_starts=train_starts,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur, t_resump=t_resump,
        out_png=out_dir / "population_3trials_spike_rasters.png",
        out_svg=out_dir / "population_3trials_spike_rasters.svg"
    )

    # =========================================================================
    # FIGURE 2: EXEMPLARY UNITS DEDICATED HIGH-RESOLUTION RASTERS
    # =========================================================================
    print("[*] Rendering Exemplary Units High-Definition Spike Rasters...")
    exemplar_ids = [
        315, # Recruited BötC
        13,  # Recruited Far Rostral
        366, # Recruited Far Rostral
        269, # Recruited rVRC
        430, # Unit 2 Pacemaker
        298, # Unit 255 Ramping Follower
        448, # Unit 163 Pre-I Pacemaker
        18,  # Unit 18 BötC Expiratory
        82,  # Laser Surge Unit
        316, # Recruited cVRG Premotor
    ]

    render_exemplar_units_raster_figure(
        exemplar_ids=exemplar_ids,
        unit_train_spikes=unit_train_spikes,
        cluster_info=cluster_info,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur, t_resump=t_resump,
        out_png=out_dir / "exemplary_units_3trials_spike_rasters.png",
        out_svg=out_dir / "exemplary_units_3trials_spike_rasters.svg"
    )

    # =========================================================================
    # 3. INTERACTIVE HTML WORKBENCH FOR SPIKE RASTERS
    # =========================================================================
    print("[*] Generating Interactive Spike Raster Workbench...")
    generate_interactive_raster_workbench(
        sorted_cids=sorted_cids,
        cluster_info=cluster_info,
        unit_train_spikes=unit_train_spikes,
        t_rel=t_rel, flow_trials=flow_trials, emg_trials=emg_trials,
        t_pre=t_pre, t_post=t_post, train_dur=train_dur, t_resump=t_resump,
        out_html=out_dir / "e097_laser_train_interactive_raster_workbench.html"
    )

    print("[DONE] All Spike Raster deliverables generated successfully!")

def get_region_color(depth):
    """Assigns color based on anatomical depth."""
    if depth < 750.0:
        return "#e64980" # BötC / Rostral VRC: Pink / Red
    elif depth < 1650.0:
        return "#228be6" # preBötC core: Blue
    elif depth < 2200.0:
        return "#12b886" # rVRG: Teal / Green
    else:
        return "#7950f2" # cVRG: Purple

def render_full_population_raster_figure(train_spikes, sorted_cids, cluster_info,
                                         t_rel, flow_trials, emg_trials, laser_trials,
                                         train_starts, t_pre, t_post, train_dur, t_resump,
                                         out_png, out_svg):
    """
    Renders a dedicated 3-panel publication figure showing EVERY SPIKE TICK for all 162 SUA units:
    - Panel A: Train 1 (-10s to +12s)
    - Panel B: Train 2 (-10s to +12s)
    - Panel C: Train 3 (-10s to +12s)
    Each panel has:
      - Top subpanel: Synchronized Airflow, EMG, and Laser
      - Main subpanel: 162 unit rows with color-coded spike ticks (BötC pink, preBötC blue, rVRG teal, cVRG purple).
    """
    fig = plt.figure(figsize=(22, 17))
    fig.patch.set_facecolor('white')

    outer_grid = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.22)
    titles = [
        f"TRAIN 1 (Onset = {train_starts[0]:.2f}s) — Population Spike Raster (All 162 SUA Units)",
        f"TRAIN 2 (Onset = {train_starts[1]:.2f}s) — Population Spike Raster (All 162 SUA Units)",
        f"TRAIN 3 (Onset = {train_starts[2]:.2f}s) — Population Spike Raster (All 162 SUA Units)"
    ]

    n_units = len(sorted_cids)
    depths = [cluster_info[c]["depth"] for c in sorted_cids]

    # Pre-compute colors for each unit row
    unit_colors = [get_region_color(d) for d in depths]

    for tr_i in range(3):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[tr_i], height_ratios=[0.20, 0.80], hspace=0.04)

        # 1. Physio Subpanel
        ax_top = fig.add_subplot(inner_grid[0])
        ax_top.set_facecolor('#fafafa')
        ax_top.axvspan(t_pre, 0.0, color='#e9ecef', alpha=0.5)
        ax_top.axvspan(0.0, train_dur, color='#ffe8cc', alpha=0.6)
        ax_top.axvspan(train_dur, t_resump, color='#fff3bf', alpha=0.45)
        ax_top.axvspan(t_resump, t_post, color='#e6fcf5', alpha=0.45)
        ax_top.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.3, alpha=0.85)

        # Plot Airflow & EMG
        ax_top.plot(t_rel, flow_trials[tr_i], color='#009688', linewidth=1.1, label='Airflow')
        ax_top.plot(t_rel, emg_trials[tr_i] * 3.5 - 2.0, color='#3f51b5', linewidth=0.85, alpha=0.8, label='Diaphragm EMG')
        # Laser pulses
        ax_top.plot(t_rel, laser_trials[tr_i] * 0.4 + 2.2, color='#ff922b', linewidth=1.0, alpha=0.9, label='Laser Stim (20 Hz)')

        ax_top.set_xlim(t_pre, t_post)
        ax_top.set_ylim(-3.5, 4.0)
        ax_top.set_title(titles[tr_i], fontsize=11, fontweight='bold', pad=4)
        ax_top.set_ylabel("Signals (V)", fontsize=8, fontweight='bold')
        ax_top.set_xticklabels([])
        if tr_i == 0:
            ax_top.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)

        # 2. Spike Raster Subpanel
        ax_ras = fig.add_subplot(inner_grid[1])
        ax_ras.set_facecolor('#12161c') # Sleek dark background for high-contrast spike ticks

        # Shading
        ax_ras.axvspan(0.0, train_dur, color='#ff9800', alpha=0.15)
        ax_ras.axvspan(train_dur, t_resump, color='#fcc419', alpha=0.08)
        ax_ras.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.3, alpha=0.9)

        # Group spikes by unit for batch plotting
        spk_data = train_spikes[tr_i]
        if len(spk_data) > 0:
            x_vals = [s[0] for s in spk_data]
            y_vals = [s[1] for s in spk_data]
            c_vals = [unit_colors[s[1]] for s in spk_data]

            ax_ras.scatter(x_vals, y_vals, s=2.5, c=c_vals, marker='|', linewidth=0.7, alpha=0.85)

        ax_ras.set_xlim(t_pre, t_post)
        ax_ras.set_ylim(-0.5, n_units - 0.5)
        ax_ras.invert_yaxis() # Top = 0 um (Rostral), Bottom = 2775 um (Caudal)

        # Y-ticks: show probe depths and anatomical landmarks
        tick_indices = [0, 25, 60, 100, 135, n_units - 1]
        tick_labels = [f"0 μm (BötC)", f"500 μm", f"1100 μm (preBötC)", f"1750 μm (rVRG)", f"2200 μm (cVRG)", f"{depths[-1]:.0f} μm"]
        ax_ras.set_yticks(tick_indices)
        ax_ras.set_yticklabels(tick_labels, fontsize=8, fontweight='bold', color='#333333')
        ax_ras.set_ylabel("Probe Depth / Region", fontsize=9, fontweight='bold')

        # Annotate regions on the left
        ax_ras.axhline(35, color='#e64980', linestyle=':', linewidth=0.8, alpha=0.6)
        ax_ras.axhline(85, color='#228be6', linestyle=':', linewidth=0.8, alpha=0.6)
        ax_ras.axhline(125, color='#12b886', linestyle=':', linewidth=0.8, alpha=0.6)

        if tr_i == 2:
            ax_ras.set_xlabel("Time from Laser Train Onset (s)  [Green dashed line: Resumption into normal breathing (~4s post-offset)]", fontsize=10, fontweight='bold')
        else:
            ax_ras.set_xticklabels([])

    plt.suptitle("FULL POPULATION SPIKE RASTERS ACROSS 3 LASER TRAINS (162 SUA Units, -10s Pre to +10s Post-Offset)\nPink: BötC/Rostral | Blue: preBötC Core | Teal: rVRG | Purple: cVRG", fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.04)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved population spike raster figure to {out_png.name} and {out_svg.name}")

def render_exemplar_units_raster_figure(exemplar_ids, unit_train_spikes, cluster_info,
                                        t_rel, flow_trials, emg_trials,
                                        t_pre, t_post, train_dur, t_resump,
                                        out_png, out_svg):
    """
    Renders dedicated multi-trial spike rasters for 10 exemplary units.
    Each unit gets a prominent 3-row raster (Train 1, Train 2, Train 3) with large crisp ticks,
    accompanied by the Train 3 airflow and EMG trace overlaid in the background.
    """
    n_units = len(exemplar_ids)
    fig, axes = plt.subplots(n_units, 1, figsize=(18, 1.8 * n_units), sharex=True)
    fig.patch.set_facecolor('white')

    trial_colors = ['#1f77b4', '#2ca02c', '#d62728'] # T1 Blue, T2 Green, T3 Red
    trial_labels = ['Train 1', 'Train 2', 'Train 3']

    for i, cid in enumerate(exemplar_ids):
        ax = axes[i]
        ax.set_facecolor('#fafafa')
        depth = cluster_info[cid]["depth"]

        # Shading
        ax.axvspan(t_pre, 0.0, color='#e9ecef', alpha=0.45)
        ax.axvspan(0.0, train_dur, color='#ffe8cc', alpha=0.6)
        ax.axvspan(train_dur, t_resump, color='#fff3bf', alpha=0.45)
        ax.axvspan(t_resump, t_post, color='#e6fcf5', alpha=0.45)
        ax.axvline(t_resump, color='#20c997', linestyle='--', linewidth=1.2, alpha=0.85)

        # Plot Airflow in background (Train 3 flow in subtle gray)
        norm_flow = (flow_trials[2] - np.min(flow_trials[2])) / (np.max(flow_trials[2]) - np.min(flow_trials[2]) + 1e-5) * 2.8 + 0.6
        ax.plot(t_rel, norm_flow, color='#adb5bd', alpha=0.35, linewidth=0.8)

        # Plot raster ticks for each trial (y = 1, 2, 3)
        for tr_i in range(3):
            spks = unit_train_spikes[cid][tr_i]
            if len(spks) > 0:
                ax.scatter(spks, np.full_like(spks, tr_i + 1), s=18, color=trial_colors[tr_i], marker='|', linewidth=1.6, alpha=0.9)

        name_str = f"Cluster {cid} ({depth:.0f} μm)"
        if cid == 430: name_str += " — Unit 2 Pacemaker"
        elif cid == 298: name_str += " — Unit 255 Ramping Follower"
        elif cid == 448: name_str += " — Unit 163 Pre-I Pacemaker"
        elif cid == 18: name_str += " — Unit 18 BötC Expiratory"
        elif cid == 315: name_str += " — Recruited BötC Interneuron"
        elif cid == 13: name_str += " — Recruited Far Rostral"
        elif cid == 366: name_str += " — Recruited Far Rostral"
        elif cid == 269: name_str += " — Recruited rVRC"
        elif cid == 82: name_str += " — Laser Surge Unit"
        elif cid == 316: name_str += " — Recruited cVRG"

        ax.set_xlim(t_pre, t_post)
        ax.set_ylim(0.4, 3.6)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['T1', 'T2', 'T3'], fontsize=8, fontweight='bold')
        ax.set_ylabel(name_str, fontsize=8.5, fontweight='bold')

        # Annotation of spike counts
        c1 = len(unit_train_spikes[cid][0])
        c2 = len(unit_train_spikes[cid][1])
        c3 = len(unit_train_spikes[cid][2])
        ax.text(0.99, 0.88, f"Spikes: T1={c1}, T2={c2}, T3={c3}", transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='#ccc'))

        if i == 0:
            ax.set_title("EXEMPLARY UNITS: 3-TRIAL SPIKE RASTERS (-10s Pre to +10s Post-Offset)\nBlue: Train 1 | Green: Train 2 | Red: Train 3 | Shaded: Pre (10s), Laser (2s), Transition (4s), Resumed (6s)", fontsize=11, fontweight='bold')

    axes[-1].set_xlabel("Time from Laser Train Onset (s)  [Green dashed line: Normal Resumption (~4s post-offset)]", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved exemplar units spike raster figure to {out_png.name} and {out_svg.name}")

def generate_interactive_raster_workbench(sorted_cids, cluster_info, unit_train_spikes,
                                          t_rel, flow_trials, emg_trials,
                                          t_pre, t_post, train_dur, t_resump,
                                          out_html):
    """
    Builds an interactive HTML application featuring:
    1. Single-Unit Multi-Trial Spike Raster View (prominent raster canvas, zoomable, with true tick marks).
    2. Interactive trial toggles and spike timestamp inspector.
    """
    units_data = {}
    for cid in sorted_cids:
        d = cluster_info[cid]["depth"]
        units_data[str(cid)] = {
            "cid": int(cid),
            "depth": round(d, 1),
            "region": "BötC" if d < 750 else ("preBötC" if d < 1650 else ("rVRG" if d < 2200 else "cVRG")),
            "t1_spks": [round(float(x), 3) for x in unit_train_spikes[cid][0]],
            "t2_spks": [round(float(x), 3) for x in unit_train_spikes[cid][1]],
            "t3_spks": [round(float(x), 3) for x in unit_train_spikes[cid][2]],
        }

    # Subsampled physio traces (100 Hz for lightweight embedding)
    flow_sub = [round(float(x), 2) for x in flow_trials[2][::10]]
    emg_sub = [round(float(x), 3) for x in emg_trials[2][::10]]
    physio_x = [round(float(x), 3) for x in t_rel[::10]]

    json_payload = json.dumps({
        "physio_x": physio_x,
        "flow_t3": flow_sub,
        "emg_t3": emg_sub,
        "units": units_data
    })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>e097 3-Trial Laser Train Spike Raster Interactive Workbench</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b0f19;
            color: #c9d1d9;
            margin: 0;
            padding: 16px 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #21262d;
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        h1 {{
            font-size: 21px;
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
            padding: 12px;
            min-height: 780px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>e097 3-Trial Laser Train Spike Raster Interactive Workbench</h1>
            <p class="subtitle">Individual Spike Ticks across Train 1, Train 2, and Train 3: -10s Pre-Laser | 2s Laser Apnea | ~4s Transition & Sigh | 6s Resumed Breathing</p>
        </div>
        <div>
            <span class="stat-badge">Animal: <b>e097</b> | Window: <b>-10.0s to +12.0s</b> | Total Duration: <b>22.0s</b></span>
        </div>
    </div>

    <div class="controls">
        <label for="unit-select"><b>Select Single Unit:</b></label>
        <select id="unit-select" onchange="updateUnit(this.value)">
            <optgroup label="⭐ Recruited Rostral Units (BötC / Far Rostral)">
                <option value="315" selected>Cluster 315 (435 μm, BötC) — Awakens 0 → 60 Hz in Train 3</option>
                <option value="13">Cluster 13 (15 μm, Far Rostral) — Recruited in Train 3</option>
                <option value="343">Cluster 343 (15 μm, Far Rostral) — Recruited in Train 3</option>
                <option value="366">Cluster 366 (30 μm, Far Rostral) — Recruited in Train 3</option>
                <option value="269">Cluster 269 (690 μm, rVRC) — Surges in Post-Resumption</option>
            </optgroup>
            <optgroup label="⭐ Tracked Baseline Archetypes">
                <option value="430">Cluster 430 (615 μm) [Unit 2 Pacemaker] — Apnea Quenched (17 Hz → 1 Hz)</option>
                <option value="298">Cluster 298 (308 μm) [Unit 255 Ramping] — Apnea Quenched (60 Hz → 16 Hz)</option>
                <option value="448">Cluster 448 (1380 μm) [Unit 163 Pre-I] — Pulse-Locked Bursts</option>
                <option value="18">Cluster 18 (645 μm) [Unit 18 BötC Exp] — Apnea Quenched & Sigh Silenced</option>
                <option value="82">Cluster 82 (1913 μm) [Laser Surge] — 33 Hz Surges across all Trains</option>
                <option value="316">Cluster 316 (2010 μm) [cVRG Premotor] — Recruited 80 Hz in Train 3</option>
            </optgroup>
            <optgroup label="All 162 Single Units (Sorted by Depth)" id="all-units-group">
            </optgroup>
        </select>
        <span class="stat-badge" id="badge-depth">Depth: <b>435 μm (BötC)</b></span>
        <span class="stat-badge" id="badge-spikes">Spikes: T1=<b>0</b> | T2=<b>6</b> | T3=<b>642</b></span>
    </div>

    <div id="plot-container"></div>

    <script>
        const DATA = {json_payload};
        const allGroup = document.getElementById("all-units-group");

        const uList = Object.values(DATA.units).sort((a, b) => a.depth - b.depth);
        uList.forEach(u => {{
            const opt = document.createElement("option");
            opt.value = u.cid;
            opt.textContent = `Cluster ${{u.cid}} (${{u.depth}} μm, ${{u.region}}, T3=${{u.t3_spks.length}} spks)`;
            allGroup.appendChild(opt);
        }});

        function updateUnit(uid) {{
            const u = DATA.units[uid];
            if (!u) return;

            document.getElementById("badge-depth").innerHTML = `Depth: <b>${{u.depth}} μm (${{u.region}})</b>`;
            document.getElementById("badge-spikes").innerHTML = `Spike Counts: Train 1=<b>${{u.t1_spks.length}}</b> | Train 2=<b>${{u.t2_spks.length}}</b> | Train 3=<b>${{u.t3_spks.length}}</b>`;

            const traces = [
                // Top subpanel: Airflow (Train 3)
                {{
                    x: DATA.physio_x, y: DATA.flow_t3,
                    mode: 'lines', line: {{ color: '#00bc8c', width: 1.5 }},
                    name: 'Airflow (T3)', xaxis: 'x', yaxis: 'y'
                }},
                // Diaphragm EMG (Train 3)
                {{
                    x: DATA.physio_x, y: DATA.emg_t3.map(v => v * 4.0 - 2.5),
                    mode: 'lines', line: {{ color: '#375a7f', width: 1.2 }},
                    name: 'Diaphragm EMG (T3)', xaxis: 'x', yaxis: 'y'
                }},

                // Main subpanel: True Spike Rasters
                // Train 1 (y = 1)
                {{
                    x: u.t1_spks, y: u.t1_spks.map(() => 1),
                    mode: 'markers', marker: {{ symbol: 'line-ns', size: 28, color: '#3498db', line: {{ width: 2.2 }} }},
                    name: 'Train 1 Spikes', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 1<br>Time: %{{x:.3f}} s<extra></extra>"
                }},
                // Train 2 (y = 2)
                {{
                    x: u.t2_spks, y: u.t2_spks.map(() => 2),
                    mode: 'markers', marker: {{ symbol: 'line-ns', size: 28, color: '#2ecc71', line: {{ width: 2.2 }} }},
                    name: 'Train 2 Spikes', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 2<br>Time: %{{x:.3f}} s<extra></extra>"
                }},
                // Train 3 (y = 3)
                {{
                    x: u.t3_spks, y: u.t3_spks.map(() => 3),
                    mode: 'markers', marker: {{ symbol: 'line-ns', size: 28, color: '#e74c3c', line: {{ width: 2.5 }} }},
                    name: 'Train 3 Spikes', xaxis: 'x2', yaxis: 'y2',
                    hovertemplate: "Train 3<br>Time: %{{x:.3f}} s<extra></extra>"
                }}
            ];

            const layout = {{
                template: 'plotly_dark',
                paper_bgcolor: '#161b22',
                plot_bgcolor: '#0d1117',
                font: {{ family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color: '#c9d1d9' }},
                height: 760,
                showlegend: true,
                legend: {{ orientation: 'h', x: 0.15, y: 1.05 }},
                margin: {{ l: 80, r: 40, t: 50, b: 60 }},
                grid: {{ rows: 2, columns: 1, pattern: 'independent' }},
                shapes: [
                    // Top: Laser Shading [0, 1.955s]
                    {{ type: 'rect', xref: 'x', yref: 'paper', x0: 0, x1: 1.955, y0: 0.78, y1: 1.0, fillcolor: '#ff9800', opacity: 0.25, line: {{ width: 0 }} }},
                    // Top: Transition [1.955, 5.955s]
                    {{ type: 'rect', xref: 'x', yref: 'paper', x0: 1.955, x1: {t_resump}, y0: 0.78, y1: 1.0, fillcolor: '#fcc419', opacity: 0.12, line: {{ width: 0 }} }},
                    // Top: Resumption Line
                    {{ type: 'line', xref: 'x', yref: 'paper', x0: {t_resump}, x1: {t_resump}, y0: 0.78, y1: 1.0, line: {{ color: '#20c997', width: 1.8, dash: 'dash' }} }},

                    // Raster: Laser Shading [0, 1.955s]
                    {{ type: 'rect', xref: 'x2', yref: 'y2', x0: 0, x1: 1.955, y0: 0.3, y1: 3.7, fillcolor: '#ff9800', opacity: 0.20, line: {{ width: 0 }} }},
                    // Raster: Transition [1.955, 5.955s]
                    {{ type: 'rect', xref: 'x2', yref: 'y2', x0: 1.955, x1: {t_resump}, y0: 0.3, y1: 3.7, fillcolor: '#fcc419', opacity: 0.10, line: {{ width: 0 }} }},
                    // Raster: Resumption Line
                    {{ type: 'line', xref: 'x2', yref: 'y2', x0: {t_resump}, x1: {t_resump}, y0: 0.3, y1: 3.7, line: {{ color: '#20c997', width: 2.0, dash: 'dash' }} }}
                ],
                annotations: [
                    {{ text: "<b>Airflow & Diaphragm EMG (Train 3 Overlay)</b>", xref: "paper", yref: "paper", x: 0.01, y: 1.01, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "<b>3-Trial Spike Raster (Individual Action Potentials)</b>", xref: "paper", yref: "paper", x: 0.01, y: 0.73, showarrow: false, font: {{ size: 12, color: "#58a6ff" }} }},
                    {{ text: "← 10s Pre-Laser Eupnea", xref: "x2", yref: "paper", x: -5.0, y: 0.02, showarrow: false, font: {{ size: 11, color: "#8b949e" }} }},
                    {{ text: "Laser (2s)", xref: "x2", yref: "paper", x: 1.0, y: 0.02, showarrow: false, font: {{ size: 11, color: "#ff9800" }} }},
                    {{ text: "Trans (~4s)", xref: "x2", yref: "paper", x: 3.9, y: 0.02, showarrow: false, font: {{ size: 10, color: "#fcc419" }} }},
                    {{ text: "Resumed Breathing (6s) →", xref: "x2", yref: "paper", x: 8.8, y: 0.02, showarrow: false, font: {{ size: 11, color: "#20c997" }} }}
                ],
                xaxis: {{ domain: [0, 1], title: "", range: [{t_pre}, {t_post}], gridcolor: "#21262d" }},
                yaxis: {{ domain: [0.78, 1.0], title: "Signals (V)", gridcolor: "#21262d" }},
                xaxis2: {{ domain: [0, 1], title: "Time from Laser Train Onset (s)  [Green dashed line: Normal Resumption (~4s post-offset)]", range: [{t_pre}, {t_post}], gridcolor: "#21262d" }},
                yaxis2: {{ domain: [0.05, 0.72], title: "Photostimulation Train", range: [0.3, 3.7], tickvals: [1, 2, 3], ticktext: ['Train 1', 'Train 2', 'Train 3'], gridcolor: "#21262d" }}
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
    print(f"[+] Exported interactive spike raster workbench to: {out_html}")

if __name__ == "__main__":
    main()
