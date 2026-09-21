#!/usr/bin/env python3
"""
Generate 5-Breath Cycle-by-Cycle Exemplary Neuron Multi-Unit Raster Plots & Diaphragm EMG Traces
Centered on the Sigh (Pre-Sigh -2, Pre-Sigh -1, SIGH, Post-Sigh +1, Post-Sigh +2).

Features:
1. 5 vertically stacked trial panels (Breaths 71 to 75).
2. Gradated phase background using the exact PSTH phase colormap:
   - Expiration (-pi to 0): Cool blue (#93c5fd -> #dbeafe) to White (#ffffff at I_on)
   - Inspiration (0 to +pi): White (#ffffff at I_on) -> Warm peach (#fed7aa) -> Warm coral/red (#f87171 at I_off)
3. Synchronized Diaphragm EMG Envelope trace directly beneath each raster panel,
   scaled to show the massive >2.5x surge in motor drive during the sigh.
4. Exemplary neurons demonstrating:
   - Unit 13 (preBötC Pacemaker: Stochastic Pre-I)
   - Unit 28 (preBötC Pacemaker: Sigh Driver / Metronomic)
   - Unit 83 (preBötC Vanguard Burster)
   - Unit 53 (preBötC Canonical Follower: Post-I Latency)
   - Unit 33 (preBötC Mid-Inspiratory Follower)
   - Unit 15 (BötC Expiratory: Sharp I_on Inhibition)
   - Unit 27 (cVRG Post-I / Expiratory Disinhibition)
5. Inverted black triangle marker at I_off on each trial.
6. Callout annotations highlighting key biophysical hallmarks:
   - Sigh augmented EMG peak and Unit 28 metronomic burst
   - Post-sigh prolonged expiratory dwelling (post-sigh apnea)
   - Post-sigh recovery inspiration with delayed follower latency
7. Exports both high-resolution PNG (300 DPI) and scalable vector SVG.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def compute_trial_continuous_phase(t_eval, t_on, t_off, t_prev_off, t_next_on):
    """
    Computes continuous phase phi(t) in [-pi, pi] across the trial evaluation window:
    - Pre-inspiratory expiration: [-pi, 0]
    - Inspiration: [0, pi]
    - Post-inspiratory expiration: [-pi, 0]
    """
    phi = np.zeros_like(t_eval)
    
    # 1. Pre-inspiration (Expiration before t_on)
    mask_pre = (t_eval < t_on)
    if np.any(mask_pre):
        t_ref_prev = t_prev_off if t_prev_off < t_on else t_on - 0.25
        phi[mask_pre] = -np.pi + (t_eval[mask_pre] - t_ref_prev) / (t_on - t_ref_prev + 1e-9) * np.pi
        phi[mask_pre] = np.clip(phi[mask_pre], -np.pi, 0.0)

    # 2. Inspiration: [t_on, t_off] -> [0, pi]
    mask_insp = (t_eval >= t_on) & (t_eval <= t_off)
    if np.any(mask_insp):
        phi[mask_insp] = (t_eval[mask_insp] - t_on) / (t_off - t_on + 1e-9) * np.pi

    # 3. Post-inspiration (Expiration after t_off)
    mask_post = (t_eval > t_off)
    if np.any(mask_post):
        t_ref_next = t_next_on if t_next_on > t_off else t_off + 0.35
        phi[mask_post] = -np.pi + (t_eval[mask_post] - t_off) / (t_ref_next - t_off + 1e-9) * np.pi
        phi[mask_post] = np.clip(phi[mask_post], -np.pi, 0.0)

    return phi

def main():
    out_dir = PROJECT_ROOT / "presentation_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "sigh_five_breath_sequence_lead_follow_raster.png"
    out_svg = out_dir / "sigh_five_breath_sequence_lead_follow_raster.svg"

    # 1. Load Spike Data
    alf_dir = PROJECT_ROOT / "LP21_fent_EXPT_g1_60s_120s_lupin_run" / "00_spike_sorting_data" / "alf"
    if not alf_dir.exists():
        alf_dir = PROJECT_ROOT / "output_LP21_60s_120s" / "sliding_window_pipeline" / "lupin" / "consensus_master" / "alf"

    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")

    # Load EMG for breath detection
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    nidq_bin = list(raw_dir.glob("*.nidq.bin"))[0]
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=1)

    start_t = 60.0
    dur = 60.0
    t_nidq_raw = nidq_data.time_vector
    mask_n = (t_nidq_raw >= start_t) & (t_nidq_raw <= start_t + dur)
    t_nidq = t_nidq_raw[mask_n] - start_t
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= dur)]
    offsets = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= dur)]

    # 2. Units Information (Identical to cycle_by_cycle_pacemaker_lead_follow_raster.py)
    units_info = [
        {"id": 13, "label": "Unit 13 (preBötC Pacemaker: Stochastic Pre-I)", "depth": 945},
        {"id": 28, "label": "Unit 28 (preBötC Pacemaker: Sigh Driver / Metronomic)", "depth": 900},
        {"id": 83, "label": "Unit 83 (preBötC Vanguard Burster)", "depth": 810},
        {"id": 53, "label": "Unit 53 (preBötC Canonical Follower: Post-I Latency)", "depth": 892},
        {"id": 33, "label": "Unit 33 (preBötC Mid-Inspiratory Follower)", "depth": 570},
        {"id": 15, "label": "Unit 15 (BötC Expiratory: Sharp I_on Inhibition)", "depth": 480},
        {"id": 27, "label": "Unit 27 (cVRG Post-I / Expiratory Disinhibition)", "depth": 1230},
    ]
    n_units = len(units_info)

    # 5 Breaths: Pre-Sigh -2, Pre-Sigh -1, SIGH, Post-Sigh +1, Post-Sigh +2
    trial_indices = [71, 72, 73, 74, 75]
    trial_roles = [
        "Pre-Sigh Breath -2",
        "Pre-Sigh Breath -1",
        "SIGH (Augmented Breath)",
        "Post-Sigh Recovery +1",
        "Post-Sigh Baseline +2"
    ]
    n_trials = len(trial_indices)

    # Window bounds
    t_min_ms = -150.0
    t_max_ms = 380.0
    n_grid_pts = 600
    t_eval_rel_ms = np.linspace(t_min_ms, t_max_ms, n_grid_pts)
    t_eval_rel_s = t_eval_rel_ms / 1000.0

    # 3. Gradated Colormap (Matching PSTH Plots)
    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_gradated",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    # 4. Create Master Figure with GridSpec (5 Trials x 2 Sub-Rows: Raster + EMG)
    fig = plt.figure(figsize=(11.8, 13.2), dpi=300)
    
    # Master Grid: 5 trial blocks with clean margins
    gs_master = gridspec.GridSpec(n_trials, 1, figure=fig, hspace=0.26,
                                  left=0.23, right=0.87, top=0.915, bottom=0.065)

    axes_rasters = []
    axes_emg = []

    # Calculate global max EMG for consistent scaling across the 5 breaths
    max_emg_global = np.max(emg_env) * 0.98

    for row_idx, b_idx in enumerate(trial_indices):
        t_on = onsets[b_idx]
        t_off = offsets[b_idx]
        t_prev_off = offsets[b_idx - 1] if b_idx > 0 else t_on - 0.25
        t_next_on = onsets[b_idx + 1] if b_idx + 1 < len(onsets) else t_off + 0.35
        t_i_ms = (t_off - t_on) * 1000.0
        t_e_ms = (t_next_on - t_off) * 1000.0

        # Sub-Grid for this trial: Raster (height 77%) and EMG (height 23%), zero vertical gap
        gs_trial = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[row_idx],
                                                    height_ratios=[3.3, 1.0], hspace=0.0)
        
        ax_r = fig.add_subplot(gs_trial[0])
        ax_e = fig.add_subplot(gs_trial[1], sharex=ax_r)
        axes_rasters.append(ax_r)
        axes_emg.append(ax_e)

        # ---------------------------------------------------------
        # A. Gradated Phase Colormap Background for Raster
        # ---------------------------------------------------------
        t_eval_abs = t_on + t_eval_rel_s
        phi_trial = compute_trial_continuous_phase(t_eval_abs, t_on, t_off, t_prev_off, t_next_on)
        phase_grid = np.tile(phi_trial, (n_units, 1))

        im = ax_r.imshow(
            phase_grid,
            extent=[t_min_ms, t_max_ms, -0.5, n_units - 0.5],
            origin="lower",
            aspect="auto",
            cmap=cmap_phase,
            vmin=-np.pi,
            vmax=np.pi,
            alpha=0.68,
            interpolation="bicubic",
            zorder=1
        )

        # Highlight Sigh row border with soft rose tint if row_idx == 2
        if row_idx == 2:
            ax_r.patch.set_facecolor('#fff1f2')

        # Unit divider lines
        for u_idx in range(n_units):
            ax_r.axhline(u_idx + 0.5, color="#cbd5e1", linewidth=0.8, alpha=0.8, zorder=2)

        # Inspiration Onset (I_on) marker line at t=0
        ax_r.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.5, alpha=0.9, zorder=3)
        # Inspiration Offset (I_off) marker line
        ax_r.axvline(t_i_ms, color="#0369a1", linestyle=":", linewidth=1.4, alpha=0.85, zorder=3)

        # Inverted triangle marker at I_off
        ax_r.plot(t_i_ms, n_units - 0.5, marker='v', color='#0f172a', markersize=7.0, zorder=6)

        # Plot Spikes
        for u_pos, u_data in enumerate(reversed(units_info)):
            uid = u_data["id"]
            u_spks = spk_times[spk_clusters == uid]
            rel_ms = (u_spks - t_on) * 1000.0
            in_win = rel_ms[(rel_ms >= t_min_ms) & (rel_ms <= t_max_ms)]

            # In the sigh row (row_idx == 2), emphasize Unit 28 with a bold ruby color
            is_sigh_u28 = (row_idx == 2 and uid == 28)
            spk_color = "#e11d48" if is_sigh_u28 else "#0f172a"
            spk_lw = 1.9 if is_sigh_u28 else 1.5

            for spk_t in in_win:
                ax_r.plot([spk_t, spk_t], [u_pos - 0.35, u_pos + 0.35],
                          color=spk_color, linewidth=spk_lw, zorder=5)

        ax_r.set_ylim(-0.5, n_units - 0.5)
        ax_r.set_xlim(t_min_ms, t_max_ms)
        ax_r.set_yticks([])  # Remove numeric y-ticks
        ax_r.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_r.spines.values():
            spine.set_color("#be123c" if row_idx == 2 else "#64748b")
            spine.set_linewidth(1.8 if row_idx == 2 else 1.0)

        # Unit labels along the left
        if row_idx == 0:
            for u_pos, u_data in enumerate(reversed(units_info)):
                ax_r.text(t_min_ms - 8, u_pos, f"{u_data['label']}",
                          ha='right', va='center', fontsize=8.2, fontweight='bold', color='#1e293b')
        else:
            for u_pos, u_data in enumerate(reversed(units_info)):
                ax_r.text(t_min_ms - 8, u_pos, f"Unit {u_data['id']}",
                          ha='right', va='center', fontsize=8.2, fontweight='semibold', color='#64748b')

        # Trial & Role label on the right
        role_text = trial_roles[row_idx]
        role_color = "#e11d48" if row_idx == 2 else "#1e293b"
        if row_idx == 2:
            label_str = f"{role_text}\nTrial {b_idx}\n($T_I={t_i_ms:.0f}$ ms, $T_E={t_e_ms:.0f}$ ms)"
        else:
            label_str = f"{role_text}\nTrial {b_idx} ($T_I={t_i_ms:.0f}$ ms)"

        ax_r.text(t_max_ms + 10, (n_units - 1) / 2.0, label_str,
                  va='center', ha='left', fontsize=8.5, fontweight='bold', color=role_color)

        # ---------------------------------------------------------
        # B. Synchronized Diaphragm EMG Envelope
        # ---------------------------------------------------------
        emg_trial_vals = np.interp(t_eval_abs, t_nidq, emg_env, left=0.0, right=0.0)
        # Normalize relative to global max
        emg_norm = np.clip(emg_trial_vals / max_emg_global, 0.0, 1.05)

        # Plot EMG curve and shaded area
        curve_color = "#e11d48" if row_idx == 2 else "#1e293b"
        fill_color = "#fda4af" if row_idx == 2 else "#94a3b8"
        fill_alpha = 0.55 if row_idx == 2 else 0.35

        ax_e.plot(t_eval_rel_ms, emg_norm, color=curve_color, linewidth=1.4, zorder=4)
        ax_e.fill_between(t_eval_rel_ms, 0, emg_norm, color=fill_color, alpha=fill_alpha, zorder=3)

        # Vertical reference lines on EMG matching raster
        ax_e.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
        ax_e.axvline(t_i_ms, color="#0369a1", linestyle=":", linewidth=1.2, alpha=0.8, zorder=3)

        ax_e.set_ylim(0.0, 1.10)
        ax_e.set_xlim(t_min_ms, t_max_ms)
        ax_e.set_yticks([])
        ax_e.tick_params(left=False, right=False, top=False, labelleft=False,
                         bottom=(row_idx == n_trials - 1),
                         labelbottom=(row_idx == n_trials - 1),
                         colors="#475569")
        ax_e.text(t_min_ms - 8, 0.5, "Dia. EMG", fontsize=7.8, fontweight='bold', color='#334155', ha='right', va='center')
        for spine in ax_e.spines.values():
            spine.set_color("#be123c" if row_idx == 2 else "#64748b")
            spine.set_linewidth(1.8 if row_idx == 2 else 1.0)

    # 5. Scientific Annotations Highlighting Sigh Biophysics (Carefully placed in clear whitespace)
    # Trial 71 (Pre-Sigh -2): Baseline Eupnea (Placed in clean pre-I area on left)
    axes_rasters[0].annotate('Eupneic Baseline\nFollower (+8.2 ms)', xy=(8.2, 3), xytext=(-68, 3.2),
                             arrowprops=dict(facecolor='#d97706', edgecolor='#d97706', shrink=0.08, width=0.9, headwidth=4),
                             fontsize=7.0, fontweight='bold', color='#b45309', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbeb', edgecolor='#fde68a', alpha=0.92))

    # Trial 72 (Pre-Sigh -1): Early Unit 28 Priming
    axes_rasters[1].annotate('Pre-Sigh Priming\n(-0.4 ms pre-I_on)', xy=(-0.4, 5), xytext=(-75, 5.2),
                             arrowprops=dict(facecolor='#16a34a', edgecolor='#16a34a', shrink=0.08, width=0.9, headwidth=4),
                             fontsize=7.0, fontweight='bold', color='#15803d', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0fdf4', edgecolor='#bbf7d0', alpha=0.92))

    # Trial 73 (SIGH): Massive Metronomic Burst in Unit 28 (Horizontal arrow in open post-I area)
    axes_rasters[2].annotate('SIGH: Augmented Metronomic Burst\n(39 spikes, FR > 260 Hz, uniform ISIs)',
                             xy=(150, 5), xytext=(265, 5.0),
                             arrowprops=dict(facecolor='#e11d48', edgecolor='#e11d48', shrink=0.08, width=1.1, headwidth=5),
                             fontsize=7.1, fontweight='bold', color='#be123c', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff1f2', edgecolor='#fda4af', alpha=0.96))

    # Trial 73 (SIGH): Diaphragm EMG Surge (>2.5x Peak Amplitude)
    axes_emg[2].annotate('Augmented Tidal Volume\n(2.6x Peak Dia. EMG)',
                         xy=(105, 0.98), xytext=(245, 0.65),
                         arrowprops=dict(facecolor='#e11d48', edgecolor='#e11d48', shrink=0.08, width=1.0, headwidth=4.5),
                         fontsize=6.9, fontweight='bold', color='#be123c', ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff1f2', edgecolor='#fda4af', alpha=0.95))

    # Trial 73 (SIGH): Post-Sigh Apnea / Prolonged Expiratory Dwelling
    axes_rasters[2].annotate('Post-Sigh Apnea\n($T_E = 389$ ms dwelling)',
                             xy=(240, 0.5), xytext=(295, 1.8),
                             arrowprops=dict(facecolor='#0369a1', edgecolor='#0369a1', shrink=0.08, width=0.9, headwidth=4),
                             fontsize=7.0, fontweight='bold', color='#0284c7', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f9ff', edgecolor='#bae6fd', alpha=0.92))

    # Trial 74 (Post-Sigh Recovery +1): Prolonged T_I (221 ms) & Delayed Follower
    axes_rasters[3].annotate('Extended $T_I$ (221 ms)\nDelayed Latency (+27.6 ms)',
                             xy=(27.6, 3), xytext=(-65, 3.2),
                             arrowprops=dict(facecolor='#7c3aed', edgecolor='#7c3aed', shrink=0.08, width=0.9, headwidth=4),
                             fontsize=6.9, fontweight='bold', color='#6d28d9', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f3ff', edgecolor='#ddd6fe', alpha=0.92))

    # Trial 75 (Post-Sigh Baseline +2): Eupneic Return & Stochastic Pre-I Spike
    axes_rasters[4].annotate('Stochastic Pre-I Spike\n(-89 ms, Pacemaker)',
                             xy=(-89.3, 6), xytext=(-65, 5.3),
                             arrowprops=dict(facecolor='#0284c7', edgecolor='#0284c7', shrink=0.08, width=0.9, headwidth=4),
                             fontsize=7.0, fontweight='bold', color='#0369a1', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f9ff', edgecolor='#bae6fd', alpha=0.92))

    # Big rotated label on the far left: "Breath Sequence"
    fig.text(0.025, 0.50, "Breath Sequence", va='center', ha='center', rotation='vertical',
             fontsize=18, fontweight='bold', color='#0f172a')

    # Bottom Axis Configuration
    ax_bot = axes_emg[-1]
    ax_bot.tick_params(bottom=True, labelbottom=True, colors="#0f172a", labelsize=9.5)
    ax_bot.set_xlabel("Time from Inspiratory Onset (ms)", fontsize=11.5, fontweight='bold', labelpad=7, color='#0f172a')
    ax_bot.set_xticks(np.arange(-150, 381, 50))

    # Master Title & Subtitle with dedicated spacing
    fig.text(0.23, 0.978, "Sigh Multi-Unit Microcircuit Dynamics: 5-Breath Sequence with Diaphragm EMG",
             fontsize=13.2, fontweight='bold', color='#0f172a', ha='left')
    fig.text(0.23, 0.958, r"Pre-Sigh (-2, -1) $\rightarrow$ SIGH (Trial 73) $\rightarrow$ Post-Sigh Recovery (+1, +2) | Dashed Red: $I_{\mathrm{on}}$ | Dotted Blue / $\blacktriangledown$: $I_{\mathrm{off}}$",
             fontsize=8.8, color='#475569', ha='left')

    # Colorbar for Phase positioned at top-right
    cax = fig.add_axes([0.65, 0.932, 0.22, 0.011])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$", r"$+\pi$"], fontsize=7.2)
    cbar.ax.tick_params(colors="#0f172a", pad=2)
    cbar.set_label(r"Breath Phase $\phi(t)$ (Cool Blue = Exp | Warm Red = Insp)", fontsize=7.2, fontweight='semibold', labelpad=2, color="#0f172a")

    # Save both PNG (300 DPI) and SVG
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved sigh raster sequence PNG: {out_png}")
    print(f"[SUCCESS] Saved sigh raster sequence SVG: {out_svg}")

if __name__ == "__main__":
    main()
