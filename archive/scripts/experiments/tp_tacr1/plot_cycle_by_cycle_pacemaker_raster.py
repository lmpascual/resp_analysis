#!/usr/bin/env python3
"""
Generate Cycle-by-Cycle Exemplary Neuron Multi-Unit Raster Plots & Diaphragm EMG Traces
with Gradated Phase Colormap (Matching the Breath-Aligned PSTH / Phase Rasters).

Features:
1. 6 vertically stacked trial panels (Breaths 18 to 23).
2. Gradated phase background using the exact PSTH phase colormap:
   - Expiration (-pi to 0): Cool blue (#93c5fd -> #dbeafe) to White (#ffffff at I_on)
   - Inspiration (0 to +pi): White (#ffffff at I_on) -> Warm peach (#fed7aa) -> Warm coral/red (#f87171 at I_off)
3. Synchronized Diaphragm EMG Envelope trace directly beneath the rasters for each trial.
4. Exemplary neurons demonstrating:
   - Candidate Pacemakers (Unit 13, Unit 28) with stochastic pre-I firing & lead/follow switching
   - Vanguard Burster (Unit 83)
   - Canonical Follower (Unit 53) with zero expiratory spikes and rigid post-I_on latency
   - Mid-Inspiratory Follower (Unit 33)
   - BötC Expiratory (Unit 15) with sharp quenching at I_on
   - cVRG Post-I / Expiratory (Unit 27)
5. Inverted black triangle marker at I_off on each trial.
6. Exports both high-resolution PNG (300 DPI) and scalable vector SVG.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
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
        t_ref_next = t_next_on if t_next_on > t_off else t_off + 0.25
        phi[mask_post] = -np.pi + (t_eval[mask_post] - t_off) / (t_ref_next - t_off + 1e-9) * np.pi
        phi[mask_post] = np.clip(phi[mask_post], -np.pi, 0.0)

    return phi

def main():
    out_dir = PROJECT_ROOT / "presentation_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cycle_by_cycle_pacemaker_lead_follow_raster.png"
    out_svg = out_dir / "cycle_by_cycle_pacemaker_lead_follow_raster.svg"

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

    # 2. Units Information
    units_info = [
        {"id": 13, "label": "Unit 13 (preBötC Pacemaker: Stochastic Pre-I)", "depth": 945},
        {"id": 28, "label": "Unit 28 (preBötC Pacemaker: Lead/Follow Switching)", "depth": 900},
        {"id": 83, "label": "Unit 83 (preBötC Vanguard Burster)", "depth": 810},
        {"id": 53, "label": "Unit 53 (preBötC Canonical Follower: +16 ms)", "depth": 892},
        {"id": 33, "label": "Unit 33 (preBötC Mid-Inspiratory Follower)", "depth": 570},
        {"id": 15, "label": "Unit 15 (BötC Expiratory: Sharp I_on Inhibition)", "depth": 480},
        {"id": 27, "label": "Unit 27 (cVRG Post-I / Expiratory Disinhibition)", "depth": 1230},
    ]
    n_units = len(units_info)

    # 6 Exemplary Consecutive Breaths
    trial_indices = [18, 19, 20, 21, 22, 23]
    n_trials = len(trial_indices)

    t_min_ms = -150.0
    t_max_ms = 350.0
    n_grid_pts = 500
    t_eval_rel_ms = np.linspace(t_min_ms, t_max_ms, n_grid_pts)
    t_eval_rel_s = t_eval_rel_ms / 1000.0

    # 3. Gradated Colormap (Matching PSTH Plots)
    # Cool Blue (-pi) -> Soft Light Blue (-pi/2) -> White (0, I_on) -> Soft Peach (+pi/2) -> Warm Red (+pi, I_off)
    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_gradated",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    # 4. Create Master Figure with GridSpec (6 Trials x 2 Sub-Rows: Raster + EMG)
    fig = plt.figure(figsize=(11.5, 14.5), dpi=300)
    
    # Master Grid: 6 trial blocks
    gs_master = gridspec.GridSpec(n_trials, 1, figure=fig, hspace=0.22,
                                  left=0.23, right=0.91, top=0.94, bottom=0.06)

    axes_rasters = []
    axes_emg = []

    for row_idx, b_idx in enumerate(trial_indices):
        t_on = onsets[b_idx]
        t_off = offsets[b_idx]
        t_prev_off = offsets[b_idx - 1] if b_idx > 0 else t_on - 0.25
        t_next_on = onsets[b_idx + 1] if b_idx + 1 < len(onsets) else t_off + 0.25
        t_i_ms = (t_off - t_on) * 1000.0

        # Sub-Grid for this trial: Raster (height 78%) and EMG (height 22%), zero vertical gap
        gs_trial = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[row_idx],
                                                    height_ratios=[3.4, 1.0], hspace=0.0)
        
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

            for spk_t in in_win:
                ax_r.plot([spk_t, spk_t], [u_pos - 0.35, u_pos + 0.35],
                          color="#0f172a", linewidth=1.5, zorder=5)

        ax_r.set_ylim(-0.5, n_units - 0.5)
        ax_r.set_xlim(t_min_ms, t_max_ms)
        ax_r.set_yticks([])  # Remove numeric y-ticks completely
        ax_r.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_r.spines.values():
            spine.set_color("#64748b")
            spine.set_linewidth(1.0)

        # Unit labels along the left
        if row_idx == 0:
            for u_pos, u_data in enumerate(reversed(units_info)):
                ax_r.text(t_min_ms - 10, u_pos, f"{u_data['label']}",
                          ha='right', va='center', fontsize=8.2, fontweight='bold', color='#1e293b')
        else:
            for u_pos, u_data in enumerate(reversed(units_info)):
                ax_r.text(t_min_ms - 10, u_pos, f"Unit {u_data['id']}",
                          ha='right', va='center', fontsize=8.2, fontweight='semibold', color='#64748b')

        # Trial label on the right
        ax_r.text(t_max_ms + 8, (n_units - 1) / 2.0, f"Trial {row_idx + 1}\n($T_I={t_i_ms:.0f}$ ms)",
                  va='center', ha='left', fontsize=8.8, fontweight='bold', color='#1e293b')

        # ---------------------------------------------------------
        # B. Synchronized Diaphragm EMG Envelope
        # ---------------------------------------------------------
        emg_trial_vals = np.interp(t_eval_abs, t_nidq, emg_env, left=0.0, right=0.0)
        # Normalize EMG relative to max in this window for crisp contrast
        emg_norm = emg_trial_vals / (np.max(emg_env) * 0.85 + 1e-12)
        emg_norm = np.clip(emg_norm, 0.0, 1.0)

        # Plot EMG curve and shaded area
        ax_e.plot(t_eval_rel_ms, emg_norm, color="#1e293b", linewidth=1.3, zorder=4)
        ax_e.fill_between(t_eval_rel_ms, 0, emg_norm, color="#94a3b8", alpha=0.35, zorder=3)

        # Vertical reference lines on EMG matching raster
        ax_e.axvline(0, color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
        ax_e.axvline(t_i_ms, color="#0369a1", linestyle=":", linewidth=1.2, alpha=0.8, zorder=3)

        ax_e.set_ylim(0.0, 1.05)
        ax_e.set_xlim(t_min_ms, t_max_ms)
        ax_e.set_yticks([])
        ax_e.tick_params(left=False, right=False, top=False, labelleft=False,
                         bottom=(row_idx == n_trials - 1),
                         labelbottom=(row_idx == n_trials - 1),
                         colors="#475569")
        ax_e.text(t_min_ms - 10, 0.5, "Dia. EMG", fontsize=7.8, fontweight='bold', color='#334155', ha='right', va='center')
        for spine in ax_e.spines.values():
            spine.set_color("#64748b")
            spine.set_linewidth(1.0)

    # 5. Callout Annotations Highlighting Biophysical Hallmarks
    # Trial 1: Unit 13 stochastic pre-I spike
    axes_rasters[0].annotate('Stochastic Pre-I\nSpike (-86 ms)', xy=(-86.5, 6), xytext=(-55, 5.2),
                             arrowprops=dict(facecolor='#0284c7', edgecolor='#0284c7', shrink=0.08, width=1.0, headwidth=4),
                             fontsize=7.2, fontweight='bold', color='#0369a1', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f9ff', edgecolor='#bae6fd', alpha=0.9))

    # Trial 1: Unit 53 Follower latency
    axes_rasters[0].annotate('Follower Latency\n(+19 ms)', xy=(19.1, 3), xytext=(72, 2.2),
                             arrowprops=dict(facecolor='#d97706', edgecolor='#d97706', shrink=0.08, width=1.0, headwidth=4),
                             fontsize=7.2, fontweight='bold', color='#b45309', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbeb', edgecolor='#fde68a', alpha=0.9))

    # Trial 3: Unit 13 isolated ectopic expiratory spike
    axes_rasters[2].annotate('Stray Expiratory\nPacemaker Spike (-132 ms)', xy=(-131.9, 6), xytext=(-80, 6.1),
                             arrowprops=dict(facecolor='#0284c7', edgecolor='#0284c7', shrink=0.08, width=1.0, headwidth=4),
                             fontsize=7.2, fontweight='bold', color='#0369a1', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f9ff', edgecolor='#bae6fd', alpha=0.9))

    # Trial 3: Unit 28 switches to follower
    axes_rasters[2].annotate('Switches to Follower\n(+5.4 ms post-I_on)', xy=(5.4, 5), xytext=(65, 5.3),
                             arrowprops=dict(facecolor='#7c3aed', edgecolor='#7c3aed', shrink=0.08, width=1.0, headwidth=4),
                             fontsize=7.2, fontweight='bold', color='#6d28d9', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f3ff', edgecolor='#ddd6fe', alpha=0.9))

    # Trial 5: Unit 28 strongly leads network
    axes_rasters[4].annotate('Leads Network\n(-21 ms pre-I_on)', xy=(-20.8, 5), xytext=(-65, 4.3),
                             arrowprops=dict(facecolor='#16a34a', edgecolor='#16a34a', shrink=0.08, width=1.0, headwidth=4),
                             fontsize=7.2, fontweight='bold', color='#15803d', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0fdf4', edgecolor='#bbf7d0', alpha=0.9))

    # Big rotated label on the far left: "Trial No."
    fig.text(0.025, 0.50, "Trial No.", va='center', ha='center', rotation='vertical',
             fontsize=19, fontweight='bold', color='#0f172a')

    # Bottom Axis Configuration
    ax_bot = axes_emg[-1]
    ax_bot.tick_params(bottom=True, labelbottom=True, colors="#0f172a", labelsize=10)
    ax_bot.set_xlabel("Time from Inspiratory Onset (ms)", fontsize=12, fontweight='bold', labelpad=8, color='#0f172a')
    ax_bot.set_xticks(np.arange(-150, 351, 50))

    # Master Title & Header
    fig.text(0.23, 0.980, "Cycle-by-Cycle Assembly & Lead/Follow Dynamics with Diaphragm EMG",
             fontsize=13.5, fontweight='bold', color='#0f172a', ha='left')
    fig.text(0.23, 0.963, r"Dashed Red: $I_{\mathrm{on}}$ ($t=0$ ms) | Dotted Blue / $\blacktriangledown$: $I_{\mathrm{off}}$ (Phase Boundary)",
             fontsize=9.0, color='#475569', ha='left')

    # Colorbar for Phase
    cax = fig.add_axes([0.68, 0.960, 0.23, 0.011])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$", r"$+\pi$"], fontsize=7.5)
    cbar.ax.tick_params(colors="#0f172a", pad=2)
    cbar.set_label(r"Breath Phase $\phi(t)$ (Cool Blue = Exp | Warm Red = Insp)", fontsize=7.6, fontweight='semibold', labelpad=2, color="#0f172a")

    # Save both PNG (300 DPI) and SVG
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_svg, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SUCCESS] Saved raster plot PNG: {out_png}")
    print(f"[SUCCESS] Saved raster plot SVG: {out_svg}")

if __name__ == "__main__":
    main()
