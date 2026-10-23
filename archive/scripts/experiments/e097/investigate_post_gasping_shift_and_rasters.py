#!/usr/bin/env python3
"""
Deep-Dive Investigation into Post-Gasping Off-Switch / Post-I Timing Shifts
and Generation of Population Rasters (All Isolated Units vs. Co-Modulated Units)
Across Pre-Gasping, During-Gasping, and Post-Gasping Epochs (e097 Intact Window).
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processors.proc_emg import process_emg, EMGConfig
from src.pipeline.processors.proc_flow import process_flow_meter
from scripts.experiments.e097.run_intact_window_dual_sorting_and_hmm_pipeline import (
    INTACT_T_START, INTACT_DURATION, load_intact_spikes_and_qc, extract_intact_breaths
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("INVESTIGATE_GASPING_SHIFT")

CANONICAL_6STATE_COLORS = [
    "#10b981", # State 1: Insp Burst (Emerald)
    "#06b6d4", # State 2: Post-I (E1) (Cyan)
    "#f97316", # State 3: Off-Switch (I_off) (Orange)
    "#38bdf8", # State 4: Late Exp (E2) (Sky blue)
    "#a855f7", # State 5: Pre-I Priming (Purple)
    "#64748b"  # State 6: Pause (Slate grey)
]

CANONICAL_6STATE_SHORT = [
    "Insp Burst",
    "Post-I (E1)",
    "Off-Switch (I_off)",
    "Late Exp (E2)",
    "Pre-I Priming",
    "Pause / Quiescence"
]

def analyze_breath_shift_mechanisms(breath_data: Dict[str, Any], state_seq: np.ndarray, dt: float = 0.010):
    """
    Quantifies the flow waveform changes, EMG burst decay, and neural state timing
    across Pre-Gasping, During-Gasping, and Post-Gasping epochs.
    """
    logger.info("Analyzing breath flow waveform changes and neural off-switch alignment...")
    fs = breath_data["fs"]
    flow_v = breath_data["flow_v"]
    emg_v = breath_data["emg_v"]
    emg_env = breath_data["emg_env"]
    t_vec = breath_data["t_vec"]
    
    valid_onsets = breath_data["valid_onsets"]
    valid_offsets = breath_data["valid_offsets"]
    flow_zcs = breath_data["flow_zcs"]
    stages = breath_data["breath_stages"]
    n_breaths = len(valid_onsets)
    
    # Define 3 broad physiological epochs:
    # 1. Pre-Gasping: Baseline Eupnea + Anoxia 1 (stages: win1_baseline, win2_anox1)
    # 2. During-Gasping: Apnea Nadir + Early Gasping + Late Gasping (stages: win3_anox2, win4_gasp_early, win5_gasp_late)
    # 3. Post-Gasping: Active Expiration Shift + Recovery (stages: win6_flow_shift, win7_recovery)
    
    epoch_groups = {
        "Pre-Gasping": ["win1_baseline", "win2_anox1"],
        "During-Gasping": ["win4_gasp_early", "win5_gasp_late"],
        "Post-Gasping": ["win6_flow_shift", "win7_recovery"]
    }
    
    epoch_metrics = {name: [] for name in epoch_groups}
    
    # Array to track per-breath timing shifts
    ti_flow_list = []
    ti_emg_list = []
    delta_emg_flow_list = [] # t_flow_zc - t_emg_off (in ms)
    peak_flow_to_zc_list = [] # time from peak negative flow to zero crossing
    flow_decay_slope_list = [] # dFlow/dt at zero crossing
    state1_end_to_flow_zc_list = [] # t_flow_zc - t_state1_end (in ms)
    state2_peak_to_flow_zc_list = []
    state3_peak_to_flow_zc_list = []
    
    # Waveform snippets aligned to flow zero-crossing [-200 ms, +300 ms]
    snippet_win_pts = int(0.500 * fs)
    pre_pts = int(0.200 * fs)
    post_pts = int(0.300 * fs)
    t_snip = np.linspace(-200, 300, pre_pts + post_pts)
    
    flow_snippets_by_epoch = {name: [] for name in epoch_groups}
    emg_snippets_by_epoch = {name: [] for name in epoch_groups}
    
    # State probabilities aligned to flow zero crossing [-200 ms, +300 ms] (50 bins of 10 ms)
    n_bins_snip = int(0.500 / dt)
    pre_bins = int(0.200 / dt)
    state_prob_snippets_by_epoch = {name: np.zeros((6, n_bins_snip)) for name in epoch_groups}
    epoch_counts = {name: 0 for name in epoch_groups}
    
    for b_idx in range(n_breaths):
        stg = stages[b_idx]
        cur_epoch = None
        for ep_name, member_stages in epoch_groups.items():
            if stg in member_stages:
                cur_epoch = ep_name
                break
        if cur_epoch is None:
            continue
            
        t_on = valid_onsets[b_idx]
        t_emg_off = valid_offsets[b_idx]
        t_flow_zc = flow_zcs[b_idx]
        
        # Inspiratory durations
        ti_flow = (t_flow_zc - t_on) * 1000.0 # ms
        ti_emg = (t_emg_off - t_on) * 1000.0  # ms
        delta_emg_flow = (t_flow_zc - t_emg_off) * 1000.0 # ms (+ means flow zc is later than EMG off)
        
        # Analyze flow waveform in the inspiratory phase
        idx_on = int(t_on * fs)
        idx_zc = int(t_flow_zc * fs)
        if 0 <= idx_on < idx_zc < len(flow_v):
            insp_flow = flow_v[idx_on:idx_zc]
            min_flow_rel_idx = np.argmin(insp_flow)
            t_peak_flow = t_on + (min_flow_rel_idx / fs)
            time_peak_to_zc = (t_flow_zc - t_peak_flow) * 1000.0 # ms
            
            # Flow decay slope at zero crossing (over 10 ms window)
            slope_pts = int(0.010 * fs)
            if idx_zc - slope_pts >= 0 and idx_zc + slope_pts < len(flow_v):
                slope = (flow_v[idx_zc + slope_pts] - flow_v[idx_zc - slope_pts]) / 0.020 # V/s
            else:
                slope = 0.0
        else:
            time_peak_to_zc = np.nan
            slope = np.nan
            
        # Extract aligned flow and EMG snippets around flow zero-crossing
        i0 = idx_zc - pre_pts
        i1 = idx_zc + post_pts
        if i0 >= 0 and i1 <= len(flow_v):
            flow_snippets_by_epoch[cur_epoch].append(flow_v[i0:i1])
            emg_snippets_by_epoch[cur_epoch].append(emg_env[i0:i1])
            
        # Extract state sequence around flow zero-crossing
        b_zc = int(t_flow_zc / dt)
        b0 = b_zc - pre_bins
        b1 = b_zc + (n_bins_snip - pre_bins)
        if b0 >= 0 and b1 <= len(state_seq):
            seq_snip = state_seq[b0:b1]
            for s in range(6):
                state_prob_snippets_by_epoch[cur_epoch][s, :] += (seq_snip == s).astype(np.float32)
            epoch_counts[cur_epoch] += 1
            
            # State 1 (Insp Burst) end time relative to flow zc
            s1_bins = np.where(seq_snip == 0)[0]
            if len(s1_bins) > 0:
                s1_last = s1_bins[-1]
                t_s1_end_rel = (s1_last - pre_bins) * (dt * 1000.0) # ms
            else:
                t_s1_end_rel = np.nan
            state1_end_to_flow_zc_list.append(t_s1_end_rel)
            
            # State 2 (Post-I) peak time relative to flow zc
            s2_bins = np.where(seq_snip == 1)[0]
            if len(s2_bins) > 0:
                s2_peak = s2_bins[len(s2_bins)//2]
                t_s2_peak_rel = (s2_peak - pre_bins) * (dt * 1000.0)
            else:
                t_s2_peak_rel = np.nan
            state2_peak_to_flow_zc_list.append(t_s2_peak_rel)
            
            # State 3 (Off-Switch Quench) peak time relative to flow zc
            s3_bins = np.where(seq_snip == 2)[0]
            if len(s3_bins) > 0:
                s3_peak = s3_bins[len(s3_bins)//2]
                t_s3_peak_rel = (s3_peak - pre_bins) * (dt * 1000.0)
            else:
                t_s3_peak_rel = np.nan
            state3_peak_to_flow_zc_list.append(t_s3_peak_rel)

        epoch_metrics[cur_epoch].append({
            "b_idx": b_idx,
            "t_on": t_on,
            "ti_flow_ms": ti_flow,
            "ti_emg_ms": ti_emg,
            "delta_emg_flow_ms": delta_emg_flow,
            "time_peak_to_zc_ms": time_peak_to_zc,
            "flow_slope": slope
        })

    # Normalize state probabilities
    for ep_name in epoch_groups:
        cnt = max(1, epoch_counts[ep_name])
        state_prob_snippets_by_epoch[ep_name] /= cnt

    # Compute summary statistics per epoch
    summary_by_epoch = {}
    for ep_name in epoch_groups:
        arrs = epoch_metrics[ep_name]
        df_ep = pd.DataFrame(arrs)
        summary_by_epoch[ep_name] = {
            "n_breaths": len(df_ep),
            "mean_ti_flow_ms": float(df_ep["ti_flow_ms"].mean()),
            "std_ti_flow_ms": float(df_ep["ti_flow_ms"].std()),
            "mean_ti_emg_ms": float(df_ep["ti_emg_ms"].mean()),
            "std_ti_emg_ms": float(df_ep["ti_emg_ms"].std()),
            "mean_delta_emg_flow_ms": float(df_ep["delta_emg_flow_ms"].mean()),
            "std_delta_emg_flow_ms": float(df_ep["delta_emg_flow_ms"].std()),
            "mean_peak_to_zc_ms": float(df_ep["time_peak_to_zc_ms"].dropna().mean()),
            "mean_flow_slope": float(df_ep["flow_slope"].dropna().mean()),
            "mean_flow_waveform": np.mean(flow_snippets_by_epoch[ep_name], axis=0).tolist() if flow_snippets_by_epoch[ep_name] else [],
            "sem_flow_waveform": (np.std(flow_snippets_by_epoch[ep_name], axis=0) / np.sqrt(len(flow_snippets_by_epoch[ep_name]))).tolist() if flow_snippets_by_epoch[ep_name] else [],
            "mean_emg_waveform": np.mean(emg_snippets_by_epoch[ep_name], axis=0).tolist() if emg_snippets_by_epoch[ep_name] else [],
            "state_probs": state_prob_snippets_by_epoch[ep_name].tolist()
        }
        logger.info(f"Epoch {ep_name} ({len(df_ep)} breaths):")
        logger.info(f"  T_I Flow: {summary_by_epoch[ep_name]['mean_ti_flow_ms']:.1f} ms | T_I EMG: {summary_by_epoch[ep_name]['mean_ti_emg_ms']:.1f} ms")
        logger.info(f"  Delta(Flow_ZC - EMG_Off): {summary_by_epoch[ep_name]['mean_delta_emg_flow_ms']:+.1f} ± {summary_by_epoch[ep_name]['std_delta_emg_flow_ms']:.1f} ms")
        logger.info(f"  Time Peak Flow to ZC: {summary_by_epoch[ep_name]['mean_peak_to_zc_ms']:.1f} ms")

    return {
        "summary_by_epoch": summary_by_epoch,
        "t_snip_ms": t_snip.tolist(),
        "t_state_ms": np.linspace(-200, 300, n_bins_snip).tolist()
    }


def find_top_comodulated_units(spike_mat: np.ndarray, state_seq: np.ndarray, depths: np.ndarray, dt: float = 0.010, top_k: int = 25) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Identifies the single units exhibiting the largest coordinated rate change (|z| > 2.0)
    specifically across the Inspiratory Burst -> Post-I / Off-Switch transitions.
    """
    logger.info("Computing unit-by-unit co-modulation strength at inspiratory off-switch transitions...")
    total_units = spike_mat.shape[1]
    box_pts = int(0.050 / dt)
    smoothed = np.zeros_like(spike_mat, dtype=np.float32)
    for u in range(total_units):
        smoothed[:, u] = np.convolve(spike_mat[:, u], np.ones(box_pts)/box_pts, mode='same')
        
    unit_stds = np.std(smoothed, axis=0) + 1e-6
    
    # Find all transitions from State 1 (Insp Burst, idx 0) to State 2 (Post-I, idx 1) or State 3 (Off-Switch, idx 2)
    trans_pts = []
    for t in range(1, len(state_seq)):
        if state_seq[t-1] == 0 and state_seq[t] in (1, 2):
            trans_pts.append(t)
            
    w_bins = int(0.050 / dt)
    z_diffs = []
    for t in trans_pts:
        if t < w_bins or t > len(state_seq) - w_bins:
            continue
        pre = smoothed[t - w_bins : t, :]
        post = smoothed[t : t + w_bins, :]
        z_trans = (np.mean(post, axis=0) - np.mean(pre, axis=0)) / unit_stds
        z_diffs.append(z_trans)
        
    z_arr = np.array(z_diffs) # (n_trans, n_units)
    mean_abs_z = np.mean(np.abs(z_arr), axis=0)
    frac_comod = np.mean(np.abs(z_arr) > 2.0, axis=0)
    
    sort_idx = np.argsort(mean_abs_z)[::-1]
    top_comod_indices = sort_idx[:top_k]
    
    df_comod = pd.DataFrame({
        "unit_idx": np.arange(total_units),
        "depth_um": depths,
        "mean_abs_z": mean_abs_z,
        "frac_transitions_comod": frac_comod
    }).sort_values(by="mean_abs_z", ascending=False).reset_index(drop=True)
    
    logger.info(f"Identified {top_k} top co-modulated units (mean |z| = {df_comod['mean_abs_z'].head(top_k).mean():.2f}).")
    return top_comod_indices, df_comod


def generate_population_rasters(
    spk_times: np.ndarray,
    spk_clusters: np.ndarray,
    df_qc: pd.DataFrame,
    nidq_file: Path,
    state_seq: np.ndarray,
    comod_unit_indices: np.ndarray,
    out_dir: Path
):
    """
    Generates high-resolution raster plots for:
    1. All isolated units
    2. Co-modulated units
    across Pre-Gasping, During-Gasping, and Post-Gasping epochs.
    """
    logger.info("Generating population rasters (All Units and Co-Modulated Units)...")
    fs = 10000.0
    n_ch = 9
    dt = 0.010
    
    # 3 Distinct Physiological Windows (20 seconds each):
    # 1. Pre-Gasping: t = 1840.0s to 1860.0s (Late Anoxic Challenge / Apnea onset)
    # 2. During-Gasping: t = 1910.0s to 1930.0s (Large Autoresuscitation Gasps)
    # 3. Post-Gasping: t = 2055.0s to 2075.0s (Active Expiration Shift / Tachypnea)
    windows = [
        {"id": "pre_gasping", "title": "1. Pre-Gasping (Late Anoxia & Apnea Transition)", "t0": 1840.0, "dur": 20.0},
        {"id": "during_gasping", "title": "2. During Gasping (Autoresuscitation Surge)", "t0": 1910.0, "dur": 20.0},
        {"id": "post_gasping", "title": "3. Post-Gasping (Active Expiration Shift)", "t0": 2055.0, "dur": 20.0}
    ]
    
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values
    n_units = len(cids)
    
    # Map cluster_id to unit row index (ordered by probe depth)
    depth_order = np.argsort(depths)
    unit_to_row = {cids[u_idx]: row for row, u_idx in enumerate(depth_order)}
    
    cmap_hmm = ListedColormap(CANONICAL_6STATE_COLORS)
    norm_hmm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=6)
    
    for raster_type in ["all_units", "comod_units"]:
        if raster_type == "all_units":
            target_indices = np.arange(n_units)
            target_title = "All Isolated QC Single Units (N = 120)"
            fname_prefix = "intact_population_raster_all_units"
        else:
            target_indices = comod_unit_indices
            target_title = f"Top Co-Modulated Driver Units (N = {len(comod_unit_indices)})"
            fname_prefix = "intact_population_raster_comod_units"
            
        fig, axes = plt.subplots(3, 1, figsize=(18, 14), dpi=300, sharex=False)
        plt.subplots_adjust(hspace=0.35)
        
        for ax_idx, win in enumerate(windows):
            ax_main = axes[ax_idx]
            t0_abs = win["t0"]
            dur = win["dur"]
            t1_abs = t0_abs + dur
            t0_rel = t0_abs - INTACT_T_START
            t1_rel = t1_abs - INTACT_T_START
            
            # Load NIDQ Flow & EMG for this window
            s0 = int(t0_abs * fs)
            s1 = int(t1_abs * fs)
            slice_nidq = raw[s0 * n_ch : s1 * n_ch].reshape(-1, n_ch)
            emg = slice_nidq[:, 0].astype(np.float32) * (10.0 / 32768.0)
            flow = slice_nidq[:, 5].astype(np.float32) * (10.0 / 32768.0)
            t_nidq = np.linspace(0, dur, len(flow))
            
            # EMG envelope
            emg_hp = emg - np.mean(emg)
            emg_rect = np.abs(emg_hp)
            box_emg = int(0.040 * fs)
            emg_env = np.convolve(emg_rect, np.ones(box_emg)/box_emg, mode='same')
            
            # Plot Spikes for target units
            ax_spk = ax_main
            for plot_row, u_idx in enumerate(target_indices):
                cid = cids[u_idx]
                spks = spk_times[spk_clusters == cid]
                spks_win = spks[(spks >= t0_rel) & (spks <= t1_rel)] - t0_rel
                if len(spks_win) > 0:
                    ax_spk.scatter(spks_win, np.full_like(spks_win, plot_row), s=3.5, color='#0f172a', alpha=0.75, marker='|', linewidths=0.7)
                    
            ax_spk.set_ylabel(f"Unit Rank (By Depth)", fontsize=10, fontweight='semibold')
            ax_spk.set_ylim([-1, len(target_indices)])
            ax_spk.set_xlim([0, dur])
            ax_spk.set_title(f"{win['title']} &mdash; {target_title}", fontsize=11, fontweight='bold', pad=8)
            
            # Overlay Flow and Integrated EMG on twin axes
            ax_flow = ax_spk.twinx()
            ax_flow.plot(t_nidq, flow, color='#2563eb', linewidth=1.2, alpha=0.85, label='Airflow (V)')
            ax_flow.plot(t_nidq, emg_env * 8.0 - 0.2, color='#ef4444', linewidth=1.0, alpha=0.75, label='Int. Diaphragm EMG')
            ax_flow.axhline(0, color='#94a3b8', linestyle=':', linewidth=0.8, alpha=0.5)
            ax_flow.set_ylabel("Flow (V) & EMG", fontsize=10, color='#1e3a8a', fontweight='semibold')
            ax_flow.set_ylim([-0.5, 0.4])
            
            # Underneath: HMM 6-State Color Bar
            b0 = int(t0_rel / dt)
            b1 = int(t1_rel / dt)
            if b0 >= 0 and b1 <= len(state_seq):
                seq_win = state_seq[b0:b1].reshape(1, -1)
                ax_hmm = ax_spk.inset_axes([0, -0.14, 1.0, 0.08])
                ax_hmm.imshow(seq_win, aspect='auto', cmap=cmap_hmm, norm=norm_hmm, extent=[0, dur, 0, 1], interpolation='nearest')
                ax_hmm.set_yticks([])
                ax_hmm.set_xticks([])
                ax_hmm.set_xlabel("Time (s) + HMM 6-State Track", fontsize=9, color='#475569')

        out_png = out_dir / f"{fname_prefix}.png"
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Successfully generated {raster_type} publication figure at {out_png}")
        
        # Copy to artifacts directory
        artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\2f5e4e38-0e84-4946-871a-26957179bf52")
        if artifact_dir.exists():
            shutil.copy2(out_png, artifact_dir / f"{fname_prefix}.png")


def generate_shift_investigation_figure(shift_data: Dict[str, Any], out_path: Path):
    """
    Generates a 300 DPI 4-panel publication figure explicitly testing the hypothesis
    of the post-gasping off-switch shift.
    """
    logger.info(f"Generating shift investigation publication figure at {out_path}...")
    summ = shift_data["summary_by_epoch"]
    t_snip = np.array(shift_data["t_snip_ms"])
    t_state = np.array(shift_data["t_state_ms"])
    
    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)
    
    epoch_colors = {
        "Pre-Gasping": "#10b981",
        "During-Gasping": "#f59e0b",
        "Post-Gasping": "#8b5cf6"
    }

    # Panel A: Mean Flow Waveform Aligned to Flow Zero Crossing
    ax_a = fig.add_subplot(gs[0, 0])
    for ep_name in ["Pre-Gasping", "During-Gasping", "Post-Gasping"]:
        m_flow = np.array(summ[ep_name]["mean_flow_waveform"])
        s_flow = np.array(summ[ep_name]["sem_flow_waveform"])
        col = epoch_colors[ep_name]
        ax_a.plot(t_snip, m_flow, color=col, linewidth=2.2, label=f"{ep_name} (N={summ[ep_name]['n_breaths']})")
        ax_a.fill_between(t_snip, m_flow - s_flow, m_flow + s_flow, color=col, alpha=0.15)
        
    ax_a.axvline(0, color='black', linestyle='--', linewidth=1.2, label=r'Flow Zero-Crossing ($t=0$)')
    ax_a.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax_a.set_title("A. Airflow Waveform Evolution Across Gasping Epochs", fontsize=12, fontweight='bold', pad=10)
    ax_a.set_xlabel("Time from Flow Zero-Crossing (ms)", fontsize=11, fontweight='semibold')
    ax_a.set_ylabel("Airflow (V)", fontsize=11, fontweight='semibold')
    ax_a.set_xlim([-150, 200])
    ax_a.grid(True, linestyle='--', alpha=0.3)
    ax_a.legend(loc='lower right', frameon=True, fontsize=9.5)

    # Panel B: Delta (Flow Zero Crossing - Diaphragm EMG Offset)
    ax_b = fig.add_subplot(gs[0, 1])
    epochs = ["Pre-Gasping", "During-Gasping", "Post-Gasping"]
    deltas = [summ[ep]["mean_delta_emg_flow_ms"] for ep in epochs]
    delta_stds = [summ[ep]["std_delta_emg_flow_ms"] for ep in epochs]
    
    bars = ax_b.bar(epochs, deltas, yerr=delta_stds, capsize=5, color=[epoch_colors[ep] for ep in epochs], edgecolor='#0f172a', width=0.55)
    for b in bars:
        h = b.get_height()
        ax_b.text(b.get_x() + b.get_width()/2., h + 2.0 if h >= 0 else h - 8.0, f"{h:+.1f} ms", ha='center', va='bottom' if h >= 0 else 'top', fontsize=10, fontweight='bold')
        
    ax_b.axhline(0, color='black', linestyle='-', linewidth=1.0)
    ax_b.set_title(r"B. Neuro-Mechanical Delay: $\Delta t = (t_{\rm Flow\_ZC} - t_{\rm EMG\_Off})$", fontsize=12, fontweight='bold', pad=10)
    ax_b.set_ylabel("Delay Time (ms)", fontsize=11, fontweight='semibold')
    ax_b.set_ylim([min(deltas) - 30, max(deltas) + 40])
    ax_b.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Panel C: State 2 (Post-I / E1) Probability Over Time relative to Flow ZC
    ax_c = fig.add_subplot(gs[1, 0])
    for ep_name in ["Pre-Gasping", "During-Gasping", "Post-Gasping"]:
        probs = np.array(summ[ep_name]["state_probs"])
        p_post_i = probs[1, :] # State 2
        col = epoch_colors[ep_name]
        ax_c.plot(t_state, p_post_i * 100.0, color=col, linewidth=2.2, label=ep_name)
        
    ax_c.axvline(0, color='black', linestyle='--', linewidth=1.2, label=r'Flow Zero-Crossing')
    ax_c.set_title("C. Post-Inspiratory State 2 ($E_1$) Time Course", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlabel("Time from Flow Zero-Crossing (ms)", fontsize=11, fontweight='semibold')
    ax_c.set_ylabel("State Occupancy Probability (%)", fontsize=11, fontweight='semibold')
    ax_c.set_xlim([-150, 200])
    ax_c.grid(True, linestyle='--', alpha=0.3)
    ax_c.legend(loc='upper right', frameon=True, fontsize=9.5)

    # Panel D: State 3 (Off-Switch Quench) Probability Over Time relative to Flow ZC
    ax_d = fig.add_subplot(gs[1, 1])
    for ep_name in ["Pre-Gasping", "During-Gasping", "Post-Gasping"]:
        probs = np.array(summ[ep_name]["state_probs"])
        p_quench = probs[2, :] # State 3
        col = epoch_colors[ep_name]
        ax_d.plot(t_state, p_quench * 100.0, color=col, linewidth=2.2, label=ep_name)
        
    ax_d.axvline(0, color='black', linestyle='--', linewidth=1.2, label=r'Flow Zero-Crossing')
    ax_d.set_title("D. Off-Switch Quench State 3 ($I_{\\rm off}$) Time Course", fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xlabel("Time from Flow Zero-Crossing (ms)", fontsize=11, fontweight='semibold')
    ax_d.set_ylabel("State Occupancy Probability (%)", fontsize=11, fontweight='semibold')
    ax_d.set_xlim([-150, 200])
    ax_d.grid(True, linestyle='--', alpha=0.3)
    ax_d.legend(loc='upper right', frameon=True, fontsize=9.5)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Successfully saved shift investigation figure to {out_path}")
    
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\2f5e4e38-0e84-4946-871a-26957179bf52")
    if artifact_dir.exists():
        shutil.copy2(out_path, artifact_dir / "investigation_post_gasping_offswitch_shift.png")


def main():
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    ks4_dir = out_base / "kilosort4"
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    
    # 1. Load Spikes and Breaths
    spk_times, spk_clusters, depths, df_qc = load_intact_spikes_and_qc(ks4_dir, max_units=120)
    breath_data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)
    
    # 2. Load 6-State Sequence from KS4 results
    ks4_json = out_base / "kilosort4_intact_6state_analysis_results.json"
    if not ks4_json.exists():
        logger.error(f"KS4 6-state JSON not found at {ks4_json}!")
        return
        
    with open(ks4_json, "r") as f:
        ks4_res = json.load(f)
        
    dt = 0.010
    time_bins = np.arange(0, INTACT_DURATION + dt, dt)
    n_bins = len(time_bins) - 1
    cids = df_qc["cluster_id"].values
    n_units = len(cids)
    
    spike_mat = np.zeros((n_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        ts = spk_times[spk_clusters == cid]
        ts_valid = ts[(ts >= 0) & (ts <= INTACT_DURATION)]
        counts, _ = np.histogram(ts_valid, bins=time_bins)
        spike_mat[:, u_idx] = counts
        
    # Predict or reconstruct state sequence
    from scripts.experiments.e097.analyze_intact_kilosort4_multi_state_population import fit_canonical_multi_state_hmm
    hmm_fit = fit_canonical_multi_state_hmm(spike_mat, n_states=6, dt=dt, breath_data=breath_data)
    state_seq = hmm_fit["state_seq"]
    
    # 3. Analyze Off-Switch Shift Mechanisms
    shift_data = analyze_breath_shift_mechanisms(breath_data, state_seq, dt=dt)
    out_shift_json = out_base / "post_gasping_offswitch_shift_analysis.json"
    with open(out_shift_json, "w") as f:
        json.dump(shift_data, f, indent=2)
    logger.info(f"Saved shift analysis data to {out_shift_json}")
    
    # 4. Generate Shift Investigation Publication Figure
    png_shift = out_base / "investigation_post_gasping_offswitch_shift.png"
    generate_shift_investigation_figure(shift_data, png_shift)
    
    # 5. Identify Top Co-Modulated Units
    top_comod_indices, df_comod = find_top_comodulated_units(spike_mat, state_seq, depths, dt=dt, top_k=25)
    df_comod.to_csv(out_base / "top_comodulated_units.csv", index=False)
    
    # 6. Generate Population Rasters (All Units and Co-Modulated Units)
    generate_population_rasters(
        spk_times=spk_times,
        spk_clusters=spk_clusters,
        df_qc=df_qc,
        nidq_file=nidq_file,
        state_seq=state_seq,
        comod_unit_indices=top_comod_indices,
        out_dir=out_base
    )
    
    logger.info("\nALL INVESTIGATIONS AND RASTER GENERATIONS COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
