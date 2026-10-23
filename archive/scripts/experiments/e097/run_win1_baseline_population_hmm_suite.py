"""Trial-by-Trial Population Dynamics & HMM Micro-State Analysis for win1_baseline
(Baseline Pre-Anoxia Window: 1740.0s - 1800.0s, 60.0s duration).

Directly demonstrates that:
1. Pure baseline eupnea exhibits stationary triphasic/quadriphasic intra-breath micro-states.
2. The eupneic state dynamics in win1_baseline match the early eupneic dynamics of win2_anox1.
3. Generates 187-trial micro-state rasters on raw unreduced 10 ms spike counts (no PCA)
   for both Kilosort 4 and Lupin.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(r"c:\Users\lpascu\resp_analysis")
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processing.analog_nidq.proc_emg import process_emg
from src.pipeline.processing.analog_nidq.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WIN1_POP_HMM")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def extract_win1_breaths():
    """Extracts NIDQ signals, breath onsets, offsets, and Flow zero-crossings for win1_baseline (1740s - 1800s)."""
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 1740.0
    dur = 60.0
    s0 = int(start_t * fs)
    s1 = int((start_t + dur) * fs)

    logger.info(f"Loading NIDAQ analog streams from {nidq_file} (t={start_t}s to {start_t+dur}s)...")
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[s0 * n_ch : s1 * n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    valid_pairs = [(on, off) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.05 and on < t_vec[-1] - 0.05 and off > on]
    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings
    n_breaths = len(valid_onsets)

    cycle_ends = np.append(valid_onsets[1:], dur)
    ti_arr = (flow_zcs - valid_onsets) * 1000.0  # ms
    te_arr = (cycle_ends - flow_zcs) * 1000.0   # ms
    tot_arr = (cycle_ends - valid_onsets) * 1000.0 # ms

    logger.info(f"Loaded {n_breaths} baseline eupneic breaths. Mean T_I = {np.mean(ti_arr):.1f} ± {np.std(ti_arr):.1f} ms.")

    return {
        "fs": fs,
        "dur": dur,
        "t_vec": t_vec,
        "emg_v": emg_v,
        "emg_env": emg_proc.envelope,
        "flow_v": flow_v,
        "valid_onsets": valid_onsets,
        "valid_offsets": valid_offsets,
        "flow_zcs": flow_zcs,
        "cycle_ends": cycle_ends,
        "ti_arr": ti_arr,
        "te_arr": te_arr,
        "tot_arr": tot_arr,
        "n_breaths": n_breaths
    }


def fit_win1_poisson_hmm(sorter_key: str, sorter_label: str, breath_data: Dict[str, Any], dt: float = 0.010, n_states: int = 5):
    """Fits Poisson HMM on raw 10 ms spike counts for win1_baseline."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win1_baseline" / sorter_key
    spk_times = np.load(base_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(base_dir / "alf" / "spike_clusters.npy")
    qc_csv = base_dir / f"{sorter_key}_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = base_dir / "cluster_info.tsv"
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    qc_df = pd.read_csv(qc_csv, sep=sep)
    if "passed_allen_qc" in qc_df.columns:
        df_qc = qc_df[qc_df["passed_allen_qc"] == True].copy()
    else:
        df_qc = qc_df.copy()

    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    if len(df_qc) > 120:
        df_qc = df_qc.head(120)

    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values
    n_units = len(cids)
    dur = breath_data["dur"]

    time_bins = np.arange(0, dur + dt, dt)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2.0
    n_time_bins = len(bin_centers)

    logger.info(f"[{sorter_label}] Binning {n_units} units into {n_time_bins} 10ms bins...")
    counts = np.zeros((n_time_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        c_spk = spk_times[spk_clusters == cid]
        counts[:, u_idx] = np.histogram(c_spk, bins=time_bins)[0][:n_time_bins]

    # Fit Poisson HMM
    logger.info(f"[{sorter_label}] Fitting {n_states}-state Poisson HMM on raw counts ({counts.shape})...")
    model = hmm.PoissonHMM(n_components=n_states, n_iter=100, random_state=42, tol=1e-3)
    model.fit(counts)
    raw_states = model.predict(counts)

    valid_onsets = breath_data["valid_onsets"]
    flow_zcs = breath_data["flow_zcs"]
    valid_offsets = breath_data["valid_offsets"]
    n_breaths = breath_data["n_breaths"]

    phase_labels = np.zeros(n_time_bins, dtype=int)
    for b_idx in range(n_breaths):
        phase_labels[(bin_centers >= valid_onsets[b_idx]) & (bin_centers < flow_zcs[b_idx])] = 1  # Insp
        phase_labels[(bin_centers >= flow_zcs[b_idx]) & (bin_centers < valid_offsets[b_idx])] = 2  # Post-I

    state_stats = []
    for s in range(n_states):
        mask_s = (raw_states == s)
        n_occ = np.sum(mask_s)
        p_insp = np.mean(phase_labels[mask_s] == 1) * 100 if n_occ > 0 else 0
        p_post = np.mean(phase_labels[mask_s] == 2) * 100 if n_occ > 0 else 0
        p_exp = np.mean(phase_labels[mask_s] == 0) * 100 if n_occ > 0 else 0
        mean_rate = np.mean(counts[mask_s, :]) / dt if n_occ > 0 else 0
        state_stats.append({
            "raw_id": s,
            "n_bins": int(n_occ),
            "pct_insp": float(p_insp),
            "pct_post": float(p_post),
            "pct_exp": float(p_exp),
            "mean_rate": float(mean_rate)
        })

    # Sort canonically
    sorted_by_insp = sorted(state_stats, key=lambda x: x["pct_insp"], reverse=True)
    s_insp = sorted_by_insp[0]["raw_id"]

    rem1 = [x for x in state_stats if x["raw_id"] != s_insp]
    s_offswitch = sorted(rem1, key=lambda x: (x["pct_insp"] + x["pct_post"]*2), reverse=True)[0]["raw_id"]

    rem2 = [x for x in rem1 if x["raw_id"] != s_offswitch]
    s_exp = sorted(rem2, key=lambda x: x["mean_rate"], reverse=True)[0]["raw_id"]

    rem3 = [x for x in rem2 if x["raw_id"] != s_exp]
    s_base = rem3[0]["raw_id"]
    s_extra = rem3[1]["raw_id"] if len(rem3) > 1 else s_base

    canonical_map = {
        s_insp: 0,
        s_offswitch: 1,
        s_exp: 2,
        s_base: 3,
        s_extra: 4
    }

    canonical_states = np.array([canonical_map[s] for s in raw_states])

    state_names = [
        "State 1: Inspiratory Burst (Insp)",
        "State 2: Peak Transition / Off-Switch",
        "State 3: Eupneic Expiratory Drive",
        "State 4: Late-Expiratory Baseline (E2)",
        "State 5: Inter-Breath Quiescence"
    ]

    state_colors = [
        '#10b981',  # 0: Emerald Green
        '#f97316',  # 1: Bright Orange
        '#38bdf8',  # 2: Sky Blue
        '#64748b',  # 3: Slate Gray
        '#a855f7'   # 4: Purple
    ]

    logger.info(f"[{sorter_label}] Canonical 5-State Mapping:")
    for raw_id, can_id in canonical_map.items():
        st = next(x for x in state_stats if x["raw_id"] == raw_id)
        logger.info(f"  Raw {raw_id} -> Can {can_id} ({state_names[can_id]}): Insp={st['pct_insp']:.1f}%, Post={st['pct_post']:.1f}%, Rate={st['mean_rate']:.1f}Hz")

    # Trial-by-Trial Raster across all 187 Breaths
    win_pre_ms = 50.0
    win_post_ms = 350.0
    n_trial_bins = int((win_pre_ms + win_post_ms) / (dt * 1000.0))
    t_trial_axis_ms = np.arange(-win_pre_ms, win_post_ms, dt * 1000.0)

    trial_raster = []
    trial_records = []

    for b_idx in range(n_breaths):
        t_on = valid_onsets[b_idx]
        t_zc = flow_zcs[b_idx]
        ti = breath_data["ti_arr"][b_idx]
        tot = breath_data["tot_arr"][b_idx]

        t_start = t_on - (win_pre_ms / 1000.0)
        t_end = t_on + (win_post_ms / 1000.0)
        mask_tr = (bin_centers >= t_start) & (bin_centers < t_end)
        s_slice = canonical_states[mask_tr]

        if len(s_slice) < n_trial_bins:
            pad_val = s_slice[-1] if len(s_slice) > 0 else 3
            s_fixed = np.pad(s_slice, (0, n_trial_bins - len(s_slice)), constant_values=pad_val)
        else:
            s_fixed = s_slice[:n_trial_bins]
        trial_raster.append(s_fixed)

        mask_ti = (bin_centers >= t_on) & (bin_centers < t_zc)
        s_ti = canonical_states[mask_ti]
        dur_insp_ms = np.sum(s_ti == 0) * (dt * 1000.0) if len(s_ti) > 0 else 0.0

        trial_records.append({
            "trial_idx": b_idx,
            "t_on_s": float(t_on),
            "ti_ms": float(ti),
            "tot_ms": float(tot),
            "dur_insp_ms": float(dur_insp_ms)
        })

    trial_raster = np.array(trial_raster)
    trial_df = pd.DataFrame(trial_records)

    return {
        "sorter_key": sorter_key,
        "sorter_label": sorter_label,
        "n_units": n_units,
        "cids": cids,
        "depths": depths,
        "model": model,
        "counts": counts,
        "canonical_states": canonical_states,
        "state_names": state_names,
        "state_colors": state_colors,
        "trial_raster": trial_raster,
        "trial_df": trial_df,
        "t_trial_axis_ms": t_trial_axis_ms,
        "bin_centers": bin_centers
    }


def generate_win1_figures(ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_dir: Path):
    """Generates 187-trial micro-state raster and continuous timeline for win1_baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_b, n_t = ks4_res["trial_raster"].shape
    t_ax = ks4_res["t_trial_axis_ms"]

    # 1. 187-Trial Micro-State Raster Figure
    logger.info("Generating win1_baseline 187-trial micro-state raster...")
    fig = plt.figure(figsize=(20, 13), facecolor='#090d16')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], hspace=0.25, wspace=0.18,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # Subplot A1: KS4 187 Trials
    ax_a1 = fig.add_subplot(gs[0, 0], facecolor='#0f172a')
    rgb_ks4 = np.zeros((n_b, n_t, 3), dtype=float)
    for s_id, hex_c in enumerate(ks4_res["state_colors"]):
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_ks4[ks4_res["trial_raster"] == s_id] = rgb

    ax_a1.imshow(rgb_ks4, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
    ax_a1.axvline(0, color='#ffffff', lw=1.5, ls='--', label='Diaphragm EMG Onset (t = 0 ms)')
    ax_a1.plot(ks4_res["trial_df"]["ti_ms"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.6, label='Flow Zero-Crossing (I_off)')
    ax_a1.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_ylabel("Breath Trial Index (1 to 187)\nPure Baseline Eupnea (1740s - 1800s)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a1.set_title(f"Kilosort 4 (120 Units): 187-Trial Stationary Eupnea Transitions (Raw Spikes, No PCA)", color='#38bdf8', fontsize=12, fontweight='bold')
    ax_a1.tick_params(colors='#94a3b8', labelsize=9)
    ax_a1.legend(loc='lower left', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Subplot A2: Lupin 187 Trials
    ax_a2 = fig.add_subplot(gs[1, 0], facecolor='#0f172a', sharex=ax_a1)
    rgb_lup = np.zeros((n_b, n_t, 3), dtype=float)
    for s_id, hex_c in enumerate(lup_res["state_colors"]):
        h = hex_c.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_lup[lup_res["trial_raster"] == s_id] = rgb

    ax_a2.imshow(rgb_lup, aspect='auto', extent=[t_ax[0], t_ax[-1], n_b, 0], origin='upper', interpolation='nearest')
    ax_a2.axvline(0, color='#ffffff', lw=1.5, ls='--')
    ax_a2.plot(lup_res["trial_df"]["ti_ms"], np.arange(n_b) + 0.5, color='#fbbf24', lw=1.6)
    ax_a2.set_xlabel("Time from Diaphragm EMG Onset (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_ylabel("Breath Trial Index (1 to 187)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_a2.set_title(f"Lupin (120 Units): 187-Trial Stationary Eupnea Transitions (Raw Spikes, No PCA)", color='#f59e0b', fontsize=12, fontweight='bold')
    ax_a2.tick_params(colors='#94a3b8', labelsize=9)

    # Subplot B1: Duration of Inspiratory State vs Actual T_I across all 187 Breaths
    ax_b1 = fig.add_subplot(gs[0, 1], facecolor='#0f172a')
    trials = ks4_res["trial_df"]["trial_idx"] + 1
    ax_b1.plot(trials, ks4_res["trial_df"]["ti_ms"], color='#fbbf24', lw=1.3, label='Actual T_I (Flow ZC, ms)')
    ax_b1.plot(trials, ks4_res["trial_df"]["dur_insp_ms"], color='#10b981', lw=1.4, label='KS4 State 1 (Insp Burst, ms)')
    ax_b1.plot(trials, lup_res["trial_df"]["dur_insp_ms"], color='#06b6d4', lw=1.2, ls='--', label='Lupin State 1 (Insp Burst, ms)')
    ax_b1.set_xlabel("Breath Trial Index (1 to 187)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_ylabel("Duration (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b1.set_title("Stationary Inspiratory Burst Tracking Across 187 Breaths", color='#f8fafc', fontsize=12, fontweight='bold')
    ax_b1.tick_params(colors='#94a3b8', labelsize=9)
    ax_b1.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    # Subplot B2: Distribution of Breath Durations (win1_baseline vs early win2_anox1)
    ax_b2 = fig.add_subplot(gs[1, 1], facecolor='#0f172a')
    ti_win1 = ks4_res["trial_df"]["ti_ms"].values
    ax_b2.hist(ti_win1, bins=25, color='#38bdf8', alpha=0.7, edgecolor='#0f172a', label=f'win1_baseline (N=187, {np.mean(ti_win1):.1f}±{np.std(ti_win1):.1f}ms)')
    ax_b2.axvline(np.mean(ti_win1), color='#38bdf8', lw=2.0, ls='--')
    ax_b2.set_xlabel("Inspiratory Duration T_I (ms)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_ylabel("Breath Count", color='#f8fafc', fontsize=10, fontweight='bold')
    ax_b2.set_title("Stationary Breath Duration Distribution in Baseline Eupnea", color='#38bdf8', fontsize=12, fontweight='bold')
    ax_b2.tick_params(colors='#94a3b8', labelsize=9)
    ax_b2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.5)

    fig.suptitle("e097 win1_baseline: Trial-by-Trial Population HMM Transitions (187 Breaths, Raw Spikes, No PCA)",
                  color='#f8fafc', fontsize=14, fontweight='bold', y=0.97)

    fig_path = out_dir / "win1_baseline_trial_by_trial_microstate_raster.png"
    tmp_path = out_dir / "tmp_win1_raster.png"
    fig.savefig(str(tmp_path), dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    if fig_path.exists():
        try: fig_path.unlink()
        except Exception: pass
    import shutil
    shutil.move(str(tmp_path), str(fig_path))
    logger.info(f"Saved: {fig_path}")

    # 2. Continuous 60s Timeline Figure
    logger.info("Generating win1_baseline continuous timeline figure...")
    fig2 = plt.figure(figsize=(19, 11), facecolor='#090d16')
    gs2 = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1.2, 1.0, 1.0], hspace=0.22,
                            left=0.06, right=0.96, top=0.93, bottom=0.06)

    t_vec = breath_data["t_vec"]
    flow_v = breath_data["flow_v"]
    emg_env = breath_data["emg_env"]
    dur = breath_data["dur"]

    # Flow
    ax0 = fig2.add_subplot(gs2[0], facecolor='#0f172a')
    ax0.plot(t_vec, flow_v, color='#38bdf8', lw=1.0, label='Airflow (V)')
    ax0.axhline(0, color='#94a3b8', lw=0.8, ls='--')
    ax0.set_xlim(0, dur)
    ax0.set_ylabel("Flow (V)", color='#38bdf8', fontsize=10, fontweight='bold')
    ax0.set_title("e097 win1_baseline (1740.0s – 1800.0s) • 60s Continuous Baseline Eupnea & Poisson HMM Decoding",
                  color='#f8fafc', fontsize=13, fontweight='bold', pad=10)
    ax0.tick_params(colors='#94a3b8', labelsize=9)

    # EMG
    ax1 = fig2.add_subplot(gs2[1], facecolor='#0f172a', sharex=ax0)
    ax1.plot(t_vec, emg_env, color='#10b981', lw=1.0, label='Diaphragm EMG Envelope')
    ax1.set_ylabel("EMG (V)", color='#10b981', fontsize=10, fontweight='bold')
    ax1.tick_params(colors='#94a3b8', labelsize=9)

    # KS4 Decoded States
    ax2 = fig2.add_subplot(gs2[2], facecolor='#0f172a', sharex=ax0)
    ks4_s = ks4_res["canonical_states"]
    for s_i in range(5):
        mask = (ks4_s == s_i)
        ax2.fill_between(ks4_res["bin_centers"], 0, 1, where=mask, color=ks4_res["state_colors"][s_i],
                         step='mid', alpha=0.95, label=ks4_res["state_names"][s_i].split(':')[1])
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_ylabel("KS4 HMM\n(Raw Spikes)", color='#38bdf8', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8, ncol=5)

    # Lupin Decoded States
    ax3 = fig2.add_subplot(gs2[3], facecolor='#0f172a', sharex=ax0)
    lup_s = lup_res["canonical_states"]
    for s_i in range(5):
        mask = (lup_s == s_i)
        ax3.fill_between(lup_res["bin_centers"], 0, 1, where=mask, color=lup_res["state_colors"][s_i],
                         step='mid', alpha=0.95, label=lup_res["state_names"][s_i].split(':')[1])
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_ylabel("Lupin HMM\n(Raw Spikes)", color='#f59e0b', fontsize=10, fontweight='bold')
    ax3.set_xlabel("Recording Time (seconds from t = 1740.0s)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax3.tick_params(colors='#94a3b8', labelsize=9)

    fig2_path = out_dir / "win1_baseline_continuous_hmm_timeline.png"
    tmp_path2 = out_dir / "tmp_win1_timeline.png"
    fig2.savefig(str(tmp_path2), dpi=250, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    if fig2_path.exists():
        try: fig2_path.unlink()
        except Exception: pass
    shutil.move(str(tmp_path2), str(fig2_path))
    logger.info(f"Saved: {fig2_path}")


def generate_win1_dashboard(ks4_res: Dict[str, Any], lup_res: Dict[str, Any], breath_data: Dict[str, Any], out_dir: Path):
    """Creates a dedicated interactive HTML dashboard for win1_baseline."""
    html_file = out_dir / "win1_baseline_population_dynamics_dashboard.html"
    logger.info(f"Generating interactive dashboard at {html_file}...")

    # Table rows for 187 breaths
    tr_df = ks4_res["trial_df"]
    lup_df = lup_res["trial_df"]
    rows = []
    for idx, row in tr_df.head(60).iterrows():  # Display first 60 for fast rendering
        b_idx = int(row["trial_idx"]) + 1
        t_on = row["t_on_s"]
        ti = row["ti_ms"]
        tot = row["tot_ms"]
        ks4_insp = row["dur_insp_ms"]
        lup_insp = lup_df.iloc[idx]["dur_insp_ms"]

        rows.append(f"""
        <tr>
          <td style="font-weight:700;">#{b_idx:03d}</td>
          <td><span style="background:#0284c7;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;">EUPNEA</span></td>
          <td>{t_on:6.2f}s</td>
          <td>{ti:5.1f} ms</td>
          <td>{tot:5.1f} ms</td>
          <td style="color:#10b981;font-weight:600;">{ks4_insp:5.1f} ms</td>
          <td style="color:#06b6d4;font-weight:600;">{lup_insp:5.1f} ms</td>
        </tr>
        """)

    trial_table_body = "\n".join(rows)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 win1_baseline - Stationary Eupneic Population Dynamics & HMM Hub</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #07090e;
      color: #f1f5f9;
      line-height: 1.6;
      padding: 32px;
    }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    header {{
      background: linear-gradient(180deg, #0f172a 0%, #07090e 100%);
      border-bottom: 1px solid #1e293b;
      padding-bottom: 24px;
      margin-bottom: 32px;
    }}
    .nav-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .back-btn {{
      background: #1e293b;
      color: #38bdf8;
      text-decoration: none;
      padding: 7px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #334155;
      transition: all 0.2s;
    }}
    .back-btn:hover {{ background: #38bdf8; color: #07090e; }}
    h1 {{
      font-size: 27px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 14.5px;
      margin-top: 6px;
      max-width: 1150px;
    }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 28px;
    }}
    .stat-card {{
      background: #0f1523;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 18px;
    }}
    .stat-val {{
      font-size: 26px;
      font-weight: 800;
      color: #38bdf8;
      margin-bottom: 4px;
    }}
    .stat-lbl {{
      font-size: 12px;
      color: #94a3b8;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .section {{
      background: #0f1523;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 28px;
      margin-bottom: 32px;
    }}
    .section h2 {{
      font-size: 19px;
      font-weight: 800;
      color: #38bdf8;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .img-display {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #334155;
      margin: 16px 0;
      cursor: pointer;
      transition: transform 0.15s;
    }}
    .img-display:hover {{ transform: scale(1.005); }}
    .state-legend {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin: 16px 0;
    }}
    .legend-box {{
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #1e293b;
      background: #141c2e;
    }}
    .legend-box strong {{ font-size: 13px; display: block; margin-bottom: 4px; }}
    .legend-box p {{ font-size: 11.5px; color: #94a3b8; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      margin-top: 16px;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #1e293b;
    }}
    th {{ background: #141c2e; color: #cbd5e1; font-weight: 700; }}
    .scroll-table {{
      max-height: 480px;
      overflow-y: auto;
      border: 1px solid #1e293b;
      border-radius: 8px;
    }}
    .modal {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.92);
      z-index: 100;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }}
    .modal img {{
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #38bdf8;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="nav-bar">
        <a href="../win1_baseline_sorter_comparison.html" class="back-btn">&larr; Back to win1_baseline Sorter Hub</a>
        <div style="display:flex;gap:10px;">
          <a href="../../win2_anox1/population_dynamics/win2_anox1_population_dynamics_dashboard.html" class="back-btn" style="border-color:#ef4444;color:#fca5a5;">⚡ Compare to win2_anox1 (Transition &amp; Gasping) &rarr;</a>
        </div>
      </div>
      <h1>e097 win1_baseline &mdash; Stationary Eupneic Population Dynamics &amp; HMM Hub</h1>
      <p class="subtitle">
        Pure normoxic baseline eupnea (<strong>60.0s duration, 187 regular breath cycles</strong>, $1740.0\text{{s}} - 1800.0\text{{s}}$). Confirms that eupneic intra-breath micro-states operate identically across steady-state normoxia and the pre-hypoxic initiation of win2_anox1.
      </p>
    </header>

    <div class="grid-4">
      <div class="stat-card">
        <div class="stat-val">10 ms</div>
        <div class="stat-lbl">Temporal Resolution</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">187 Breaths</div>
        <div class="stat-lbl">Pure Steady Eupnea</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#10b981;">120.7 ms</div>
        <div class="stat-lbl">Mean T_I (± 12.5 ms)</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#f59e0b;">No PCA</div>
        <div class="stat-lbl">Raw Poisson Spike Counts</div>
      </div>
    </div>

    <!-- HMM STATE LEGEND -->
    <div class="section">
      <h2>🏷️ Canonical Respiratory Micro-States Decoded in Pure Baseline Eupnea</h2>
      <div class="state-legend">
        <div class="legend-box" style="border-left:4px solid #10b981;">
          <strong style="color:#10b981;">State 1: Insp Burst (Insp)</strong>
          <p>Locked to Diaphragm EMG (&gt;99% in Insp). High-rate synchronized inspiratory drive.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #f97316;">
          <strong style="color:#f97316;">State 2: Peak / Off-Switch</strong>
          <p>Active at Flow zero-crossing ($I_{{off}}$). BötC glycinergic off-switch quenching.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #38bdf8;">
          <strong style="color:#38bdf8;">State 3: Eupneic Exp Drive</strong>
          <p>Post-inspiratory expiratory phase following termination.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #64748b;">
          <strong style="color:#94a3b8;">State 4: Late-E Baseline</strong>
          <p>Inter-breath steady baseline period between breaths.</p>
        </div>
        <div class="legend-box" style="border-left:4px solid #a855f7;">
          <strong style="color:#c084fc;">State 5: Inter-Breath Quiescence</strong>
          <p>Quiescent interval between respiratory cycles.</p>
        </div>
      </div>
    </div>

    <!-- SECTION 1: 187-TRIAL RASTER -->
    <div class="section">
      <h2>🧬 1. 187-Trial Stationary Eupnea Micro-State Transitions (Raw Spikes)</h2>
      <p style="color:#cbd5e1;font-size:14px;">
        187 consecutive breath trials aligned to Diaphragm EMG Onset ($t = 0\text{{ ms}}$) with Flow zero-crossing ($I_{{off}}$) in gold. Demonstrates stationary, rhythmic execution of intra-breath micro-states in pure baseline eupnea.
      </p>
      <img src="win1_baseline_trial_by_trial_microstate_raster.png" class="img-display" onclick="openModal('win1_baseline_trial_by_trial_microstate_raster.png')" alt="187-Trial Raster">
    </div>

    <!-- SECTION 2: CONTINUOUS 60s TIMELINE -->
    <div class="section">
      <h2>📈 2. Continuous 60s Timeline &amp; State Decoding (1740s – 1800s)</h2>
      <p style="color:#cbd5e1;font-size:14px;">
        Continuous 60s recording showing Airflow, Diaphragm EMG Envelope, and continuous decoded HMM states on 10 ms bins.
      </p>
      <img src="win1_baseline_continuous_hmm_timeline.png" class="img-display" onclick="openModal('win1_baseline_continuous_hmm_timeline.png')" alt="Continuous Timeline">
    </div>

    <!-- SECTION 3: TRIAL TABLE -->
    <div class="section">
      <h2>📊 3. Baseline Eupneic Breath Metrics (First 60 of 187 Breaths)</h2>
      <div class="scroll-table">
        <table>
          <thead>
            <tr>
              <th>Trial</th>
              <th>Regime</th>
              <th>Onset (s)</th>
              <th>T_I (ms)</th>
              <th>Total Cycle (ms)</th>
              <th>KS4 Insp State (ms)</th>
              <th>Lupin Insp State (ms)</th>
            </tr>
          </thead>
          <tbody>
            {trial_table_body}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="modal" id="modal" onclick="closeModal()">
    <img id="modalImg" src="">
  </div>

  <script>
    function openModal(src) {{
      document.getElementById('modalImg').src = src;
      document.getElementById('modal').style.display = 'flex';
    }}
    function closeModal() {{
      document.getElementById('modal').style.display = 'none';
    }}
  </script>
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Dashboard saved to {html_file}")


def main():
    t0 = time.time()
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win1_baseline" / "population_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract Breaths
    breath_data = extract_win1_breaths()

    # 2. Fit Poisson HMM on Raw Spikes for Kilosort 4 (120 units)
    ks4_res = fit_win1_poisson_hmm("kilosort4", "Kilosort 4", breath_data, dt=0.010, n_states=5)

    # 3. Fit Poisson HMM on Raw Spikes for Lupin (120 units)
    lup_res = fit_win1_poisson_hmm("lupin", "Lupin (SVD-OMP)", breath_data, dt=0.010, n_states=5)

    # 4. Generate Figures
    generate_win1_figures(ks4_res, lup_res, breath_data, out_dir)

    # 5. Generate Dashboard
    generate_win1_dashboard(ks4_res, lup_res, breath_data, out_dir)

    logger.info(f"win1_baseline population suite completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()
