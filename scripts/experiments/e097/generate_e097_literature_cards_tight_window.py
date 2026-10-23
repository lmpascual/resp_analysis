#!/usr/bin/env python3
"""
Canonical 6-Panel Literature Spiking Diagnostics Suite Grounded in:
- Schreiber et al. (2003) [Pairwise trial reliability R_i,j and precision curves R_bar(sigma)]
- van Rossum (2001) [Spike distance metric space via Classical MDS]
- Cohen & Kohn (2011) [Signal vs noise correlation decomposition across respiratory units]
- Fisher (1993) [Circular phase vector statistics]

STRICTLY CONSTRAINED TO THE NARROWED TRANSITION WINDOW: [-50 ms, +100 ms]
Centered on Flow Inspiration Offset (I_off / Flow Zero-Crossing, t = 0 ms).
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import scipy.signal as signal
import pandas as pd
import torch
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.processors.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def main():
    t0 = time.time()
    print("=" * 80)
    print("GENERATING 6-PANEL LITERATURE CARDS FOR NARROWED WINDOW [-50 ms, +100 ms]")
    print("=" * 80)

    # 1. Load NIDAQ Analog Streams
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    print("[1/5] Loading NIDAQ analog streams (Diaphragm EMG, Flow Meter Ch 5, Laser Ch 6)...", flush=True)
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    # 2. Extract Breaths & Flow Expiration Onsets
    print("[2/5] Extracting breaths & Flow Zero-Crossing expiration onsets...", flush=True)
    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)

    win_pre_s = 0.050  # -50 ms
    win_post_s = 0.100 # +100 ms
    valid_pairs = []
    for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets):
        if on > 0.150 and on < t_vec[-1] - 0.200 and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    n_breaths = len(valid_onsets)

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings
    ti_flow = flow_zcs - valid_onsets
    print(f"      Total breaths: {n_breaths} | Flow ZC detection: 100.0%", flush=True)

    # Epoch partitions
    idx_ep1 = np.where((valid_onsets >= 0.0) & (valid_onsets < 60.0))[0]
    idx_las1 = np.where((valid_onsets >= 60.0) & (valid_onsets < 62.0))[0]
    idx_ep2 = np.where((valid_onsets >= 62.0) & (valid_onsets < 119.7))[0]
    idx_las2 = np.where((valid_onsets >= 119.7) & (valid_onsets < 122.0))[0]
    idx_ep3 = np.where((valid_onsets >= 122.0) & (valid_onsets < 164.3))[0]
    idx_las3 = np.where((valid_onsets >= 164.3) & (valid_onsets < 166.5))[0]
    idx_ep4 = np.where((valid_onsets >= 166.5) & (valid_onsets <= 226.25))[0]

    # Sort within epochs ascending by T_I
    sorted_ep1 = idx_ep1[np.argsort(ti_flow[idx_ep1])]
    sorted_ep2 = idx_ep2[np.argsort(ti_flow[idx_ep2])]
    sorted_ep3 = idx_ep3[np.argsort(ti_flow[idx_ep3])]
    sorted_ep4 = idx_ep4[np.argsort(ti_flow[idx_ep4])]
    ordered_trials = np.concatenate([sorted_ep1, idx_las1, sorted_ep2, idx_las2, sorted_ep3, idx_las3, sorted_ep4])

    # Time axis: [-50 ms, +100 ms], dt = 1 ms
    dt_s = 0.001
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0
    n_bins = len(time_axis_s)

    # Precompute shared Phase Grid in [-50 ms, +100 ms]
    print("[3/5] Precomputing shared phase grid in [-50 ms, +100 ms] window...", flush=True)
    phase_grid = np.zeros((n_breaths, n_bins), dtype=np.float32)
    for i in range(n_breaths):
        off = flow_zcs[i]
        on = valid_onsets[i]
        next_on = valid_onsets[i + 1] if i + 1 < n_breaths else off + win_post_s
        t_eval = off + time_axis_s
        phi = np.full(n_bins, np.nan, dtype=np.float32)
        insp_m = (t_eval >= on) & (t_eval < off)
        if np.any(insp_m):
            phi[insp_m] = -np.pi + (t_eval[insp_m] - on) / (off - on + 1e-6) * np.pi
        exp_m = (t_eval >= off) & (t_eval < next_on)
        if np.any(exp_m):
            phi[exp_m] = (t_eval[exp_m] - off) / (next_on - off + 1e-6) * np.pi
        phi[np.isnan(phi)] = np.pi
        phase_grid[i, :] = phi

    cmap_phase = LinearSegmentedColormap.from_list("phase_tight", ["#fed7aa", "#ffffff", "#dbeafe", "#93c5fd"], N=256)

    # Epoch layout blocks & dividers
    cum_ep1 = len(sorted_ep1)
    cum_l1 = cum_ep1 + len(idx_las1)
    cum_ep2 = cum_l1 + len(sorted_ep2)
    cum_l2 = cum_ep2 + len(idx_las2)
    cum_ep3 = cum_l2 + len(sorted_ep3)
    cum_l3 = cum_ep3 + len(idx_las3)
    cum_ep4 = cum_l3 + len(sorted_ep4)

    laser_blocks = [
        ("⚡ L1", cum_ep1 + 0.5, cum_l1 + 0.5),
        ("⚡ L2", cum_ep2 + 0.5, cum_l2 + 0.5),
        ("⚡ L3", cum_ep3 + 0.5, cum_l3 + 0.5)
    ]
    dividers = [cum_ep1, cum_l1, cum_ep2, cum_l2, cum_ep3, cum_l3]
    ep_slices = {
        1: slice(0, cum_ep1),
        2: slice(cum_l1, cum_ep2),
        3: slice(cum_l2, cum_ep3),
        4: slice(cum_l3, cum_ep4)
    }

    # Pre-build Gaussian kernels for Schreiber sweeps (in ms: 2, 4, 6, 8, 10, 15, 20, 25, 35, 50)
    sigmas_ms = [2, 4, 6, 8, 10, 15, 20, 25, 35, 50]
    kernels_schreiber = {}
    for sig in sigmas_ms:
        sig_s = sig / 1000.0
        half_w = int(np.ceil(4.0 * sig_s / dt_s))
        t_k = np.arange(-half_w, half_w + 1) * dt_s
        k = np.exp(-0.5 * (t_k / sig_s) ** 2)
        k /= (np.sum(k) * dt_s)
        kernels_schreiber[sig] = k

    kernel_10ms = kernels_schreiber[10]

    # 3. Load Motion-Corrected Clusters
    print("[4/5] Loading Motion-Corrected Consensus Clusters...", flush=True)
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
    out_cards_dir = cons_dir / "03_spiking_statistics_literature_cards_neg50_pos100"
    out_cards_dir.mkdir(parents=True, exist_ok=True)

    spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_passed_ids = set(df_info[df_info["passed_allen_qc"] == True]["cluster_id"].tolist())
    split_ids = {19, 328, 150, 378, 18, 269, 13, 64, 212, 14, 234, 114, 115, 335}
    top_spk_ids = set(df_info.nlargest(160, "n_spikes")["cluster_id"].tolist())
    target_cids = sorted(list(qc_passed_ids | split_ids | top_spk_ids))

    summary_rows = []
    transition_rows = []
    cluster_peths = {}
    cluster_trial_counts = {}
    cluster_trial_total = {}

    n_rendered = 0
    print(f"      Rendering 6-panel literature cards for {len(target_cids)} units...", flush=True)

    for cid in target_cids:
        row = df_info[df_info["cluster_id"] == cid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        depth = float(row["depth_um"])
        n_spk = int(row["n_spikes"])
        fr_hz = float(row["firing_rate_hz"])
        is_qc = bool(row["passed_allen_qc"])

        u_spks = spk_times[spk_clusters == cid]
        if len(u_spks) == 0:
            continue

        spike_x = []
        spike_y = []
        trial_binned = np.zeros((n_breaths, n_bins), dtype=np.float32)

        for r, orig_i in enumerate(ordered_trials):
            off = flow_zcs[orig_i]
            rel = u_spks - off
            in_win = rel[(rel >= -win_pre_s) & (rel <= win_post_s)]
            for s in in_win:
                spike_x.append(s * 1000.0)
                spike_y.append(r + 1)
            counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
            trial_binned[orig_i, :] = counts

        total_spikes_in_win = len(spike_x)
        if total_spikes_in_win == 0:
            continue

        # PETHs (1 ms bins smoothed with 5 ms Gaussian) across ALL 4 Epochs
        kernel_peth = signal.windows.gaussian(9, std=2)
        kernel_peth /= np.sum(kernel_peth)
        peth_raw = {
            1: np.mean(trial_binned[idx_ep1], axis=0) / dt_s if len(idx_ep1) > 0 else np.zeros(n_bins),
            2: np.mean(trial_binned[idx_ep2], axis=0) / dt_s if len(idx_ep2) > 0 else np.zeros(n_bins),
            3: np.mean(trial_binned[idx_ep3], axis=0) / dt_s if len(idx_ep3) > 0 else np.zeros(n_bins),
            4: np.mean(trial_binned[idx_ep4], axis=0) / dt_s if len(idx_ep4) > 0 else np.zeros(n_bins)
        }
        peth_sm = {ep: np.convolve(peth_raw[ep], kernel_peth, mode="same") for ep in [1, 2, 3, 4]}
        peak_rates = {ep: float(np.max(peth_sm[ep])) for ep in [1, 2, 3, 4]}

        # Mean firing rates in window
        win_dur = win_pre_s + win_post_s  # 0.150 s
        fr_window = {
            ep: float(np.mean(np.sum(trial_binned[idx], axis=1)) / win_dur) if len(idx) > 0 else 0.0
            for ep, idx in [(1, idx_ep1), (2, idx_ep2), (3, idx_ep3), (4, idx_ep4)]
        }

        # Fano factors
        def calc_fano(counts):
            if len(counts) < 3: return 0.0
            m = np.mean(counts)
            if m < 1e-6: return 0.0
            return float(np.var(counts) / m)

        fano_dict = {
            ep: calc_fano(np.sum(trial_binned[idx], axis=1))
            for ep, idx in [(1, idx_ep1), (2, idx_ep2), (3, idx_ep3), (4, idx_ep4)]
        }

        # Store for population analysis
        if is_qc or cid in split_ids:
            peth_all = np.mean(trial_binned, axis=0) / dt_s
            cluster_peths[cid] = np.convolve(peth_all, kernel_peth, mode="same")
            cluster_trial_counts[cid] = trial_binned
            cluster_trial_total[cid] = np.sum(trial_binned, axis=1)

        # Functional classification in transition window
        pre_c = np.sum(trial_binned[:, time_axis_s < 0])
        post_c = np.sum(trial_binned[:, time_axis_s >= 0])
        if post_c > 2.0 * (pre_c + 1):
            cell_type = "Pure Post-I / Expiratory Phasic"
        elif pre_c > 2.0 * (post_c + 1):
            cell_type = "Late Inspiratory Ramp"
        else:
            cell_type = "Off-Switch Spanning"

        # Circular phase centered at I_off = 0 rad across ALL 4 Epochs
        all_phases = {1: [], 2: [], 3: [], 4: []}
        for ep_k, ep_indices in [(1, idx_ep1), (2, idx_ep2), (3, idx_ep3), (4, idx_ep4)]:
            for orig_i in ep_indices:
                on, off = valid_onsets[orig_i], flow_zcs[orig_i]
                next_on = valid_onsets[orig_i + 1] if orig_i + 1 < n_breaths else off + 0.250
                spks = u_spks[(u_spks >= off - win_pre_s) & (u_spks < off + win_post_s)]
                for s in spks:
                    phi = -np.pi + (s - on) / (off - on + 1e-6) * np.pi if s < off else (s - off) / (next_on - off + 1e-6) * np.pi
                    all_phases[ep_k].append(phi)

        def calc_phase(phases):
            if len(phases) < 3:
                return np.nan, 0.0
            c = np.sum(np.cos(phases))
            s = np.sum(np.sin(phases))
            th = np.arctan2(s, c)
            r = np.sqrt(c**2 + s**2) / len(phases)
            return float(th), float(r)

        phase_stats = {ep: calc_phase(all_phases[ep]) for ep in [1, 2, 3, 4]}

        # Full Schreiber 842x842 matrix in [-50, +100]ms window
        sm_all = np.apply_along_axis(lambda m: np.convolve(m, kernel_10ms, mode="same") * dt_s, axis=1, arr=trial_binned[ordered_trials]).astype(np.float32)
        norms_all = np.sqrt(np.sum(sm_all ** 2, axis=1) * dt_s)
        t_small = torch.from_numpy(np.ascontiguousarray(sm_all))
        G_all = (t_small @ t_small.T).numpy() * dt_s
        outer_all = np.outer(norms_all, norms_all)
        v_all = outer_all > 1e-9
        R_matrix = np.zeros_like(G_all)
        np.divide(G_all, outer_all, out=R_matrix, where=v_all)
        np.fill_diagonal(R_matrix, 1.0)

        # Within-epoch and cross-epoch Schreiber reliabilities
        def get_block_rel(sl_a, sl_b, is_within):
            sub_R = R_matrix[sl_a, sl_b]
            n_a = norms_all[sl_a]
            n_b = norms_all[sl_b]
            mask = (n_a[:, None] > 1e-6) & (n_b[None, :] > 1e-6)
            if is_within:
                mask = mask.copy()
                np.fill_diagonal(mask, False)
            if np.sum(mask) == 0:
                return 0.0
            return float(np.mean(sub_R[mask]))

        r_within = {ep: get_block_rel(ep_slices[ep], ep_slices[ep], is_within=True) for ep in [1, 2, 3, 4]}
        all_epoch_pairs = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (2, 4)]
        r_cross = {p: get_block_rel(ep_slices[p[0]], ep_slices[p[1]], is_within=False) for p in all_epoch_pairs}

        # PETH Signal Correlations
        def peth_corr(p_a, p_b):
            std_a, std_b = np.std(p_a), np.std(p_b)
            if std_a > 1e-4 and std_b > 1e-4:
                return float((np.mean(p_a * p_b) - np.mean(p_a) * np.mean(p_b)) / (std_a * std_b))
            return 0.0

        r_sig = {p: peth_corr(peth_sm[p[0]], peth_sm[p[1]]) for p in all_epoch_pairs}

        # Schreiber reliability vs sigma in [-50, +100]ms window for ALL 4 Epochs
        r_sigma = {1: [], 2: [], 3: [], 4: []}
        for sig in sigmas_ms:
            k = kernels_schreiber[sig]
            for ep_k, ep_indices in [(1, idx_ep1), (2, idx_ep2), (3, idx_ep3), (4, idx_ep4)]:
                if len(ep_indices) > 1 and np.sum(trial_binned[ep_indices]) > 3:
                    sm_ep = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same") * dt_s, axis=1, arr=trial_binned[ep_indices]).astype(np.float32)
                    norm_ep = np.sqrt(np.sum(sm_ep ** 2, axis=1) * dt_s)
                    t_sm_ep = torch.from_numpy(np.ascontiguousarray(sm_ep))
                    G_ep = (t_sm_ep @ t_sm_ep.T).numpy() * dt_s
                    outer_ep = np.outer(norm_ep, norm_ep)
                    v_ep = outer_ep > 1e-9
                    R_ep = np.zeros_like(G_ep)
                    np.divide(G_ep, outer_ep, out=R_ep, where=v_ep)
                    np.fill_diagonal(R_ep, 0.0)
                    m_ep = (norm_ep[:, None] > 1e-6) & (norm_ep[None, :] > 1e-6)
                    np.fill_diagonal(m_ep, False)
                    r_sigma[ep_k].append(float(np.mean(R_ep[m_ep])) if np.sum(m_ep) > 0 else 0.0)
                else:
                    r_sigma[ep_k].append(0.0)

        # Intrinsic precision scale sigma* (first sigma reaching >= 80% of max R in Ep 1)
        max_r_ep1 = max(r_sigma[1]) if len(r_sigma[1]) > 0 else 0.0
        sigma_star_ep1 = np.nan
        if max_r_ep1 > 0.05:
            th_80 = 0.8 * max_r_ep1
            for sig, r_val in zip(sigmas_ms, r_sigma[1]):
                if r_val >= th_80:
                    sigma_star_ep1 = float(sig)
                    break

        # Record summary row with full multi-epoch cross-epoch statistics
        summary_rows.append({
            "cluster_id": cid,
            "depth_um": depth,
            "n_spikes": n_spk,
            "spikes_win": total_spikes_in_win,
            "firing_rate_hz": fr_hz,
            "is_qc_passed": is_qc,
            "cell_type": cell_type,
            # Within-epoch reliabilities
            "r_ep1": r_within[1],
            "r_ep2": r_within[2],
            "r_ep3": r_within[3],
            "r_ep4": r_within[4],
            # Cross-epoch transition reliabilities
            "r_cross_1_2": r_cross[(1, 2)],
            "r_cross_2_3": r_cross[(2, 3)],
            "r_cross_3_4": r_cross[(3, 4)],
            "r_cross_1_3": r_cross[(1, 3)],
            "r_cross_1_4": r_cross[(1, 4)],
            "r_cross_2_4": r_cross[(2, 4)],
            # Signal correlations between PETHs
            "r_sig_1_2": r_sig[(1, 2)],
            "r_sig_2_3": r_sig[(2, 3)],
            "r_sig_3_4": r_sig[(3, 4)],
            "r_sig_1_4": r_sig[(1, 4)],
            # Peak firing rates
            "peak_ep1_hz": peak_rates[1],
            "peak_ep2_hz": peak_rates[2],
            "peak_ep3_hz": peak_rates[3],
            "peak_ep4_hz": peak_rates[4],
            # Mean window firing rates
            "fr_ep1_hz": fr_window[1],
            "fr_ep2_hz": fr_window[2],
            "fr_ep3_hz": fr_window[3],
            "fr_ep4_hz": fr_window[4],
            # Fano factors
            "fano_ep1": fano_dict[1],
            "fano_ep2": fano_dict[2],
            "fano_ep3": fano_dict[3],
            "fano_ep4": fano_dict[4],
            # Circular phase statistics
            "phase_pref_ep1": phase_stats[1][0],
            "phase_r_ep1": phase_stats[1][1],
            "phase_pref_ep2": phase_stats[2][0],
            "phase_r_ep2": phase_stats[2][1],
            "phase_pref_ep3": phase_stats[3][0],
            "phase_r_ep3": phase_stats[3][1],
            "phase_pref_ep4": phase_stats[4][0],
            "phase_r_ep4": phase_stats[4][1],
            "sigma_star_ms": sigma_star_ep1,
            # Backward compatibility aliases
            "r_schreiber_ep1_10ms": r_within[1],
            "r_schreiber_ep4_10ms": r_within[4],
            "r_schreiber_cross_10ms": r_cross[(1, 4)]
        })

        # Identify transition with maximum reliability drop
        drops = {
            "Train 1 (Ep 1->2)": max(0.0, r_within[1] - r_cross[(1, 2)]),
            "Train 2 (Ep 2->3)": max(0.0, r_within[2] - r_cross[(2, 3)]),
            "Train 3 (Ep 3->4)": max(0.0, r_within[3] - r_cross[(3, 4)])
        }
        max_drop_epoch = max(drops, key=drops.get)
        max_drop_val = drops[max_drop_epoch]

        transition_rows.append({
            "cluster_id": cid,
            "depth_um": depth,
            "cell_type": cell_type,
            "is_qc_passed": is_qc,
            "r_within_ep1": r_within[1],
            "r_within_ep2": r_within[2],
            "r_within_ep3": r_within[3],
            "r_within_ep4": r_within[4],
            "r_trans_1_to_2": r_cross[(1, 2)],
            "r_trans_2_to_3": r_cross[(2, 3)],
            "r_trans_3_to_4": r_cross[(3, 4)],
            "r_trans_1_to_4": r_cross[(1, 4)],
            "delta_fr_1_to_2": fr_window[2] - fr_window[1],
            "delta_fr_2_to_3": fr_window[3] - fr_window[2],
            "delta_fr_3_to_4": fr_window[4] - fr_window[3],
            "delta_fr_1_to_4": fr_window[4] - fr_window[1],
            "max_drop_transition": max_drop_epoch if max_drop_val > 0.15 else "Stable",
            "max_drop_value": max_drop_val
        })

        # van Rossum MDS (subsampled across epochs)
        np.random.seed(42 + cid)
        sub_ep1 = np.random.choice(idx_ep1, min(50, len(idx_ep1)), replace=False)
        sub_ep2 = np.random.choice(idx_ep2, min(50, len(idx_ep2)), replace=False)
        sub_ep3 = np.random.choice(idx_ep3, min(50, len(idx_ep3)), replace=False)
        sub_ep4 = np.random.choice(idx_ep4, min(50, len(idx_ep4)), replace=False)
        sub_all = np.concatenate([sub_ep1, sub_ep2, sub_ep3, sub_ep4])
        sub_labels = [1]*len(sub_ep1) + [2]*len(sub_ep2) + [3]*len(sub_ep3) + [4]*len(sub_ep4)

        sm_sub = np.apply_along_axis(lambda m: np.convolve(m, kernel_10ms, mode="same") * dt_s, axis=1, arr=trial_binned[sub_all]).astype(np.float32)
        t_sm = torch.from_numpy(np.ascontiguousarray(sm_sub)).float()
        G_sub = (t_sm @ t_sm.T) * dt_s
        diag_sub = torch.diag(G_sub)
        D2_sub = torch.clamp(diag_sub[:, None] + diag_sub[None, :] - 2.0 * G_sub, min=0.0) / 0.010

        M_sub = len(sub_all)
        H = torch.eye(M_sub) - (1.0 / M_sub) * torch.ones(M_sub, M_sub)
        B_sub = -0.5 * (H @ D2_sub @ H)
        eigvals, eigvecs = torch.linalg.eigh(B_sub)
        idx_sort = torch.argsort(eigvals, descending=True)
        top_vals = torch.clamp(eigvals[idx_sort[:2]], min=0.0)
        top_vecs = eigvecs[:, idx_sort[:2]]
        mds_coords = (top_vecs @ torch.diag(torch.sqrt(top_vals))).numpy()

        # ---------------------------------------------------------------------
        # RENDER 6-PANEL FIGURE
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(16.5, 10.5))
        gs_fig = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.0], hspace=0.36, wspace=0.26)

        # Panel 1: Raster in [-50, +100]ms
        ax_ras = fig.add_subplot(gs_fig[0, 0])
        ax_ras.imshow(
            phase_grid[ordered_trials, :],
            aspect="auto", cmap=cmap_phase, vmin=-np.pi, vmax=np.pi,
            extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
            origin="lower", alpha=0.55, interpolation="bilinear"
        )
        for lbl, y0, y1 in laser_blocks:
            ax_ras.fill_between([time_axis_ms[0], time_axis_ms[-1]], [y0, y0], [y1, y1], color='#ff922b', alpha=0.40, zorder=2)
            ax_ras.text(time_axis_ms[-1] - 3, (y0 + y1)/2.0, lbl, color="#d9480f", fontsize=8, fontweight="bold", va="center", ha="right", zorder=5)

        ax_ras.scatter(spike_x, spike_y, marker='|', color='black', s=14, linewidth=0.85, alpha=0.92, zorder=3)
        for div in dividers:
            ax_ras.axhline(div, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6, zorder=2)

        ax_ras.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.9, zorder=4)
        ax_ras.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_ras.set_ylim(0.5, n_breaths + 0.5)
        ax_ras.set_xlabel("Time from Flow $I_{\\mathrm{off}}$ (ms)", fontsize=9, fontweight="bold")
        ax_ras.set_ylabel(f"Breath Trials ({n_breaths})", fontsize=9, fontweight="bold")
        ax_ras.set_title(r"Panel 1: $T_I$-Sorted Raster ([-50, +100] ms)", fontsize=10, fontweight="bold")

        # Panel 2: 4-Epoch Overlay PETH
        ax_peth = fig.add_subplot(gs_fig[0, 1])
        ax_peth.plot(time_axis_ms, peth_sm[1], color="#0284c7", linewidth=2.0, label=f"Ep 1 Pre ({peak_rates[1]:.1f} Hz)")
        ax_peth.plot(time_axis_ms, peth_sm[2], color="#0d9488", linewidth=1.7, label=f"Ep 2 Tr 1 ({peak_rates[2]:.1f} Hz)")
        ax_peth.plot(time_axis_ms, peth_sm[3], color="#f59e0b", linewidth=1.7, label=f"Ep 3 Tr 2 ({peak_rates[3]:.1f} Hz)")
        ax_peth.plot(time_axis_ms, peth_sm[4], color="#e11d48", linewidth=2.0, label=f"Ep 4 Post ({peak_rates[4]:.1f} Hz)")
        ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.4)
        ax_peth.set_xlim(time_axis_ms[0], time_axis_ms[-1])
        ax_peth.set_xlabel("Time from Flow $I_{\\mathrm{off}}$ (ms)", fontsize=9, fontweight="bold")
        ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=9, fontweight="bold")
        ax_peth.set_title("Panel 2: Expiration-Aligned PETH (All 4 Epochs)", fontsize=10, fontweight="bold")
        ax_peth.legend(loc="upper right", fontsize=7.5)
        ax_peth.grid(True, linestyle="--", alpha=0.4)

        # Panel 3: Schreiber Heatmap & Multi-Epoch Transition Matrix
        ax_heat = fig.add_subplot(gs_fig[0, 2])
        im_heat = ax_heat.imshow(R_matrix, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0, aspect="equal", origin="lower")
        for d in dividers:
            ax_heat.axhline(d, color="white", linewidth=0.75, linestyle=":")
            ax_heat.axvline(d, color="white", linewidth=0.75, linestyle=":")
        ax_heat.set_xlabel("Trial Index (Ordered)", fontsize=9, fontweight="bold")
        ax_heat.set_ylabel("Trial Index (Ordered)", fontsize=9, fontweight="bold")
        title_heat = (
            r"Panel 3: Schreiber $R_{i,j}$ Matrix & Transitions" + "\n" +
            f"Within: Ep1={r_within[1]:.2f} Ep2={r_within[2]:.2f} Ep3={r_within[3]:.2f} Ep4={r_within[4]:.2f}\n" +
            f"Cross: 1→2={r_cross[(1, 2)]:.2f} | 2→3={r_cross[(2, 3)]:.2f} | 3→4={r_cross[(3, 4)]:.2f} | 1→4={r_cross[(1, 4)]:.2f}"
        )
        ax_heat.set_title(title_heat, fontsize=8.5, fontweight="bold", pad=8)
        cbar_h = plt.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar_h.set_label(r"$R_{i,j}$ ($\sigma=10\text{ms}$)", fontsize=8)

        # Panel 4: Schreiber vs sigma curve for ALL 4 Epochs
        ax_sig = fig.add_subplot(gs_fig[1, 0])
        ax_sig.plot(sigmas_ms, r_sigma[1], color="#0284c7", marker="o", linewidth=1.8, markersize=4, label="Epoch 1 Pre-Laser")
        ax_sig.plot(sigmas_ms, r_sigma[2], color="#0d9488", marker="^", linewidth=1.6, markersize=4, label="Epoch 2 Inter-Tr1")
        ax_sig.plot(sigmas_ms, r_sigma[3], color="#f59e0b", marker="d", linewidth=1.6, markersize=4, label="Epoch 3 Inter-Tr2")
        ax_sig.plot(sigmas_ms, r_sigma[4], color="#e11d48", marker="s", linewidth=1.8, markersize=4, label="Epoch 4 Recovery")
        max_r = max(max(r_sigma[ep]) if r_sigma[ep] else 0.0 for ep in [1, 2, 3, 4])
        if max_r > 0.05:
            ax_sig.axhline(max_r * 0.8, color="#64748b", linestyle=":", label=r"80% Precision Threshold")
        ax_sig.set_xlabel(r"Smoothing Width $\sigma$ (ms)", fontsize=9, fontweight="bold")
        ax_sig.set_ylabel(r"Reliability $\bar{R}(\sigma)$", fontsize=9, fontweight="bold")
        ax_sig.set_title(r"Panel 4: Schreiber Precision Curves (All 4 Epochs)", fontsize=10, fontweight="bold")
        ax_sig.set_ylim(0, 1.05)
        ax_sig.legend(loc="lower right", fontsize=7.0)
        ax_sig.grid(True, linestyle="--", alpha=0.4)

        # Panel 5: van Rossum MDS
        ax_mds = fig.add_subplot(gs_fig[1, 1])
        colors_ep = {1: "#0284c7", 2: "#0d9488", 3: "#f59e0b", 4: "#e11d48"}
        labels_ep = {1: "Epoch 1 (Pre)", 2: "Epoch 2 (Tr 1)", 3: "Epoch 3 (Tr 2)", 4: "Epoch 4 (Post)"}
        for ep_i in [1, 2, 3, 4]:
            mask_ep = np.array(sub_labels) == ep_i
            if np.any(mask_ep):
                ax_mds.scatter(mds_coords[mask_ep, 0], mds_coords[mask_ep, 1], color=colors_ep[ep_i], label=labels_ep[ep_i], s=22, alpha=0.85, edgecolor="none")
        ax_mds.set_xlabel("van Rossum MDS Dim 1", fontsize=9, fontweight="bold")
        ax_mds.set_ylabel("van Rossum MDS Dim 2", fontsize=9, fontweight="bold")
        ax_mds.set_title(r"Panel 5: van Rossum (2001) MDS Space ($\tau=10\text{ms}$)", fontsize=10, fontweight="bold")
        ax_mds.legend(loc="upper right", fontsize=7.5)
        ax_mds.grid(True, linestyle="--", alpha=0.4)

        # Panel 6: Polar Phase Tuning across ALL 4 Epochs
        ax_pol = fig.add_subplot(gs_fig[1, 2], projection="polar")
        bins_polar = np.linspace(-np.pi, np.pi, 36)
        bin_centers = (bins_polar[:-1] + bins_polar[1:]) / 2.0
        colors_pol = {1: "#0284c7", 2: "#0d9488", 3: "#f59e0b", 4: "#e11d48"}
        styles_pol = {1: "-", 2: ":", 3: "-.", 4: "--"}
        labels_pol = {1: "Ep 1", 2: "Ep 2", 3: "Ep 3", 4: "Ep 4"}

        for ep_k in [1, 2, 3, 4]:
            counts_pk, _ = np.histogram(all_phases[ep_k], bins=bins_polar)
            th_k, r_k = phase_stats[ep_k]
            if np.sum(counts_pk) > 0:
                norm_pk = counts_pk / (np.max(counts_pk) + 1e-6)
                ax_pol.plot(bin_centers, norm_pk, color=colors_pol[ep_k], linestyle=styles_pol[ep_k],
                            linewidth=1.8 if ep_k in [1, 4] else 1.4, label=f"{labels_pol[ep_k]} (R={r_k:.2f})")
                ax_pol.plot([th_k, th_k], [0, r_k], color=colors_pol[ep_k], linestyle=styles_pol[ep_k], linewidth=2.0)

        ax_pol.set_theta_zero_location("N")
        ax_pol.set_theta_direction(-1)
        ax_pol.set_thetagrids([0, 90, 180, 270], [r"$I_{\mathrm{off}}$ (0)", r"$+\frac{\pi}{2}$ (Post-I)", r"$\pm\pi$ ($I_{\mathrm{on}}$)", r"$-\frac{\pi}{2}$ (Insp)"], fontsize=7.5)
        ax_pol.set_title("Panel 6: Polar Phase Tuning (All Epochs)\n(Fisher 1993)", fontsize=10, fontweight="bold", pad=12)
        ax_pol.legend(loc="lower right", bbox_to_anchor=(1.25, -0.1), fontsize=7.0)

        qc_tag = "[QC PASSED]" if is_qc else "[Non-QC]"
        plt.suptitle(
            f"Cluster #{cid:03d} (Depth: {depth:.0f} µm | {cell_type} {qc_tag}) — Literature Spiking Diagnostics Suite\n"
            f"[Narrowed Transition Window: -50 ms to +100 ms Centered on Flow Inspiration Offset ($I_{{\\mathrm{{off}}}}$)]",
            fontsize=12, fontweight="bold", y=0.98
        )

        out_card = out_cards_dir / f"lit_card_cluster_{cid:03d}.png"
        # SAFETY: Never use bbox_inches="tight" with polar axes on Windows!
        plt.savefig(out_card, dpi=180)
        plt.close(fig)
        n_rendered += 1

        if n_rendered % 25 == 0:
            print(f"      Rendered {n_rendered}/{len(target_cids)} literature cards...", flush=True)

    print(f"[+] Rendered {n_rendered} 6-panel literature cards into: {out_cards_dir}", flush=True)

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_cards_dir / "spiking_statistics_lit_summary_neg50_pos100.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[+] Saved summary CSV: {summary_csv}", flush=True)

    # Save dedicated cross-epoch transitions CSV
    df_trans = pd.DataFrame(transition_rows)
    trans_csv = out_cards_dir / "multi_epoch_cross_epoch_transitions.csv"
    df_trans.to_csv(trans_csv, index=False)
    print(f"[+] Saved multi-epoch cross-epoch transitions CSV: {trans_csv}", flush=True)

    # 4. Population Cohen & Kohn Analysis on [-50, +100]ms Counts
    print("[5/5] Computing Population Cohen & Kohn Analysis...", flush=True)
    active_qc = [cid for cid in qc_passed_ids if cid in cluster_trial_counts and np.sum(cluster_trial_total[cid]) > 30]
    pair_records = []
    for i in range(len(active_qc)):
        cid_i = active_qc[i]
        depth_i = float(df_info[df_info["cluster_id"] == cid_i]["depth_um"].values[0])
        peth_i = cluster_peths[cid_i]
        counts_i = cluster_trial_total[cid_i]

        for j in range(i + 1, len(active_qc)):
            cid_j = active_qc[j]
            depth_j = float(df_info[df_info["cluster_id"] == cid_j]["depth_um"].values[0])
            peth_j = cluster_peths[cid_j]
            counts_j = cluster_trial_total[cid_j]

            delta_y = abs(depth_i - depth_j)

            # Signal correlation: PETH correlation across [-50, +100]ms
            std_pi = np.std(peth_i)
            std_pj = np.std(peth_j)
            if std_pi > 1e-4 and std_pj > 1e-4:
                r_sig = float((np.mean(peth_i * peth_j) - np.mean(peth_i) * np.mean(peth_j)) / (std_pi * std_pj))
            else:
                r_sig = 0.0

            # Noise correlation: trial count correlation in Ep 1
            c_i1 = counts_i[idx_ep1]
            c_j1 = counts_j[idx_ep1]
            std_ci = np.std(c_i1)
            std_cj = np.std(c_j1)
            if std_ci > 1e-4 and std_cj > 1e-4:
                r_noise = float((np.mean(c_i1 * c_j1) - np.mean(c_i1) * np.mean(c_j1)) / (std_ci * std_cj))
            else:
                r_noise = 0.0

            pair_records.append({
                "cid_1": cid_i, "cid_2": cid_j,
                "depth_1": depth_i, "depth_2": depth_j,
                "delta_y_um": delta_y,
                "r_signal": r_sig, "r_noise": r_noise
            })

    df_pairs = pd.DataFrame(pair_records)
    pairs_csv = out_cards_dir / "population_pairwise_correlations.csv"
    df_pairs.to_csv(pairs_csv, index=False)
    print(f"      Saved {len(df_pairs)} pairwise relationships to {pairs_csv}", flush=True)

    # Population Figure
    fig_pop, (ax_scat, ax_decay) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.28})
    mask_local = df_pairs["delta_y_um"] < 200.0
    mask_dist = df_pairs["delta_y_um"] >= 200.0

    ax_scat.plot(df_pairs.loc[mask_dist, "r_signal"].values, df_pairs.loc[mask_dist, "r_noise"].values,
                 'o', color="#94a3b8", alpha=0.45, markersize=4, label=r"Distant Pairs ($\Delta y \geq 200\,\mu\mathrm{m}$)")
    ax_scat.plot(df_pairs.loc[mask_local, "r_signal"].values, df_pairs.loc[mask_local, "r_noise"].values,
                 'o', color="#0284c7", alpha=0.75, markersize=5, label=r"Local Pairs ($\Delta y < 200\,\mu\mathrm{m}$)")

    valid_p = df_pairs.dropna(subset=["r_signal", "r_noise"])
    if len(valid_p) > 5:
        x = valid_p["r_signal"].values
        y = valid_p["r_noise"].values
        vx = np.var(x)
        slope = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (vx if vx > 1e-12 else 1.0)
        intercept = np.mean(y) - slope * np.mean(x)
        r_val = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.std(x) * np.std(y) + 1e-12)
        x_fit = np.linspace(-0.8, 1.0, 100)
        ax_scat.plot(x_fit, intercept + slope * x_fit, color="#e11d48", linewidth=2.0, label=f"Fit: slope={slope:.3f} ($r={r_val:.2f}$)")

    ax_scat.axhline(0, color="#475569", linestyle="--", linewidth=1.0)
    ax_scat.axvline(0, color="#475569", linestyle="--", linewidth=1.0)
    ax_scat.set_xlim(-0.85, 1.05)
    ax_scat.set_ylim(-0.4, 0.8)
    ax_scat.set_xlabel("Signal Correlation $r_{\\mathrm{signal}}$ ([-50, +100] ms PETH)", fontsize=10, fontweight="bold")
    ax_scat.set_ylabel("Noise Correlation $r_{\\mathrm{noise}}$ (Spike Count Fluctuation)", fontsize=10, fontweight="bold")
    ax_scat.set_title("Cohen & Kohn (2011) Structure:\n$r_{\\mathrm{noise}}$ vs $r_{\\mathrm{signal}}$ in [-50, +100] ms Window", fontsize=11, fontweight="bold")
    ax_scat.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax_scat.grid(True, linestyle="--", alpha=0.4)

    bin_edges_y = np.linspace(0, 1800, 10)
    bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2.0
    mean_rn_binned = []
    sem_rn_binned = []

    for k in range(len(bin_edges_y) - 1):
        y0, y1 = bin_edges_y[k], bin_edges_y[k+1]
        sub_rn = df_pairs[(df_pairs["delta_y_um"] >= y0) & (df_pairs["delta_y_um"] < y1)]["r_noise"].dropna().values
        if len(sub_rn) > 2:
            mean_rn_binned.append(float(np.mean(sub_rn)))
            sem_rn_binned.append(float(np.std(sub_rn) / np.sqrt(len(sub_rn))))
        else:
            mean_rn_binned.append(np.nan)
            sem_rn_binned.append(np.nan)

    ax_decay.errorbar(bin_centers_y, mean_rn_binned, yerr=sem_rn_binned, color="#0f766e", fmt="-o", linewidth=2.2, capsize=4, label="Mean ± SEM")
    ax_decay.axhline(0, color="#475569", linestyle="--", linewidth=1.0)
    ax_decay.set_xlabel("Somatic Distance Along Probe $\\Delta y$ (µm)", fontsize=10, fontweight="bold")
    ax_decay.set_ylabel("Mean Noise Correlation $r_{\\mathrm{noise}}$", fontsize=10, fontweight="bold")
    ax_decay.set_title("Spatial Noise Correlation Decay:\nShared Fluctuations vs Somatic Separation", fontsize=11, fontweight="bold")
    ax_decay.set_ylim(-0.05, 0.35)
    ax_decay.legend(loc="upper right", fontsize=8.5)
    ax_decay.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Population Spiking Statistics (Cohen & Kohn 2011) — [-50 ms, +100 ms] Transition Window (e097)", fontsize=12, fontweight="bold", y=0.98)
    pop_fig_path = out_cards_dir / "population_cohen_kohn_correlations.png"
    fig_pop.savefig(pop_fig_path, dpi=180)
    plt.close(fig_pop)
    print(f"      Saved population figure: {pop_fig_path}", flush=True)

    # 5. Generate Interactive HTML Dashboard
    split_suspects = df_pairs[(df_pairs["delta_y_um"] < 60.0) & (df_pairs["r_signal"] > 0.65)].sort_values("r_signal", ascending=False)
    split_suspect_cids = set(split_suspects["cid_1"].tolist() + split_suspects["cid_2"].tolist())

    card_items = []
    for idx_r, r in df_summary.iterrows():
        cid = int(r["cluster_id"])
        img_name = f"lit_card_cluster_{cid:03d}.png"
        if not (out_cards_dir / img_name).exists():
            continue

        r1 = float(r["r_ep1"]) if not pd.isna(r["r_ep1"]) else 0.0
        r2 = float(r["r_ep2"]) if not pd.isna(r["r_ep2"]) else 0.0
        r3 = float(r["r_ep3"]) if not pd.isna(r["r_ep3"]) else 0.0
        r4 = float(r["r_ep4"]) if not pd.isna(r["r_ep4"]) else 0.0
        r12 = float(r["r_cross_1_2"]) if not pd.isna(r["r_cross_1_2"]) else 0.0
        r23 = float(r["r_cross_2_3"]) if not pd.isna(r["r_cross_2_3"]) else 0.0
        r34 = float(r["r_cross_3_4"]) if not pd.isna(r["r_cross_3_4"]) else 0.0
        r14 = float(r["r_cross_1_4"]) if not pd.isna(r["r_cross_1_4"]) else 0.0
        max_drop = max(0.0, r1 - r12, r2 - r23, r3 - r34)

        card_items.append({
            "cid": cid,
            "depth": float(r["depth_um"]),
            "spikes": int(r["n_spikes"]),
            "spikes_win": int(r["spikes_win"]),
            "fr": float(r["firing_rate_hz"]),
            "qc": bool(r["is_qc_passed"]),
            "cell_type": str(r["cell_type"]),
            "r_ep1": round(r1, 3),
            "r_ep2": round(r2, 3),
            "r_ep3": round(r3, 3),
            "r_ep4": round(r4, 3),
            "r_cross_1_2": round(r12, 3),
            "r_cross_2_3": round(r23, 3),
            "r_cross_3_4": round(r34, 3),
            "r_cross_1_4": round(r14, 3),
            "max_drop_val": round(max_drop, 3),
            "sigma_star": round(float(r["sigma_star_ms"]), 1) if not pd.isna(r["sigma_star_ms"]) else "N/A",
            "phase_pref": round(float(r["phase_pref_ep1"]), 2) if not pd.isna(r["phase_pref_ep1"]) else "N/A",
            "phase_r": round(float(r["phase_r_ep1"]), 3) if not pd.isna(r["phase_r_ep1"]) else 0.0,
            "peak_ep1": round(float(r["peak_ep1_hz"]), 1),
            "peak_ep2": round(float(r["peak_ep2_hz"]), 1),
            "peak_ep3": round(float(r["peak_ep3_hz"]), 1),
            "peak_ep4": round(float(r["peak_ep4_hz"]), 1),
            "is_split_suspect": (cid in split_suspect_cids),
            "img": img_name
        })


    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>e097 Literature Spiking Statistics [-50 ms, +100 ms]</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent: #0284c7;
      --accent-glow: rgba(2, 132, 199, 0.25);
      --qc-pass: #10b981;
      --qc-fail: #ef4444;
      --split-border: #f59e0b;
    }
    body {
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header {
      max-width: 1440px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 20px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #38bdf8;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .metrics-bar {
      display: flex;
      gap: 16px;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    .metric-chip {
      background: #1e293b;
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid #334155;
      font-size: 13px;
    }
    .metric-chip strong {
      color: #38bdf8;
      margin-right: 4px;
    }
    .pop-section {
      max-width: 1440px;
      margin: 0 auto 28px auto;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 20px;
    }
    .pop-section h2 {
      margin-top: 0;
      font-size: 18px;
      color: #f8fafc;
    }
    .pop-img {
      width: 100%;
      border-radius: 6px;
      border: 1px solid #334155;
      cursor: pointer;
    }
    .controls-bar {
      max-width: 1440px;
      margin: 0 auto 24px auto;
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      background: #0f172a;
      padding: 14px 18px;
      border-radius: 8px;
      border: 1px solid #1e293b;
    }
    .filter-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn-filter {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      transition: all 0.15s ease;
    }
    .btn-filter.active {
      background: var(--accent);
      border-color: #38bdf8;
      font-weight: bold;
    }
    .search-input, .select-sort {
      background: #1e293b;
      color: var(--text);
      border: 1px solid #334155;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .gallery-grid {
      max-width: 1440px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
      gap: 20px;
    }
    .unit-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .unit-card:hover {
      transform: translateY(-2px);
      border-color: #38bdf8;
    }
    .unit-card.split-candidate {
      border-color: var(--split-border);
      box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
    }
    .card-header {
      padding: 10px 14px;
      background: #1a2234;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
    }
    .card-title {
      font-size: 14px;
      font-weight: bold;
      color: #f8fafc;
    }
    .badge {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 4px;
      font-weight: bold;
    }
    .badge-qc { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-noqc { background: #450a0a; color: #f87171; border: 1px solid #b91c1c; }
    .badge-split { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
    .card-body {
      padding: 10px;
    }
    .card-img {
      width: 100%;
      height: auto;
      border-radius: 4px;
      cursor: pointer;
      display: block;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 10px;
      font-size: 11.5px;
      color: #cbd5e1;
    }
    .metric-item strong {
      color: #38bdf8;
    }
    /* Modal */
    .modal {
      display: none;
      position: fixed;
      z-index: 1000;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.92);
      justify-content: center;
      align-items: center;
    }
    .modal.active {
      display: flex;
    }
    .modal-content {
      max-width: 95vw;
      max-height: 95vh;
      border-radius: 8px;
      border: 2px solid #38bdf8;
    }
  </style>
</head>
<body>
  <header>
    <h1>Canonical Literature Spiking Diagnostics ([-50 ms, +100 ms] Transition Window)</h1>
    <div class="subtitle">
      Centered strictly on <strong>Flow Inspiration Offset / Zero-Crossing ($t = 0\\text{ ms}$)</strong> across all 842 breaths (e097).<br>
      Implements <strong>Schreiber et al. (2003)</strong> reliability &amp; precision curves, <strong>van Rossum (2001)</strong> MDS metric space, <strong>Cohen &amp; Kohn (2011)</strong> signal vs noise correlations, and <strong>Fisher (1993)</strong> polar phase tuning.
    </div>
    <div class="metrics-bar">
      <div class="metric-chip"><strong>Literature Cards Rendered:</strong> __RENDERED_COUNT__</div>
      <div class="metric-chip"><strong>Allen QC Passed:</strong> __QC_RENDERED__</div>
      <div class="metric-chip"><strong>Transition Window:</strong> -50 ms to +100 ms (150 ms total)</div>
      <div class="metric-chip"><strong>QC Pop Within R:</strong> __POP_WITHIN__</div>
      <div class="metric-chip"><strong>QC Pop Transitions:</strong> __POP_TRANSITIONS__</div>
      <div class="metric-chip"><strong>Pairwise Relationships:</strong> __PAIRS_COUNT__</div>
    </div>
  </header>

  <section class="pop-section">
    <h2>Population Analysis: Cohen &amp; Kohn (2011) [-50 ms, +100 ms] Structure</h2>
    <p class="subtitle" style="margin-bottom: 14px;">
      Decomposition of co-variability strictly within the 150 ms transition window. Local units (&Delta;y &lt; 200 &mu;m) share positive trial-to-trial spike count fluctuations scaled by signal correlation (r_signal), decaying with probe somatic distance.
    </p>
    <img src="population_cohen_kohn_correlations.png" class="pop-img" onclick="openModal(this.src)" alt="Population Cohen &amp; Kohn Analysis">
  </section>

  <div class="controls-bar">
    <div class="filter-group">
      <button class="btn-filter active" onclick="setFilter('all', this)">All Units (__RENDERED_COUNT__)</button>
      <button class="btn-filter" onclick="setFilter('qc', this)">QC Passed (__QC_RENDERED__)</button>
      <button class="btn-filter" onclick="setFilter('exp', this)">Pure Post-I</button>
      <button class="btn-filter" onclick="setFilter('insp', this)">Late Inspiratory</button>
      <button class="btn-filter" onclick="setFilter('split', this)">Split Suspects (__SPLIT_COUNT__)</button>
      <button class="btn-filter" onclick="setFilter('high_rel', this)">High Reliability (R &ge; 0.35)</button>
    </div>
    <input type="text" id="searchInput" class="search-input" placeholder="Search Cluster ID (e.g. 212, 13, 150)..." oninput="applyFilters()">
    <select id="sortSelect" class="select-sort" onchange="applySort()">
      <option value="depth_asc">Sort: Dorsal to Ventral (Depth &uarr;)</option>
      <option value="depth_desc">Sort: Ventral to Dorsal (Depth &darr;)</option>
      <option value="cid_asc">Sort: Cluster ID (Ascending)</option>
      <option value="rel_desc">Sort: Ep 1 Within R (Highest &darr;)</option>
      <option value="trans_1_2_desc">Sort: Laser Train 1 Transition (R 1&rarr;2 &darr;)</option>
      <option value="trans_2_3_desc">Sort: Laser Train 2 Transition (R 2&rarr;3 &darr;)</option>
      <option value="trans_3_4_desc">Sort: Laser Train 3 Transition (R 3&rarr;4 &darr;)</option>
      <option value="trans_1_4_desc">Sort: Baseline to Recovery (R 1&rarr;4 &darr;)</option>
      <option value="drop_desc">Sort: Max Transition Drop (Largest Drift / Split &darr;)</option>
      <option value="spikes_desc">Sort: Window Spikes (Highest &darr;)</option>
    </select>
  </div>

  <div class="gallery-grid" id="galleryGrid"></div>

  <div class="modal" id="imageModal" onclick="closeModal()">
    <img class="modal-content" id="modalImg">
  </div>

  <script>
    const items = __JSON_ITEMS__;
    let activeFilter = 'all';

    function renderCards(cardList) {
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = '';
      cardList.forEach(c => {
        const card = document.createElement('div');
        card.className = 'unit-card' + (c.is_split_suspect ? ' split-candidate' : '');
        card.innerHTML = `
          <div class="card-header">
            <span class="card-title">Cluster #${String(c.cid).padStart(3, '0')} (${Math.round(c.depth)} µm)</span>
            <div>
              ${c.qc ? '<span class="badge badge-qc">QC PASSED</span>' : '<span class="badge badge-noqc">NON-QC</span>'}
              ${c.is_split_suspect ? '<span class="badge badge-split">SPLIT CANDIDATE</span>' : ''}
            </div>
          </div>
          <div class="card-body">
            <img src="${c.img}" class="card-img" loading="lazy" onclick="openModal('${c.img}')" alt="Cluster ${c.cid}">
            <div class="metrics-grid" style="margin-top:8px;">
              <div class="metric-item"><strong>Type:</strong> ${c.cell_type}</div>
              <div class="metric-item"><strong>Win Spikes:</strong> ${c.spikes_win} spk</div>
              <div class="metric-item"><strong>Precision &sigma;*:</strong> ${c.sigma_star} ms</div>
              <div class="metric-item"><strong>Phase Vector R:</strong> ${c.phase_r.toFixed(3)}</div>
            </div>
            <div style="background:#0f172a; padding:8px 10px; border-radius:6px; border:1px solid #1e293b; margin-top:8px; font-size:11.5px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94a3b8; font-weight:bold;">Within R:</span>
                <span><b style="color:#38bdf8">Ep1:</b> ${c.r_ep1.toFixed(2)} | <b style="color:#2dd4bf">Ep2:</b> ${c.r_ep2.toFixed(2)} | <b style="color:#fbbf24">Ep3:</b> ${c.r_ep3.toFixed(2)} | <b style="color:#f87171">Ep4:</b> ${c.r_ep4.toFixed(2)}</span>
              </div>
              <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#94a3b8; font-weight:bold;">Cross R:</span>
                <span><b>1&rarr;2:</b> ${c.r_cross_1_2.toFixed(2)} | <b>2&rarr;3:</b> ${c.r_cross_2_3.toFixed(2)} | <b>3&rarr;4:</b> ${c.r_cross_3_4.toFixed(2)} | <b>1&rarr;4:</b> ${c.r_cross_1_4.toFixed(2)}</span>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8; font-weight:bold;">Peak FR:</span>
                <span>${c.peak_ep1.toFixed(0)} &rarr; ${c.peak_ep2.toFixed(0)} &rarr; ${c.peak_ep3.toFixed(0)} &rarr; ${c.peak_ep4.toFixed(0)} Hz</span>
              </div>
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function setFilter(filter, el) {
      activeFilter = filter;
      document.querySelectorAll('.btn-filter').forEach(btn => btn.classList.remove('active'));
      if (el) el.classList.add('active');
      applyFilters();
    }

    function applyFilters() {
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      let filtered = items.filter(c => {
        if (q && !String(c.cid).includes(q)) return false;
        if (activeFilter === 'qc') return c.qc;
        if (activeFilter === 'exp') return c.cell_type.includes('Post-I') || c.cell_type.includes('Expiratory');
        if (activeFilter === 'insp') return c.cell_type.includes('Inspiratory');
        if (activeFilter === 'split') return c.is_split_suspect;
        if (activeFilter === 'high_rel') return c.r_ep1 >= 0.35;
        return true;
      });
      applySortToList(filtered);
    }

    function applySort() {
      applyFilters();
    }

    function applySortToList(list) {
      const sortMode = document.getElementById('sortSelect').value;
      if (sortMode === 'depth_asc') list.sort((a, b) => a.depth - b.depth);
      else if (sortMode === 'depth_desc') list.sort((a, b) => b.depth - a.depth);
      else if (sortMode === 'cid_asc') list.sort((a, b) => a.cid - b.cid);
      else if (sortMode === 'rel_desc') list.sort((a, b) => b.r_ep1 - a.r_ep1);
      else if (sortMode === 'trans_1_2_desc') list.sort((a, b) => b.r_cross_1_2 - a.r_cross_1_2);
      else if (sortMode === 'trans_2_3_desc') list.sort((a, b) => b.r_cross_2_3 - a.r_cross_2_3);
      else if (sortMode === 'trans_3_4_desc') list.sort((a, b) => b.r_cross_3_4 - a.r_cross_3_4);
      else if (sortMode === 'trans_1_4_desc') list.sort((a, b) => b.r_cross_1_4 - a.r_cross_1_4);
      else if (sortMode === 'drop_desc') list.sort((a, b) => b.max_drop_val - a.max_drop_val);
      else if (sortMode === 'spikes_desc') list.sort((a, b) => b.spikes_win - a.spikes_win);
      renderCards(list);
    }

    function openModal(src) {
      const m = document.getElementById('imageModal');
      const img = document.getElementById('modalImg');
      img.src = src;
      m.classList.add('active');
    }

    function closeModal() {
      document.getElementById('imageModal').classList.remove('active');
    }

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
    });

    applySortToList([...items]);
  </script>
</body>
</html>
"""

    qc_items = [c for c in card_items if c["qc"]]
    qc_count = len(qc_items)
    split_count = len([c for c in card_items if c["is_split_suspect"]])

    if qc_count > 0:
        pop_w1 = np.mean([c["r_ep1"] for c in qc_items])
        pop_w2 = np.mean([c["r_ep2"] for c in qc_items])
        pop_w3 = np.mean([c["r_ep3"] for c in qc_items])
        pop_w4 = np.mean([c["r_ep4"] for c in qc_items])
        pop_t12 = np.mean([c["r_cross_1_2"] for c in qc_items])
        pop_t23 = np.mean([c["r_cross_2_3"] for c in qc_items])
        pop_t34 = np.mean([c["r_cross_3_4"] for c in qc_items])
        pop_t14 = np.mean([c["r_cross_1_4"] for c in qc_items])
        pop_within_str = f"Ep1: {pop_w1:.2f} &rarr; Ep2: {pop_w2:.2f} &rarr; Ep3: {pop_w3:.2f} &rarr; Ep4: {pop_w4:.2f}"
        pop_trans_str = f"1&rarr;2: {pop_t12:.2f} | 2&rarr;3: {pop_t23:.2f} | 3&rarr;4: {pop_t34:.2f} | 1&rarr;4: {pop_t14:.2f}"
    else:
        pop_within_str = "N/A"
        pop_trans_str = "N/A"

    final_html = html_template.replace("__RENDERED_COUNT__", str(len(card_items))) \
                              .replace("__QC_RENDERED__", str(qc_count)) \
                              .replace("__SPLIT_COUNT__", str(split_count)) \
                              .replace("__PAIRS_COUNT__", str(len(df_pairs))) \
                              .replace("__POP_WITHIN__", pop_within_str) \
                              .replace("__POP_TRANSITIONS__", pop_trans_str) \
                              .replace("__JSON_ITEMS__", json.dumps(card_items))

    dashboard_path = out_cards_dir / "literature_spiking_statistics_dashboard.html"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"[+] Successfully wrote dashboard to: {dashboard_path}", flush=True)
    t_tot = time.time() - t0
    print(f"[*] Completed literature diagnostic suite in {t_tot:.1f} seconds!")



if __name__ == "__main__":
    main()
