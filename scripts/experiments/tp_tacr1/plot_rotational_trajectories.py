#!/usr/bin/env python3
"""
Population Rotational Trajectories & Inspiration-Off Attractor Dynamics
Strictly implementing the analytical methodology of Bush et al. (2025)
("An inspiration-off attractor supports the robust and flexible control of breathing", bioRxiv 2025.09.23.678177):

Plots INDIVIDUAL BREATH TRAJECTORIES (thin semi-transparent lines) for each group along with
bold group mean trajectories to visualize trial-to-trial variability and phase-space divergence.

1. 5 ms time binning (200 Hz sampling rate).
2. Gaussian kernel smoothing (sigma = 10 ms).
3. Square-root transformation (sqrt(FR)) prior to PCA fitting.
4. Continuous respiratory phase mapping phi in [-pi, pi] from Diaphragm EMG onsets and offsets.
5. Mazurek vector mean unit classification (Tonic vs Phasic Inspiratory / Expiratory).
6. Latent trajectory speed calculation ||dPC/dt||.
7. Inspiration-Off Attractor Target (I_off) identification and Euclidean distance d(t, I_off) metric.
8. Individual breath plotting across 4 experimental epochs: Baseline (0-1m), Laser Perturbation, Non-Laser Protocol, and Post-Laser Recovery (+3-4m).
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    base_session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1")
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_rotational_dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = base_session_dir / "alf_session"
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Computing Individual Breath Rotational Dynamics (Bush et al. 2025 Methods)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Kilosort4 spike data & channel geometry
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0  # Cover post-laser recovery up to 524s
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    info_df['firing_rate_550s'] = info_df['cluster_id'].map(lambda u: np.sum(spikes_clusters == u) / MAX_TIME_S)

    # 2. Load NIDQ Diaphragm EMG & Laser TTL for 0-550s
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]
    ch6_laser = nidq_data.all_channels[6][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    emg_env = emg_proc.envelope
    breath_onsets = emg_proc.breath_onsets
    breath_offsets = emg_proc.breath_offsets

    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]

    n_pulses = min(len(laser_onsets), len(laser_offsets))
    laser_onsets = laser_onsets[:n_pulses]
    laser_offsets = laser_offsets[:n_pulses]
    pulse_durations = laser_offsets - laser_onsets

    t_first_laser = laser_onsets[0] if len(laser_onsets) > 0 else 60.0
    t_last_laser = laser_offsets[-1] if len(laser_offsets) > 0 else 284.0
    t_post_rec_start = t_last_laser + 180.0
    t_post_rec_end = t_post_rec_start + 60.0

    # 3. Compute Continuous Respiratory Phase phi(t) in [-pi, pi] (Bush et al. 2025 Page 21-22)
    dt = 0.005 # 5 ms time bin size (200 Hz rate)
    t_bins = np.arange(0.0, MAX_TIME_S, dt)
    n_bins = len(t_bins)

    phi_vector = np.zeros(n_bins)

    n_cycles = min(len(breath_onsets) - 1, len(breath_offsets))
    for i in range(n_cycles):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i] if breath_offsets[i] > t_on else t_on + 0.15
        t_next = breath_onsets[i + 1]

        # Inspiration phase: [t_on, t_off] -> [0, pi]
        idx_insp = np.where((t_bins >= t_on) & (t_bins < t_off))[0]
        if len(idx_insp) > 0:
            phi_vector[idx_insp] = np.linspace(0, np.pi, len(idx_insp))

        # Expiration phase: [t_off, t_next] -> [-pi, 0)
        idx_exp = np.where((t_bins >= t_off) & (t_bins < t_next))[0]
        if len(idx_exp) > 0:
            phi_vector[idx_exp] = np.linspace(-np.pi, 0.0, len(idx_exp))

    # 4. Deduplicate Good Units & Classification (Mazurek Vector Mean R)
    good_df = info_df[(info_df['group'] == 'good') & (info_df['firing_rate_550s'] >= 0.5)].copy().reset_index(drop=True)
    spiketrains = {u: np.sort(spikes_times[spikes_clusters == u]) for u in good_df['cluster_id']}

    win_s = 0.0005
    discard_uids = set()
    good_uids = list(spiketrains.keys())

    for i in range(len(good_uids)):
        u1 = good_uids[i]
        if u1 in discard_uids: continue
        st1 = spiketrains[u1]
        n1 = len(st1)

        for j in range(i + 1, len(good_uids)):
            u2 = good_uids[j]
            if u2 in discard_uids: continue
            st2 = spiketrains[u2]
            n2 = len(st2)

            left = np.searchsorted(st2, st1 - win_s)
            right = np.searchsorted(st2, st1 + win_s)
            n_coinc = np.sum(right > left)
            if n_coinc > 0:
                if n_coinc / n2 >= 0.50 and n1 > n2:
                    discard_uids.add(u2)
                elif n_coinc / n1 >= 0.50 and n2 >= n1:
                    discard_uids.add(u1)
                    break

    clean_good_df = good_df[~good_df['cluster_id'].isin(discard_uids)].copy().reset_index(drop=True)

    phototagged_flags = []
    laser_ratios = []
    vector_strengths = []

    for idx, row in clean_good_df.iterrows():
        uid = int(row['cluster_id'])
        u_spks = spiketrains[uid]

        l_counts, b_counts, lats = [], [], []
        for p_i in range(n_pulses):
            ton = laser_onsets[p_i]
            toff = laser_offsets[p_i]
            dur = toff - ton

            nl = np.sum((u_spks >= ton) & (u_spks < toff))
            nb = np.sum((u_spks >= ton - dur) & (u_spks < ton))
            l_counts.append(nl)
            b_counts.append(nb)

            spks_after = u_spks[u_spks >= ton]
            if len(spks_after) > 0 and spks_after[0] < toff:
                lats.append((spks_after[0] - ton) * 1000.0)

        l_counts = np.array(l_counts)
        b_counts = np.array(b_counts)
        m_l_rate = np.mean(l_counts) / max(1e-4, np.mean(pulse_durations))
        m_b_rate = np.mean(b_counts) / max(1e-4, np.mean(pulse_durations))
        ratio = (m_l_rate + 0.1) / (m_b_rate + 0.1)

        if np.sum(l_counts != b_counts) > 5:
            _, p_val = stats.wilcoxon(l_counts, b_counts)
        else:
            p_val = 1.0

        mean_lat = np.mean(lats) if len(lats) > 0 else 999.0
        is_tagged = (p_val < 0.01) and (ratio >= 2.5) and (m_l_rate > 3.0) and (mean_lat <= 11.0)

        phototagged_flags.append(is_tagged)
        laser_ratios.append(round(ratio, 2))

        spk_bin_indices = np.searchsorted(t_bins, u_spks) - 1
        spk_bin_indices = spk_bin_indices[(spk_bin_indices >= 0) & (spk_bin_indices < n_bins)]
        spk_phases = phi_vector[spk_bin_indices]

        if len(spk_phases) < 15:
            vector_strengths.append(0.0)
        else:
            cos_m = np.mean(np.cos(spk_phases))
            sin_m = np.mean(np.sin(spk_phases))
            vector_strengths.append(np.sqrt(cos_m**2 + sin_m**2))

    clean_good_df['is_phototagged'] = phototagged_flags
    clean_good_df['laser_ratio'] = laser_ratios
    clean_good_df['vector_strength'] = vector_strengths

    # Select 50 Units (Phototagged + Phasic + Tonic)
    fr_col = 'firing_rate_550s'
    tagged_sub = clean_good_df[clean_good_df['is_phototagged']].sort_values(by='laser_ratio', ascending=False).head(15)
    phasic_sub = clean_good_df[(~clean_good_df['cluster_id'].isin(tagged_sub['cluster_id'])) & (clean_good_df['vector_strength'] >= 0.20)].sort_values(by='vector_strength', ascending=False).head(25)
    tonic_sub = clean_good_df[(~clean_good_df['cluster_id'].isin(tagged_sub['cluster_id'])) & (~clean_good_df['cluster_id'].isin(phasic_sub['cluster_id'])) & (clean_good_df['vector_strength'] < 0.20)].sort_values(by=fr_col, ascending=False).head(10)

    selected_50 = pd.concat([tagged_sub, phasic_sub, tonic_sub]).drop_duplicates(subset=['cluster_id']).reset_index(drop=True)
    selected_50 = selected_50.sort_values(by=['shank', 'depth'], ascending=[True, True]).reset_index(drop=True)
    selected_uids = selected_50['cluster_id'].values

    print(f"Loaded {len(selected_uids)} selected units for individual breath rotational trajectory analysis.")

    # 5. Build Time-Binned Matrix, Gaussian Smoothing (sigma=10ms) & Square-Root Transformation (sqrt(FR))
    sigma_ms = 0.010
    sigma_bins = sigma_ms / dt
    n_units = len(selected_uids)

    rate_matrix = np.zeros((n_units, n_bins))

    for i, uid in enumerate(selected_uids):
        u_spks = spiketrains[uid]
        counts, _ = np.histogram(u_spks, bins=t_bins)
        rate = counts / dt
        rate_smooth = gaussian_filter1d(rate, sigma=sigma_bins)
        rate_matrix[i, :-1] = rate_smooth

    # Square-Root Transformation: sqrt(FR) (Bush et al. Page 22)
    sqrt_matrix = np.sqrt(np.maximum(0.0, rate_matrix)) # (N_units, N_bins)

    # Transpose for PCA (N_bins x N_units)
    X_pop = sqrt_matrix.T

    # 6. Fit Unified PCA Space
    pca = PCA(n_components=5)
    proj = pca.fit_transform(X_pop) # (N_bins, 5)
    exp_var = pca.explained_variance_ratio_

    # 7. Compute Trajectory Speed ||dPC/dt||
    diff_proj = np.diff(proj[:, :3], axis=0)
    speed_vector = np.concatenate([[0.0], np.sqrt(np.sum(diff_proj**2, axis=1)) / dt])

    # 8. Identify Inspiration-Off Attractor Target (I_off)
    mask_baseline = (t_bins < t_first_laser)
    mask_ioff = mask_baseline & (phi_vector >= -np.pi) & (phi_vector < (-np.pi + np.pi / 25.0))
    if np.sum(mask_ioff) > 0:
        I_off_target = np.mean(proj[mask_ioff, :3], axis=0)
    else:
        I_off_target = np.mean(proj[mask_baseline, :3], axis=0)

    dist_to_ioff = np.sqrt(np.sum((proj[:, :3] - I_off_target)**2, axis=1))

    # 9. Extract Individual Trajectories for the 4 Experimental Epochs
    pre_win_s = 0.15
    post_win_s = 0.65
    n_pre_bins = int(pre_win_s / dt)
    n_post_bins = int(post_win_s / dt)
    n_epoch_bins = n_pre_bins + n_post_bins
    t_epoch = np.linspace(-pre_win_s, post_win_s, n_epoch_bins)

    def extract_individual_epoch_data(onsets_list):
        trajs_list, speeds_list, dists_list = [], [], []
        for onset in onsets_list:
            idx_onset = int(onset / dt)
            i_start = idx_onset - n_pre_bins
            i_end = idx_onset + n_post_bins
            if 0 <= i_start and i_end <= n_bins:
                trajs_list.append(proj[i_start:i_end, :3])
                speeds_list.append(speed_vector[i_start:i_end])
                dists_list.append(dist_to_ioff[i_start:i_end])
        return trajs_list, speeds_list, dists_list

    # Epoch 1: Baseline Breaths (0.0 - 60.55 s)
    baseline_onsets = [o for o in breath_onsets if o < t_first_laser]
    base_trajs, base_speeds, base_dists = extract_individual_epoch_data(baseline_onsets)

    # Epoch 2: Laser Perturbation Trajectories (60.55 - 284.00 s)
    laser_trajs, laser_speeds, laser_dists = [], [], []
    for l_on in laser_onsets:
        idx_lon = int(l_on / dt)
        i_start = idx_lon - n_pre_bins
        i_end = idx_lon + n_post_bins
        if 0 <= i_start and i_end <= n_bins:
            laser_trajs.append(proj[i_start:i_end, :3])
            laser_speeds.append(speed_vector[i_start:i_end])
            laser_dists.append(dist_to_ioff[i_start:i_end])

    # Epoch 3: Non-Laser Breaths during Protocol (60.55 - 284.00 s)
    non_laser_proto_onsets = [o for o in breath_onsets if t_first_laser <= o <= t_last_laser and np.min(np.abs(o - laser_onsets)) > 0.35]
    non_laser_trajs, non_laser_speeds, non_laser_dists = extract_individual_epoch_data(non_laser_proto_onsets)

    # Epoch 4: Post-Laser Recovery Breaths (464.00 - 524.00 s)
    post_rec_onsets = [o for o in breath_onsets if t_post_rec_start <= o <= t_post_rec_end]
    post_rec_trajs, post_rec_speeds, post_rec_dists = extract_individual_epoch_data(post_rec_onsets)

    # Calculate Group Means
    mean_traj_base = np.mean(base_trajs, axis=0) if len(base_trajs)>0 else np.zeros((n_epoch_bins,3))
    mean_traj_laser = np.mean(laser_trajs, axis=0) if len(laser_trajs)>0 else np.zeros((n_epoch_bins,3))
    mean_traj_non_laser = np.mean(non_laser_trajs, axis=0) if len(non_laser_trajs)>0 else np.zeros((n_epoch_bins,3))
    mean_traj_post_rec = np.mean(post_rec_trajs, axis=0) if len(post_rec_trajs)>0 else np.zeros((n_epoch_bins,3))

    mean_speed_base = np.mean(base_speeds, axis=0) if len(base_speeds)>0 else np.zeros(n_epoch_bins)
    mean_speed_laser = np.mean(laser_speeds, axis=0) if len(laser_speeds)>0 else np.zeros(n_epoch_bins)
    mean_speed_non_laser = np.mean(non_laser_speeds, axis=0) if len(non_laser_speeds)>0 else np.zeros(n_epoch_bins)
    mean_speed_post_rec = np.mean(post_rec_speeds, axis=0) if len(post_rec_speeds)>0 else np.zeros(n_epoch_bins)

    mean_dist_base = np.mean(base_dists, axis=0) if len(base_dists)>0 else np.zeros(n_epoch_bins)
    mean_dist_laser = np.mean(laser_dists, axis=0) if len(laser_dists)>0 else np.zeros(n_epoch_bins)
    mean_dist_non_laser = np.mean(non_laser_dists, axis=0) if len(non_laser_dists)>0 else np.zeros(n_epoch_bins)
    mean_dist_post_rec = np.mean(post_rec_dists, axis=0) if len(post_rec_dists)>0 else np.zeros(n_epoch_bins)

    print(f"Extracted Individual Trajectories:")
    print(f" -> Baseline: {len(base_trajs)} individual breaths")
    print(f" -> Laser Perturbation: {len(laser_trajs)} individual laser pulses")
    print(f" -> Non-Laser Protocol: {len(non_laser_trajs)} individual breaths")
    print(f" -> Post-Laser Recovery: {len(post_rec_trajs)} individual breaths")

    # ---------------------------------------------------------
    # 10. Interactive 3D & 2D Plotly Dashboard (Individual Breaths)
    # ---------------------------------------------------------
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]
        ],
        subplot_titles=(
            f"1. 3D Individual Rotational Orbits (PC1-3) [Var: {np.sum(exp_var[:3])*100:.1f}%]",
            "2. 2D Individual Rotational Loops (PC1 vs PC2)",
            "3. 2D Individual Rotational Loops (PC2 vs PC3)",
            "4. Latent Trajectory Speed ||dPC/dt||",
            "5. Distance to Inspiration-Off Attractor d(t, I_off)",
            "6. PC1 Score Temporal Dynamics"
        ),
        horizontal_spacing=0.07,
        vertical_spacing=0.10
    )

    c_base = 'rgba(31, 119, 180, 0.25)'      # Blue transparent
    c_laser = 'rgba(227, 119, 194, 0.35)'     # Pink/Magenta transparent
    c_non_laser = 'rgba(44, 160, 44, 0.25)'  # Green transparent
    c_post_rec = 'rgba(255, 127, 14, 0.25)'  # Orange transparent

    c_base_bold = '#1f77b4'
    c_laser_bold = '#e377c2'
    c_non_laser_bold = '#2ca02c'
    c_post_rec_bold = '#ff7f0e'

    # Plot Individual Breaths on 3D Subplot
    for tr in base_trajs:
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode='lines', line=dict(color=c_base, width=2), showlegend=False), row=1, col=1)
    for tr in non_laser_trajs:
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode='lines', line=dict(color=c_non_laser, width=2), showlegend=False), row=1, col=1)
    for tr in post_rec_trajs:
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode='lines', line=dict(color=c_post_rec, width=2), showlegend=False), row=1, col=1)
    for tr in laser_trajs:
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode='lines', line=dict(color=c_laser, width=2.5), showlegend=False), row=1, col=1)

    # Plot Bold Group Means on 3D Subplot
    fig.add_trace(go.Scatter3d(x=mean_traj_base[:, 0], y=mean_traj_base[:, 1], z=mean_traj_base[:, 2], mode='lines', name=f'Baseline Mean (N={len(base_trajs)})', line=dict(color=c_base_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=mean_traj_laser[:, 0], y=mean_traj_laser[:, 1], z=mean_traj_laser[:, 2], mode='lines', name=f'Laser Perturb Mean (N={len(laser_trajs)})', line=dict(color=c_laser_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=mean_traj_non_laser[:, 0], y=mean_traj_non_laser[:, 1], z=mean_traj_non_laser[:, 2], mode='lines', name=f'Non-Laser Proto Mean (N={len(non_laser_trajs)})', line=dict(color=c_non_laser_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=mean_traj_post_rec[:, 0], y=mean_traj_post_rec[:, 1], z=mean_traj_post_rec[:, 2], mode='lines', name=f'Post-Laser Rec Mean (N={len(post_rec_trajs)})', line=dict(color=c_post_rec_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[I_off_target[0]], y=[I_off_target[1]], z=[I_off_target[2]], mode='markers', name='⭐ Inspiration-Off Target (I_off)', marker=dict(size=10, color='red')), row=1, col=1)

    # Subplot 2: 2D PC1 vs PC2 (Individual Breaths + Means)
    for tr in base_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=1, col=2)
    for tr in non_laser_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=1, col=2)
    for tr in post_rec_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=1, col=2)
    for tr in laser_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=mean_traj_base[:, 0], y=mean_traj_base[:, 1], mode='lines', name='Baseline Mean', line=dict(color=c_base_bold, width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_traj_laser[:, 0], y=mean_traj_laser[:, 1], mode='lines', name='Laser Perturb Mean', line=dict(color=c_laser_bold, width=4, dash='dash')), row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_traj_non_laser[:, 0], y=mean_traj_non_laser[:, 1], mode='lines', name='Non-Laser Proto Mean', line=dict(color=c_non_laser_bold, width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_traj_post_rec[:, 0], y=mean_traj_post_rec[:, 1], mode='lines', name='Post-Laser Rec Mean', line=dict(color=c_post_rec_bold, width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[I_off_target[0]], y=[I_off_target[1]], mode='markers', name='I_off Target', marker=dict(size=12, color='red', symbol='star')), row=1, col=2)

    # Subplot 3: 2D PC2 vs PC3
    for tr in base_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 1], y=tr[:, 2], mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=1, col=3)
    for tr in non_laser_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 1], y=tr[:, 2], mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=1, col=3)
    for tr in post_rec_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 1], y=tr[:, 2], mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=1, col=3)
    for tr in laser_trajs:
        fig.add_trace(go.Scatter(x=tr[:, 1], y=tr[:, 2], mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=1, col=3)

    fig.add_trace(go.Scatter(x=mean_traj_base[:, 1], y=mean_traj_base[:, 2], mode='lines', line=dict(color=c_base_bold, width=4), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=mean_traj_laser[:, 1], y=mean_traj_laser[:, 2], mode='lines', line=dict(color=c_laser_bold, width=4, dash='dash'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=mean_traj_non_laser[:, 1], y=mean_traj_non_laser[:, 2], mode='lines', line=dict(color=c_non_laser_bold, width=4), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=mean_traj_post_rec[:, 1], y=mean_traj_post_rec[:, 2], mode='lines', line=dict(color=c_post_rec_bold, width=4), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[I_off_target[1]], y=[I_off_target[2]], mode='markers', marker=dict(size=12, color='red', symbol='star'), showlegend=False), row=1, col=3)

    # Subplot 4: Trajectory Speed ||dPC/dt|| (Individual + Mean)
    for sp in base_speeds:
        fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=2, col=1)
    for sp in non_laser_speeds:
        fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=2, col=1)
    for sp in post_rec_speeds:
        fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=2, col=1)
    for sp in laser_speeds:
        fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_base, mode='lines', line=dict(color=c_base_bold, width=3.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_laser, mode='lines', line=dict(color=c_laser_bold, width=3.5, dash='dash'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_non_laser, mode='lines', line=dict(color=c_non_laser_bold, width=3.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_post_rec, mode='lines', line=dict(color=c_post_rec_bold, width=3.5), showlegend=False), row=2, col=1)

    # Subplot 5: Distance to I_off Attractor d(t, I_off)
    for ds in base_dists:
        fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=2, col=2)
    for ds in non_laser_dists:
        fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=2, col=2)
    for ds in post_rec_dists:
        fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=2, col=2)
    for ds in laser_dists:
        fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=2, col=2)

    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_base, mode='lines', line=dict(color=c_base_bold, width=3.5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_laser, mode='lines', line=dict(color=c_laser_bold, width=3.5, dash='dash'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_non_laser, mode='lines', line=dict(color=c_non_laser_bold, width=3.5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_post_rec, mode='lines', line=dict(color=c_post_rec_bold, width=3.5), showlegend=False), row=2, col=2)

    # Subplot 6: PC1 Score Timecourse
    for tr in base_trajs:
        fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=2, col=3)
    for tr in non_laser_trajs:
        fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=2, col=3)
    for tr in post_rec_trajs:
        fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=2, col=3)
    for tr in laser_trajs:
        fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=2, col=3)

    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_base[:, 0], mode='lines', line=dict(color=c_base_bold, width=3.5), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_laser[:, 0], mode='lines', line=dict(color=c_laser_bold, width=3.5, dash='dash'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_non_laser[:, 0], mode='lines', line=dict(color=c_non_laser_bold, width=3.5), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_post_rec[:, 0], mode='lines', line=dict(color=c_post_rec_bold, width=3.5), showlegend=False), row=2, col=3)

    fig.update_layout(
        title="<b>Subject TP_Tacr1_1: Individual Breath Rotational Dynamics & Inspiration-Off Attractor Target</b>",
        template="plotly_white",
        height=1000,
        margin=dict(l=50, r=40, t=80, b=50),
        xaxis2=dict(title="PC1 Score"), yaxis2=dict(title="PC2 Score"),
        xaxis3=dict(title="PC2 Score"), yaxis3=dict(title="PC3 Score"),
        xaxis4=dict(title="Time relative to Insp Onset (s)"), yaxis4=dict(title="Trajectory Speed ||dPC/dt||"),
        xaxis5=dict(title="Time relative to Insp Onset (s)"), yaxis5=dict(title="Distance to I-Off Target d(t, I_off)"),
        xaxis6=dict(title="Time relative to Insp Onset (s)"), yaxis6=dict(title="PC1 Score")
    )

    html_out = out_dir / "population_rotational_trajectories.html"
    fig.write_html(str(html_out))
    print(f"\n -> Saved Interactive Individual Breath HTML Dashboard: {html_out}")

    # ---------------------------------------------------------
    # 11. Create High-Resolution Static PNG Figure (Matplotlib)
    # ---------------------------------------------------------
    print("Generating high-resolution static PNG figure (Individual Breaths + Group Means)...")
    fig_mpl = plt.figure(figsize=(20, 11))

    # 1. 3D Subplot
    ax3d = fig_mpl.add_subplot(2, 3, 1, projection='3d')
    for tr in base_trajs:
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_base_bold, alpha=0.15, linewidth=0.8)
    for tr in non_laser_trajs:
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_non_laser_bold, alpha=0.15, linewidth=0.8)
    for tr in post_rec_trajs:
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_post_rec_bold, alpha=0.15, linewidth=0.8)
    for tr in laser_trajs:
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_laser_bold, alpha=0.25, linewidth=1.0)

    ax3d.plot(mean_traj_base[:, 0], mean_traj_base[:, 1], mean_traj_base[:, 2], color=c_base_bold, linewidth=3.5, label=f'Baseline Mean (N={len(base_trajs)})')
    ax3d.plot(mean_traj_laser[:, 0], mean_traj_laser[:, 1], mean_traj_laser[:, 2], color=c_laser_bold, linestyle='--', linewidth=3.5, label=f'Laser Perturb Mean (N={len(laser_trajs)})')
    ax3d.plot(mean_traj_non_laser[:, 0], mean_traj_non_laser[:, 1], mean_traj_non_laser[:, 2], color=c_non_laser_bold, linewidth=3, label=f'Non-Laser Proto Mean (N={len(non_laser_trajs)})')
    ax3d.plot(mean_traj_post_rec[:, 0], mean_traj_post_rec[:, 1], mean_traj_post_rec[:, 2], color=c_post_rec_bold, linewidth=3, label=f'Post-Laser Rec Mean (N={len(post_rec_trajs)})')
    ax3d.scatter(I_off_target[0], I_off_target[1], I_off_target[2], color='red', s=140, marker='*', zorder=10, label='I_off Attractor Target')

    ax3d.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontweight='bold')
    ax3d.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontweight='bold')
    ax3d.set_zlabel(f"PC3 ({exp_var[2]*100:.1f}%)", fontweight='bold')
    ax3d.set_title(f"3D Individual Rotational Orbits ({np.sum(exp_var[:3])*100:.1f}% Var)", fontsize=11, fontweight='bold')
    ax3d.legend(loc='upper left', fontsize=7.5)

    # 2. 2D PC1 vs PC2
    ax2d_1 = fig_mpl.add_subplot(2, 3, 2)
    for tr in base_trajs:
        ax2d_1.plot(tr[:, 0], tr[:, 1], color=c_base_bold, alpha=0.15, linewidth=0.8)
    for tr in non_laser_trajs:
        ax2d_1.plot(tr[:, 0], tr[:, 1], color=c_non_laser_bold, alpha=0.15, linewidth=0.8)
    for tr in post_rec_trajs:
        ax2d_1.plot(tr[:, 0], tr[:, 1], color=c_post_rec_bold, alpha=0.15, linewidth=0.8)
    for tr in laser_trajs:
        ax2d_1.plot(tr[:, 0], tr[:, 1], color=c_laser_bold, alpha=0.25, linewidth=1.0)

    ax2d_1.plot(mean_traj_base[:, 0], mean_traj_base[:, 1], color=c_base_bold, linewidth=3, label='Baseline Mean')
    ax2d_1.plot(mean_traj_laser[:, 0], mean_traj_laser[:, 1], color=c_laser_bold, linestyle='--', linewidth=3, label='Laser Perturbation Mean')
    ax2d_1.plot(mean_traj_non_laser[:, 0], mean_traj_non_laser[:, 1], color=c_non_laser_bold, linewidth=3, label='Non-Laser Protocol Mean')
    ax2d_1.plot(mean_traj_post_rec[:, 0], mean_traj_post_rec[:, 1], color=c_post_rec_bold, linewidth=3, label='Post-Laser Recovery Mean')
    ax2d_1.scatter(I_off_target[0], I_off_target[1], color='red', s=140, marker='*', zorder=10, label='I_off Target')

    ax2d_1.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontweight='bold')
    ax2d_1.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontweight='bold')
    ax2d_1.set_title("2D Phase-Space Orbit (PC1 vs PC2)", fontsize=11, fontweight='bold')
    ax2d_1.grid(True, linestyle=':', alpha=0.5)
    ax2d_1.legend(loc='upper right', fontsize=8)

    # 3. 2D PC2 vs PC3
    ax2d_2 = fig_mpl.add_subplot(2, 3, 3)
    for tr in base_trajs:
        ax2d_2.plot(tr[:, 1], tr[:, 2], color=c_base_bold, alpha=0.15, linewidth=0.8)
    for tr in non_laser_trajs:
        ax2d_2.plot(tr[:, 1], tr[:, 2], color=c_non_laser_bold, alpha=0.15, linewidth=0.8)
    for tr in post_rec_trajs:
        ax2d_2.plot(tr[:, 1], tr[:, 2], color=c_post_rec_bold, alpha=0.15, linewidth=0.8)
    for tr in laser_trajs:
        ax2d_2.plot(tr[:, 1], tr[:, 2], color=c_laser_bold, alpha=0.25, linewidth=1.0)

    ax2d_2.plot(mean_traj_base[:, 1], mean_traj_base[:, 2], color=c_base_bold, linewidth=3)
    ax2d_2.plot(mean_traj_laser[:, 1], mean_traj_laser[:, 2], color=c_laser_bold, linestyle='--', linewidth=3)
    ax2d_2.plot(mean_traj_non_laser[:, 1], mean_traj_non_laser[:, 2], color=c_non_laser_bold, linewidth=3)
    ax2d_2.plot(mean_traj_post_rec[:, 1], mean_traj_post_rec[:, 2], color=c_post_rec_bold, linewidth=3)
    ax2d_2.scatter(I_off_target[1], I_off_target[2], color='red', s=140, marker='*', zorder=10)

    ax2d_2.set_xlabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontweight='bold')
    ax2d_2.set_ylabel(f"PC3 ({exp_var[2]*100:.1f}%)", fontweight='bold')
    ax2d_2.set_title("2D Phase-Space Orbit (PC2 vs PC3)", fontsize=11, fontweight='bold')
    ax2d_2.grid(True, linestyle=':', alpha=0.5)

    # 4. Latent Trajectory Speed ||dPC/dt||
    ax_spd = fig_mpl.add_subplot(2, 3, 4)
    for sp in base_speeds:
        ax_spd.plot(t_epoch, sp, color=c_base_bold, alpha=0.12, linewidth=0.7)
    for sp in non_laser_speeds:
        ax_spd.plot(t_epoch, sp, color=c_non_laser_bold, alpha=0.12, linewidth=0.7)
    for sp in post_rec_speeds:
        ax_spd.plot(t_epoch, sp, color=c_post_rec_bold, alpha=0.12, linewidth=0.7)
    for sp in laser_speeds:
        ax_spd.plot(t_epoch, sp, color=c_laser_bold, alpha=0.20, linewidth=0.8)

    ax_spd.plot(t_epoch, mean_speed_base, color=c_base_bold, linewidth=2.5, label='Baseline Mean')
    ax_spd.plot(t_epoch, mean_speed_laser, color=c_laser_bold, linestyle='--', linewidth=2.5, label='Laser Perturb Mean')
    ax_spd.plot(t_epoch, mean_speed_non_laser, color=c_non_laser_bold, linewidth=2.5, label='Non-Laser Proto Mean')
    ax_spd.plot(t_epoch, mean_speed_post_rec, color=c_post_rec_bold, linewidth=2.5, label='Post-Laser Rec Mean')
    ax_spd.axvline(0.0, color='gray', linestyle=':', linewidth=1.5)

    ax_spd.set_xlabel("Time relative to Insp Onset (s)", fontweight='bold')
    ax_spd.set_ylabel("Trajectory Speed ||dPC/dt||", fontweight='bold')
    ax_spd.set_title("Latent Trajectory Speed (Individual + Mean)", fontsize=11, fontweight='bold')
    ax_spd.grid(True, linestyle=':', alpha=0.5)
    ax_spd.legend(loc='upper right', fontsize=8)

    # 5. Distance to Inspiration-Off Attractor d(t, I_off)
    ax_dst = fig_mpl.add_subplot(2, 3, 5)
    for ds in base_dists:
        ax_dst.plot(t_epoch, ds, color=c_base_bold, alpha=0.12, linewidth=0.7)
    for ds in non_laser_dists:
        ax_dst.plot(t_epoch, ds, color=c_non_laser_bold, alpha=0.12, linewidth=0.7)
    for ds in post_rec_dists:
        ax_dst.plot(t_epoch, ds, color=c_post_rec_bold, alpha=0.12, linewidth=0.7)
    for ds in laser_dists:
        ax_dst.plot(t_epoch, ds, color=c_laser_bold, alpha=0.20, linewidth=0.8)

    ax_dst.plot(t_epoch, mean_dist_base, color=c_base_bold, linewidth=2.5, label='Baseline Mean')
    ax_dst.plot(t_epoch, mean_dist_laser, color=c_laser_bold, linestyle='--', linewidth=2.5, label='Laser Perturb Mean')
    ax_dst.plot(t_epoch, mean_dist_non_laser, color=c_non_laser_bold, linewidth=2.5, label='Non-Laser Proto Mean')
    ax_dst.plot(t_epoch, mean_dist_post_rec, color=c_post_rec_bold, linewidth=2.5, label='Post-Laser Rec Mean')
    ax_dst.axvline(0.0, color='gray', linestyle=':', linewidth=1.5)

    ax_dst.set_xlabel("Time relative to Insp Onset (s)", fontweight='bold')
    ax_dst.set_ylabel("Distance to I-off Target d(t, I_off)", fontweight='bold')
    ax_dst.set_title("Convergence to I-Off Attractor", fontsize=11, fontweight='bold')
    ax_dst.grid(True, linestyle=':', alpha=0.5)
    ax_dst.legend(loc='upper right', fontsize=8)

    # 6. PC1 Score Dynamics
    ax_pc1 = fig_mpl.add_subplot(2, 3, 6)
    for tr in base_trajs:
        ax_pc1.plot(t_epoch, tr[:, 0], color=c_base_bold, alpha=0.12, linewidth=0.7)
    for tr in non_laser_trajs:
        ax_pc1.plot(t_epoch, tr[:, 0], color=c_non_laser_bold, alpha=0.12, linewidth=0.7)
    for tr in post_rec_trajs:
        ax_pc1.plot(t_epoch, tr[:, 0], color=c_post_rec_bold, alpha=0.12, linewidth=0.7)
    for tr in laser_trajs:
        ax_pc1.plot(t_epoch, tr[:, 0], color=c_laser_bold, alpha=0.20, linewidth=0.8)

    ax_pc1.plot(t_epoch, mean_traj_base[:, 0], color=c_base_bold, linewidth=2.5, label='Baseline Mean')
    ax_pc1.plot(t_epoch, mean_traj_laser[:, 0], color=c_laser_bold, linestyle='--', linewidth=2.5, label='Laser Perturb Mean')
    ax_pc1.plot(t_epoch, mean_traj_non_laser[:, 0], color=c_non_laser_bold, linewidth=2.5, label='Non-Laser Proto Mean')
    ax_pc1.plot(t_epoch, mean_traj_post_rec[:, 0], color=c_post_rec_bold, linewidth=2.5, label='Post-Laser Rec Mean')
    ax_pc1.axvline(0.0, color='gray', linestyle=':', linewidth=1.5)

    ax_pc1.set_xlabel("Time relative to Insp Onset (s)", fontweight='bold')
    ax_pc1.set_ylabel("PC1 Score", fontweight='bold')
    ax_pc1.set_title("PC1 Score Dynamics (Individual + Mean)", fontsize=11, fontweight='bold')
    ax_pc1.grid(True, linestyle=':', alpha=0.5)
    ax_pc1.legend(loc='upper right', fontsize=8)

    plt.suptitle("Subject TP_Tacr1_1: Individual Breath Rotational Dynamics & Inspiration-Off Attractor", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_out = out_dir / "population_rotational_trajectories.png"
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved High-Res PNG Rotational Trajectory Figure: {png_out}")

    # Copy PNG to artifact folder for markdown embedding
    artifact_png = artifact_dir / "population_rotational_trajectories.png"
    shutil.copy2(png_out, artifact_png)

    print("\n" + "=" * 90)
    print(" [SUCCESS] Individual Breath Rotational Trajectory Analysis Complete!")
    print(f" HTML:     {html_out}")
    print(f" PNG:      {png_out}")
    print(f" Artifact: {artifact_png}")
    print("=" * 90)

if __name__ == "__main__":
    main()
