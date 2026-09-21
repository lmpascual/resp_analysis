#!/usr/bin/env python3
"""
Rotational Population Trajectories & Expiratory Attractor Analysis
strictly using the official `inspiration-off-attractor` repository code and `cibrrig` pipeline.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# Add project root and VLAD repo to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
VLAD_REPO = PROJECT_ROOT / "inspiration-off-attractor"
VLAD_DIR = VLAD_REPO / "VLAD"
LOWD_DIR = VLAD_DIR / "low_d"

for p in [str(PROJECT_ROOT), str(VLAD_REPO), str(VLAD_DIR), str(LOWD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cibrrig.analysis.population import Population
from compute_pca_perturbations import compute_expiratory_attractor, compute_distance_to_attractor

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_rotational_dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Executing Analysis via Official `inspiration-off-attractor` Repository Code ")
    print(f" Repository Path: {VLAD_REPO}")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Neural Spikes & Clusters
    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S)
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    info_df['firing_rate_550s'] = info_df['cluster_id'].map(lambda u: np.sum(spikes_clusters == u) / MAX_TIME_S)

    # 2. Load NIDQ Signals (EMG & Laser TTL)
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
    breath_onsets = emg_proc.breath_onsets
    breath_offsets = emg_proc.breath_offsets

    laser_active = (ch6_laser > 0.8).astype(int)
    laser_onsets = t_nidq[np.where(np.diff(laser_active) == 1)[0]]
    laser_offsets = t_nidq[np.where(np.diff(laser_active) == -1)[0]]

    n_pulses = min(len(laser_onsets), len(laser_offsets))
    laser_onsets = laser_onsets[:n_pulses]
    laser_offsets = laser_offsets[:n_pulses]

    t_first_laser = laser_onsets[0] if len(laser_onsets) > 0 else 60.0
    t_last_laser = laser_offsets[-1] if len(laser_offsets) > 0 else 284.0
    t_post_rec_start = t_last_laser + 180.0
    t_post_rec_end = t_post_rec_start + 60.0

    # Continuous Respiratory Phase phi(t) in [-pi, pi]
    dt = 0.005 # 5 ms time bin
    t_bins = np.arange(0.0, MAX_TIME_S, dt)
    n_bins = len(t_bins)
    phi_vector = np.zeros(n_bins)

    n_cycles = min(len(breath_onsets) - 1, len(breath_offsets))
    for i in range(n_cycles):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i] if breath_offsets[i] > t_on else t_on + 0.15
        t_next = breath_onsets[i + 1]

        idx_insp = np.where((t_bins >= t_on) & (t_bins < t_off))[0]
        if len(idx_insp) > 0:
            phi_vector[idx_insp] = np.linspace(0, np.pi, len(idx_insp))

        idx_exp = np.where((t_bins >= t_off) & (t_bins < t_next))[0]
        if len(idx_exp) > 0:
            phi_vector[idx_exp] = np.linspace(-np.pi, 0.0, len(idx_exp))

    # 3. Deduplicate Good Units & Unit Selection
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
    selected_uids = clean_good_df['cluster_id'].values

    # Subset spikes to clean unit IDs
    mask_clean = np.isin(spikes_clusters, selected_uids)
    spikes_times_clean = spikes_times[mask_clean]
    spikes_clusters_clean = spikes_clusters[mask_clean]

    print(f" Cleaned Units: {len(selected_uids)} units after deduplication filter.")

    # 4. Instantiate cibrrig Population object
    pop = Population(spikes_times_clean, spikes_clusters_clean, ndims=5, binsize=dt, sigma=0.010, t0=0, tf=MAX_TIME_S)
    pop.compute_projection()
    pop.compute_projection_speed(ndims=3)

    # Class container for VLAD compatibility
    class RecordingContainer:
        def __init__(self, phi, phi_t):
            self.phi = phi
            self.phi_t = phi_t

    rec = RecordingContainer(phi_vector, t_bins)

    # 5. Compute Attractor Target I_off using VLAD compute_expiratory_attractor
    attractor_target = compute_expiratory_attractor(rec, pop, ndims=3, binsize=np.pi/25, t0=0, tf=t_first_laser)
    print(f" Attractor Target Coordinate (I_off in PC1-3): {attractor_target}")

    # Compute Euclidean distance from each time bin state to attractor target
    proj_3d = pop.projection[:, :3]
    dist_to_ioff = np.sqrt(np.sum((proj_3d - attractor_target)**2, axis=1))
    speed_vector = pop.projection_speed

    # 6. Extract Trajectories for 4 Experimental Epochs
    pre_win_s = 0.15
    post_win_s = 0.65
    n_pre_bins = int(pre_win_s / dt)
    n_post_bins = int(post_win_s / dt)
    n_epoch_bins = n_pre_bins + n_post_bins
    t_epoch = np.linspace(-pre_win_s, post_win_s, n_epoch_bins)

    def extract_epoch_data(onsets_list):
        trajs_list, speeds_list, dists_list = [], [], []
        for onset in onsets_list:
            idx_onset = int(onset / dt)
            i_start = idx_onset - n_pre_bins
            i_end = idx_onset + n_post_bins
            if 0 <= i_start and i_end <= n_bins:
                trajs_list.append(proj_3d[i_start:i_end, :3])
                speeds_list.append(speed_vector[i_start:i_end])
                dists_list.append(dist_to_ioff[i_start:i_end])
        return trajs_list, speeds_list, dists_list

    baseline_onsets = [o for o in breath_onsets if o < t_first_laser]
    base_trajs, base_speeds, base_dists = extract_epoch_data(baseline_onsets)

    laser_trajs, laser_speeds, laser_dists = [], [], []
    for l_on in laser_onsets:
        idx_lon = int(l_on / dt)
        i_start = idx_lon - n_pre_bins
        i_end = idx_lon + n_post_bins
        if 0 <= i_start and i_end <= n_bins:
            laser_trajs.append(proj_3d[i_start:i_end, :3])
            laser_speeds.append(speed_vector[i_start:i_end])
            laser_dists.append(dist_to_ioff[i_start:i_end])

    non_laser_proto_onsets = [o for o in breath_onsets if t_first_laser <= o <= t_last_laser and np.min(np.abs(o - laser_onsets)) > 0.35]
    non_laser_trajs, non_laser_speeds, non_laser_dists = extract_epoch_data(non_laser_proto_onsets)

    post_rec_onsets = [o for o in breath_onsets if t_post_rec_start <= o <= t_post_rec_end]
    post_rec_trajs, post_rec_speeds, post_rec_dists = extract_epoch_data(post_rec_onsets)

    mean_traj_base = np.mean(base_trajs, axis=0) if len(base_trajs)>0 else np.zeros((n_epoch_bins,3))
    mean_traj_laser = np.mean(laser_trajs, axis=0) if len(laser_trajs)>0 else np.zeros((n_epoch_bins,3))
    mean_traj_non_laser = np.mean(non_laser_trajs, axis=0) if len(non_laser_trajs)>0 else np.zeros((n_epoch_bins,3))
    mean_traj_post_rec = np.mean(post_rec_trajs, axis=0) if len(post_rec_trajs)>0 else np.zeros((n_epoch_bins,3))

    mean_speed_base = np.mean(base_speeds, axis=0)
    mean_speed_laser = np.mean(laser_speeds, axis=0)
    mean_speed_non_laser = np.mean(non_laser_speeds, axis=0)
    mean_speed_post_rec = np.mean(post_rec_speeds, axis=0)

    mean_dist_base = np.mean(base_dists, axis=0)
    mean_dist_laser = np.mean(laser_dists, axis=0)
    mean_dist_non_laser = np.mean(non_laser_dists, axis=0)
    mean_dist_post_rec = np.mean(post_rec_dists, axis=0)

    exp_var = pop.pca.explained_variance_ratio_

    # 7. Render 6-Panel Interactive Plotly Dashboard
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
            "4. Latent Trajectory Speed ||dPC/dt|| (cibrrig)",
            "5. Distance to Expiratory Attractor d(t, I_off) (VLAD)",
            "6. PC1 Score Temporal Dynamics"
        ),
        horizontal_spacing=0.07, vertical_spacing=0.10
    )

    c_base, c_laser, c_non_laser, c_post_rec = 'rgba(31, 119, 180, 0.25)', 'rgba(227, 119, 194, 0.35)', 'rgba(44, 160, 44, 0.25)', 'rgba(255, 127, 14, 0.25)'
    c_base_bold, c_laser_bold, c_non_laser_bold, c_post_rec_bold = '#1f77b4', '#e377c2', '#2ca02c', '#ff7f0e'

    for tr in base_trajs: fig.add_trace(go.Scatter3d(x=tr[:,0], y=tr[:,1], z=tr[:,2], mode='lines', line=dict(color=c_base, width=2), showlegend=False), row=1, col=1)
    for tr in non_laser_trajs: fig.add_trace(go.Scatter3d(x=tr[:,0], y=tr[:,1], z=tr[:,2], mode='lines', line=dict(color=c_non_laser, width=2), showlegend=False), row=1, col=1)
    for tr in post_rec_trajs: fig.add_trace(go.Scatter3d(x=tr[:,0], y=tr[:,1], z=tr[:,2], mode='lines', line=dict(color=c_post_rec, width=2), showlegend=False), row=1, col=1)
    for tr in laser_trajs: fig.add_trace(go.Scatter3d(x=tr[:,0], y=tr[:,1], z=tr[:,2], mode='lines', line=dict(color=c_laser, width=2.5), showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter3d(x=mean_traj_base[:,0], y=mean_traj_base[:,1], z=mean_traj_base[:,2], mode='lines', name=f'Baseline Mean (N={len(base_trajs)})', line=dict(color=c_base_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=mean_traj_laser[:,0], y=mean_traj_laser[:,1], z=mean_traj_laser[:,2], mode='lines', name=f'Laser Perturb Mean (N={len(laser_trajs)})', line=dict(color=c_laser_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=mean_traj_non_laser[:,0], y=mean_traj_non_laser[:,1], z=mean_traj_non_laser[:,2], mode='lines', name=f'Non-Laser Proto Mean (N={len(non_laser_trajs)})', line=dict(color=c_non_laser_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=mean_traj_post_rec[:,0], y=mean_traj_post_rec[:,1], z=mean_traj_post_rec[:,2], mode='lines', name=f'Post-Laser Rec Mean (N={len(post_rec_trajs)})', line=dict(color=c_post_rec_bold, width=7)), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=[attractor_target[0]], y=[attractor_target[1]], z=[attractor_target[2]], mode='markers', name='⭐ Expiratory Attractor (VLAD)', marker=dict(size=10, color='red')), row=1, col=1)

    for tr in base_trajs: fig.add_trace(go.Scatter(x=tr[:,0], y=tr[:,1], mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=1, col=2)
    for tr in non_laser_trajs: fig.add_trace(go.Scatter(x=tr[:,0], y=tr[:,1], mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=1, col=2)
    for tr in post_rec_trajs: fig.add_trace(go.Scatter(x=tr[:,0], y=tr[:,1], mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=1, col=2)
    for tr in laser_trajs: fig.add_trace(go.Scatter(x=tr[:,0], y=tr[:,1], mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=mean_traj_base[:,0], y=mean_traj_base[:,1], mode='lines', name='Baseline Mean', line=dict(color=c_base_bold, width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_traj_laser[:,0], y=mean_traj_laser[:,1], mode='lines', name='Laser Perturb Mean', line=dict(color=c_laser_bold, width=4, dash='dash')), row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_traj_non_laser[:,0], y=mean_traj_non_laser[:,1], mode='lines', name='Non-Laser Proto Mean', line=dict(color=c_non_laser_bold, width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=mean_traj_post_rec[:,0], y=mean_traj_post_rec[:,1], mode='lines', name='Post-Laser Rec Mean', line=dict(color=c_post_rec_bold, width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=[attractor_target[0]], y=[attractor_target[1]], mode='markers', name='Attractor Target', marker=dict(size=12, color='red', symbol='star')), row=1, col=2)

    for tr in base_trajs: fig.add_trace(go.Scatter(x=tr[:,1], y=tr[:,2], mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=1, col=3)
    for tr in non_laser_trajs: fig.add_trace(go.Scatter(x=tr[:,1], y=tr[:,2], mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=1, col=3)
    for tr in post_rec_trajs: fig.add_trace(go.Scatter(x=tr[:,1], y=tr[:,2], mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=1, col=3)
    for tr in laser_trajs: fig.add_trace(go.Scatter(x=tr[:,1], y=tr[:,2], mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=1, col=3)

    fig.add_trace(go.Scatter(x=mean_traj_base[:,1], y=mean_traj_base[:,2], mode='lines', line=dict(color=c_base_bold, width=4), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=mean_traj_laser[:,1], y=mean_traj_laser[:,2], mode='lines', line=dict(color=c_laser_bold, width=4, dash='dash'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=mean_traj_non_laser[:,1], y=mean_traj_non_laser[:,2], mode='lines', line=dict(color=c_non_laser_bold, width=4), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=mean_traj_post_rec[:,1], y=mean_traj_post_rec[:,2], mode='lines', line=dict(color=c_post_rec_bold, width=4), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[attractor_target[1]], y=[attractor_target[2]], mode='markers', marker=dict(size=12, color='red', symbol='star'), showlegend=False), row=1, col=3)

    for sp in base_speeds: fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=2, col=1)
    for sp in non_laser_speeds: fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=2, col=1)
    for sp in post_rec_speeds: fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=2, col=1)
    for sp in laser_speeds: fig.add_trace(go.Scatter(x=t_epoch, y=sp, mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_base, mode='lines', line=dict(color=c_base_bold, width=3.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_laser, mode='lines', line=dict(color=c_laser_bold, width=3.5, dash='dash'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_non_laser, mode='lines', line=dict(color=c_non_laser_bold, width=3.5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_speed_post_rec, mode='lines', line=dict(color=c_post_rec_bold, width=3.5), showlegend=False), row=2, col=1)

    for ds in base_dists: fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=2, col=2)
    for ds in non_laser_dists: fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=2, col=2)
    for ds in post_rec_dists: fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=2, col=2)
    for ds in laser_dists: fig.add_trace(go.Scatter(x=t_epoch, y=ds, mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=2, col=2)

    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_base, mode='lines', line=dict(color=c_base_bold, width=3.5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_laser, mode='lines', line=dict(color=c_laser_bold, width=3.5, dash='dash'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_non_laser, mode='lines', line=dict(color=c_non_laser_bold, width=3.5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_dist_post_rec, mode='lines', line=dict(color=c_post_rec_bold, width=3.5), showlegend=False), row=2, col=2)

    for tr in base_trajs: fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_base, width=1), showlegend=False), row=2, col=3)
    for tr in non_laser_trajs: fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_non_laser, width=1), showlegend=False), row=2, col=3)
    for tr in post_rec_trajs: fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_post_rec, width=1), showlegend=False), row=2, col=3)
    for tr in laser_trajs: fig.add_trace(go.Scatter(x=t_epoch, y=tr[:, 0], mode='lines', line=dict(color=c_laser, width=1.2), showlegend=False), row=2, col=3)

    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_base[:, 0], mode='lines', line=dict(color=c_base_bold, width=3.5), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_laser[:, 0], mode='lines', line=dict(color=c_laser_bold, width=3.5, dash='dash'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_non_laser[:, 0], mode='lines', line=dict(color=c_non_laser_bold, width=3.5), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=t_epoch, y=mean_traj_post_rec[:, 0], mode='lines', line=dict(color=c_post_rec_bold, width=3.5), showlegend=False), row=2, col=3)

    fig.update_layout(
        title="<b>Subject TP_Tacr1_1: Rotational Dynamics via Official `inspiration-off-attractor` Repo</b>",
        template="plotly_white",
        height=1000,
        margin=dict(l=50, r=40, t=80, b=50),
        xaxis2=dict(title="PC1 Score"), yaxis2=dict(title="PC2 Score"),
        xaxis3=dict(title="PC2 Score"), yaxis3=dict(title="PC3 Score"),
        xaxis4=dict(title="Time relative to Insp Onset (s)"), yaxis4=dict(title="Trajectory Speed ||dPC/dt||"),
        xaxis5=dict(title="Time relative to Insp Onset (s)"), yaxis5=dict(title="Distance to Attractor d(t, I_off)"),
        xaxis6=dict(title="Time relative to Insp Onset (s)"), yaxis6=dict(title="PC1 Score")
    )

    html_out = out_dir / "population_rotational_trajectories.html"
    fig.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    # 8. High-Resolution Static PNG Figure
    fig_mpl = plt.figure(figsize=(20, 11))
    ax3d = fig_mpl.add_subplot(2, 3, 1, projection='3d')
    for tr in base_trajs: ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_base_bold, alpha=0.15, linewidth=0.8)
    for tr in non_laser_trajs: ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_non_laser_bold, alpha=0.15, linewidth=0.8)
    for tr in post_rec_trajs: ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_post_rec_bold, alpha=0.15, linewidth=0.8)
    for tr in laser_trajs: ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c_laser_bold, alpha=0.25, linewidth=1.0)

    ax3d.plot(mean_traj_base[:, 0], mean_traj_base[:, 1], mean_traj_base[:, 2], color=c_base_bold, linewidth=3.5, label=f'Baseline Mean (N={len(base_trajs)})')
    ax3d.plot(mean_traj_laser[:, 0], mean_traj_laser[:, 1], mean_traj_laser[:, 2], color=c_laser_bold, linestyle='--', linewidth=3.5, label=f'Laser Perturb Mean (N={len(laser_trajs)})')
    ax3d.plot(mean_traj_non_laser[:, 0], mean_traj_non_laser[:, 1], mean_traj_non_laser[:, 2], color=c_non_laser_bold, linewidth=3, label=f'Non-Laser Proto Mean (N={len(non_laser_trajs)})')
    ax3d.plot(mean_traj_post_rec[:, 0], mean_traj_post_rec[:, 1], mean_traj_post_rec[:, 2], color=c_post_rec_bold, linewidth=3, label=f'Post-Laser Rec Mean (N={len(post_rec_trajs)})')
    ax3d.scatter(attractor_target[0], attractor_target[1], attractor_target[2], color='red', s=140, marker='*', zorder=10, label='Expiratory Attractor (VLAD)')

    ax3d.set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontweight='bold')
    ax3d.set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontweight='bold')
    ax3d.set_zlabel(f"PC3 ({exp_var[2]*100:.1f}%)", fontweight='bold')
    ax3d.set_title(f"3D Rotational Orbits ({np.sum(exp_var[:3])*100:.1f}% Var)", fontsize=11, fontweight='bold')
    ax3d.legend(loc='upper left', fontsize=7.5)

    ax2d1 = fig_mpl.add_subplot(2, 3, 2)
    for tr in base_trajs: ax2d1.plot(tr[:, 0], tr[:, 1], color=c_base_bold, alpha=0.15, linewidth=0.8)
    for tr in non_laser_trajs: ax2d1.plot(tr[:, 0], tr[:, 1], color=c_non_laser_bold, alpha=0.15, linewidth=0.8)
    for tr in post_rec_trajs: ax2d1.plot(tr[:, 0], tr[:, 1], color=c_post_rec_bold, alpha=0.15, linewidth=0.8)
    for tr in laser_trajs: ax2d1.plot(tr[:, 0], tr[:, 1], color=c_laser_bold, alpha=0.25, linewidth=1.0)
    ax2d1.plot(mean_traj_base[:, 0], mean_traj_base[:, 1], color=c_base_bold, linewidth=3.5, label='Baseline')
    ax2d1.plot(mean_traj_laser[:, 0], mean_traj_laser[:, 1], color=c_laser_bold, linestyle='--', linewidth=3.5, label='Laser Perturb')
    ax2d1.plot(mean_traj_non_laser[:, 0], mean_traj_non_laser[:, 1], color=c_non_laser_bold, linewidth=3, label='Non-Laser Proto')
    ax2d1.plot(mean_traj_post_rec[:, 0], mean_traj_post_rec[:, 1], color=c_post_rec_bold, linewidth=3, label='Post-Laser Rec')
    ax2d1.scatter(attractor_target[0], attractor_target[1], color='red', s=160, marker='*', zorder=10)
    ax2d1.set_xlabel("PC1 Score", fontweight='bold'); ax2d1.set_ylabel("PC2 Score", fontweight='bold')
    ax2d1.set_title("2D Projection (PC1 vs PC2)", fontsize=11, fontweight='bold')

    ax2d2 = fig_mpl.add_subplot(2, 3, 3)
    for tr in base_trajs: ax2d2.plot(tr[:, 1], tr[:, 2], color=c_base_bold, alpha=0.15, linewidth=0.8)
    for tr in non_laser_trajs: ax2d2.plot(tr[:, 1], tr[:, 2], color=c_non_laser_bold, alpha=0.15, linewidth=0.8)
    for tr in post_rec_trajs: ax2d2.plot(tr[:, 1], tr[:, 2], color=c_post_rec_bold, alpha=0.15, linewidth=0.8)
    for tr in laser_trajs: ax2d2.plot(tr[:, 1], tr[:, 2], color=c_laser_bold, alpha=0.25, linewidth=1.0)
    ax2d2.plot(mean_traj_base[:, 1], mean_traj_base[:, 2], color=c_base_bold, linewidth=3.5)
    ax2d2.plot(mean_traj_laser[:, 1], mean_traj_laser[:, 2], color=c_laser_bold, linestyle='--', linewidth=3.5)
    ax2d2.plot(mean_traj_non_laser[:, 1], mean_traj_non_laser[:, 2], color=c_non_laser_bold, linewidth=3)
    ax2d2.plot(mean_traj_post_rec[:, 1], mean_traj_post_rec[:, 2], color=c_post_rec_bold, linewidth=3)
    ax2d2.scatter(attractor_target[1], attractor_target[2], color='red', s=160, marker='*', zorder=10)
    ax2d2.set_xlabel("PC2 Score", fontweight='bold'); ax2d2.set_ylabel("PC3 Score", fontweight='bold')
    ax2d2.set_title("2D Projection (PC2 vs PC3)", fontsize=11, fontweight='bold')

    ax_sp = fig_mpl.add_subplot(2, 3, 4)
    ax_sp.plot(t_epoch, mean_speed_base, color=c_base_bold, linewidth=3, label='Baseline')
    ax_sp.plot(t_epoch, mean_speed_laser, color=c_laser_bold, linestyle='--', linewidth=3, label='Laser Perturb')
    ax_sp.plot(t_epoch, mean_speed_non_laser, color=c_non_laser_bold, linewidth=3, label='Non-Laser Proto')
    ax_sp.plot(t_epoch, mean_speed_post_rec, color=c_post_rec_bold, linewidth=3, label='Post-Laser Rec')
    ax_sp.set_xlabel("Time relative to Insp Onset (s)", fontweight='bold')
    ax_sp.set_ylabel("Trajectory Speed ||dPC/dt||", fontweight='bold')
    ax_sp.set_title("Latent Trajectory Speed Profile (cibrrig)", fontsize=11, fontweight='bold')
    ax_sp.legend(loc='upper right', fontsize=8)

    ax_dst = fig_mpl.add_subplot(2, 3, 5)
    ax_dst.plot(t_epoch, mean_dist_base, color=c_base_bold, linewidth=3, label='Baseline')
    ax_dst.plot(t_epoch, mean_dist_laser, color=c_laser_bold, linestyle='--', linewidth=3, label='Laser Perturb')
    ax_dst.plot(t_epoch, mean_dist_non_laser, color=c_non_laser_bold, linewidth=3, label='Non-Laser Proto')
    ax_dst.plot(t_epoch, mean_dist_post_rec, color=c_post_rec_bold, linewidth=3, label='Post-Laser Rec')
    ax_dst.set_xlabel("Time relative to Insp Onset (s)", fontweight='bold')
    ax_dst.set_ylabel("Distance to Attractor d(t, I_off)", fontweight='bold')
    ax_dst.set_title("Distance to Expiratory Attractor (VLAD)", fontsize=11, fontweight='bold')

    ax_pc1 = fig_mpl.add_subplot(2, 3, 6)
    ax_pc1.plot(t_epoch, mean_traj_base[:, 0], color=c_base_bold, linewidth=3)
    ax_pc1.plot(t_epoch, mean_traj_laser[:, 0], color=c_laser_bold, linestyle='--', linewidth=3)
    ax_pc1.plot(t_epoch, mean_traj_non_laser[:, 0], color=c_non_laser_bold, linewidth=3)
    ax_pc1.plot(t_epoch, mean_traj_post_rec[:, 0], color=c_post_rec_bold, linewidth=3)
    ax_pc1.set_xlabel("Time relative to Insp Onset (s)", fontweight='bold')
    ax_pc1.set_ylabel("PC1 Score", fontweight='bold')
    ax_pc1.set_title("PC1 Temporal Dynamics", fontsize=11, fontweight='bold')

    plt.suptitle("Subject TP_Tacr1_1: Population Rotational Orbits & Attractor Dynamics (VLAD Repository Implementation)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    png_out = out_dir / "population_rotational_trajectories.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy to artifact dir
    artifact_png = artifact_dir / "population_rotational_trajectories.png"
    import shutil
    shutil.copy2(png_out, artifact_png)

    print(f" Saved High-Res PNG Figure: {png_out}")
    print("\n==========================================================================================")
    print(" [SUCCESS] VLAD Repository Rotational Trajectory Analysis Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
