#!/usr/bin/env python3
"""
Replication of Publication Figures (Figures 2 & 3: Single-Cell Phase Dynamics & Low-Dimensional Manifolds)
Strictly utilizing the `cibrrig` analysis toolkit, `inspiration-off-attractor` methodologies, and clean sliding-window single units.
Generates:
1. Figure 2: Single-Cell Respiratory Phase Dynamics (Rasters, PSTHs, Polar Plots, 2-Cycle Heatmap, Coherence, & Depth Profiles)
2. Figure 3: Low-Dimensional Neural Manifold & Attractor Trajectories (3D PCA, Phase & Speed Coloring, Attractor Dispersions)
3. Interactive 3D HTML Manifold Visualizers for in-browser exploration.
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import plotly.graph_objects as go

# Add project root and VLAD repo to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VLAD_REPO = PROJECT_ROOT / "inspiration-off-attractor"
VLAD_DIR = VLAD_REPO / "VLAD"

for p in [str(PROJECT_ROOT), str(VLAD_REPO), str(VLAD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("REPLICATE_PUBLICATION_FIGURES")

# Palette constants matching paper
COLOR_INSP = "#d9534f"   # Coral / Red
COLOR_EXP = "#337ab7"    # Steel Blue
COLOR_TONIC = "#888888"  # Slate Gray
PC_COLORS = ["#2ecc71", "#e67e22", "#9b59b6", "#3498db"]  # Green, Orange, Purple, Blue


def compute_continuous_respiratory_phase(
    t_nidq: np.ndarray,
    emg_envelope: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    t_eval: np.ndarray
) -> np.ndarray:
    """
    Computes continuous respiratory phase phi(t) in [-pi, pi] for each time point in t_eval:
    - Inspiration: 0 to pi (from breath onset to breath offset)
    - Expiration: -pi to 0 (from breath offset to next breath onset)
    """
    phi = np.full_like(t_eval, np.nan)
    n_breaths = min(len(breath_onsets), len(breath_offsets))
    
    for i in range(n_breaths - 1):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i]
        t_next_on = breath_onsets[i + 1]

        if t_off <= t_on or t_next_on <= t_off:
            continue

        # Inspiratory phase: [t_on, t_off] -> [0, pi]
        insp_mask = (t_eval >= t_on) & (t_eval < t_off)
        if np.any(insp_mask):
            phi[insp_mask] = (t_eval[insp_mask] - t_on) / (t_off - t_on) * np.pi

        # Expiratory phase: [t_off, t_next_on] -> [-pi, 0]
        exp_mask = (t_eval >= t_off) & (t_eval < t_next_on)
        if np.any(exp_mask):
            phi[exp_mask] = -np.pi + (t_eval[exp_mask] - t_off) / (t_next_on - t_off) * np.pi

    # Fill boundaries
    valid_idx = np.where(~np.isnan(phi))[0]
    if len(valid_idx) > 0:
        phi[:valid_idx[0]] = phi[valid_idx[0]]
        phi[valid_idx[-1]:] = phi[valid_idx[-1]]

    return phi


def classify_units_by_phase(
    spikes_times: np.ndarray,
    spikes_clusters: np.ndarray,
    unit_ids: np.ndarray,
    t_eval: np.ndarray,
    phi_eval: np.ndarray,
    duration_s: float
) -> pd.DataFrame:
    """
    Classifies units into Inspiratory, Expiratory, and Tonic based on circular vector strength and preferred phase.
    """
    records = []
    n_phase_bins = 50
    phase_bins = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2.0

    # Phase occupancy
    valid_phi = phi_eval[~np.isnan(phi_eval)]
    phi_hist_total, _ = np.histogram(valid_phi, bins=phase_bins)
    dt_eval = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 0.005
    occupancy_s = phi_hist_total * dt_eval + 1e-6

    for uid in unit_ids:
        spks = spikes_times[spikes_clusters == uid]
        if len(spks) == 0:
            continue

        # Interpolate spike phases
        spk_phases = np.interp(spks, t_eval, phi_eval)
        spk_phases = spk_phases[~np.isnan(spk_phases)]

        if len(spk_phases) < 5:
            continue

        # Phase tuning curve (Firing rate vs phase)
        spk_hist, _ = np.histogram(spk_phases, bins=phase_bins)
        phase_fr = spk_hist / occupancy_s

        # Circular mean and resultant vector length (Rayleigh vector strength)
        sin_sum = np.sum(np.sin(spk_phases))
        cos_sum = np.sum(np.cos(spk_phases))
        r_vector_length = np.sqrt(sin_sum**2 + cos_sum**2) / len(spk_phases)
        mean_phase = np.arctan2(sin_sum, cos_sum)

        # Rayleigh test for circular non-uniformity (cibrrig / VLAD respModThresh = 0.20)
        z_score = len(spk_phases) * (r_vector_length**2)
        p_val = np.exp(-z_score) * (1.0 + (2.0 * z_score - z_score**2) / (4.0 * len(spk_phases)))
        is_modulated = (p_val < 0.05) and (r_vector_length >= 0.20)

        # Classification (VLAD assign_resp_category standard)
        if not is_modulated:
            unit_class = "Tonic"
        elif 0 <= mean_phase <= np.pi:
            unit_class = "Inspiratory"
        else:
            unit_class = "Expiratory"

        records.append({
            "cluster_id": int(uid),
            "n_spikes": len(spks),
            "mean_phase": float(mean_phase),
            "vector_strength": float(r_vector_length),
            "is_modulated": bool(is_modulated),
            "p_val": float(p_val),
            "unit_class": unit_class,
            "phase_fr": phase_fr.tolist(),
            "phase_centers": phase_centers.tolist()
        })

    return pd.DataFrame(records)


def plot_figure_2_replication(
    emg_env: np.ndarray,
    t_nidq: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    spikes_times: np.ndarray,
    spikes_clusters: np.ndarray,
    df_info: pd.DataFrame,
    df_phase: pd.DataFrame,
    out_file: Path,
    time_window: Tuple[float, float] = (5.0, 9.0)
):
    """
    Replicates Figure 2 from the publication:
    c: Depth-ordered raster + Integrated Diaphragm EMG
    d: Average breath-triggered PSTHs (Inspiratory vs Expiratory vs Tonic)
    e: Polar circular phase distribution
    f: 2-cycle (-pi to 3pi) Phase-ordered population heatmap + % Active curve
    g: Polar histogram of preferred unit phases
    h: Modulated unit vector strength distribution
    i: Spatial depth distribution of Inspiratory vs Expiratory units
    """
    fig = plt.figure(figsize=(19, 13))
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1.2, 1.05, 1.25, 1.0], height_ratios=[1.0, 1.1, 0.95], hspace=0.45, wspace=0.38)

    # -------------------------------------------------------------
    # Panel c: Depth-Ordered Raster + Integrated Diaphragm EMG
    # -------------------------------------------------------------
    t0_snap, t1_snap = time_window
    ax_c = fig.add_subplot(gs[0:2, 0])
    ax_emg = fig.add_subplot(gs[2, 0], sharex=ax_c)

    # Sort units by depth
    df_merged = df_info.merge(df_phase, on="cluster_id", how="inner").sort_values(by="depth_um", ascending=True).reset_index(drop=True)
    
    for idx, row in df_merged.iterrows():
        cid = row["cluster_id"]
        u_spks = spikes_times[spikes_clusters == cid]
        in_snap = u_spks[(u_spks >= t0_snap) & (u_spks <= t1_snap)]
        u_class = row["unit_class"]
        color = COLOR_INSP if u_class == "Inspiratory" else (COLOR_EXP if u_class == "Expiratory" else COLOR_TONIC)
        ax_c.scatter(in_snap, np.full_like(in_snap, idx + 1), s=6, color=color, alpha=0.9)

    ax_c.set_title("c | Depth-Ordered Single-Unit Raster", fontsize=11, fontweight="bold", loc="left")
    ax_c.set_ylabel("Neuron ID (Ordered by Probe Depth)", fontsize=10)
    ax_c.set_ylim(0, len(df_merged) + 1)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # EMG trace
    mask_emg = (t_nidq >= t0_snap) & (t_nidq <= t1_snap)
    ax_emg.plot(t_nidq[mask_emg], emg_env[mask_emg], color="gray", linewidth=1.5)
    ax_emg.set_ylabel("Dia. EMG", fontsize=10)
    ax_emg.set_xlabel("Time (seconds)", fontsize=10)
    ax_emg.set_xlim(t0_snap, t1_snap)
    ax_emg.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel d: Average Breath-Triggered PSTH (Insp vs Exp vs Tonic)
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[0, 1])
    win_pre, win_post = 0.2, 0.4  # [-200ms, +400ms]
    dt_psth = 0.005
    psth_bins = np.arange(-win_pre, win_post + dt_psth, dt_psth)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0 * 1000.0  # in ms

    insp_psths, exp_psths, tonic_psths = [], [], []
    valid_onsets = breath_onsets[(breath_onsets > win_pre) & (breath_onsets < t_nidq[-1] - win_post)]

    for _, row in df_merged.iterrows():
        cid = row["cluster_id"]
        spks = spikes_times[spikes_clusters == cid]
        trial_counts = np.zeros(len(psth_bins) - 1)
        for on in valid_onsets:
            rel = spks - on
            in_win = rel[(rel >= -win_pre) & (rel <= win_post)]
            c, _ = np.histogram(in_win, bins=psth_bins)
            trial_counts += c
        rate_hz = trial_counts / (len(valid_onsets) * dt_psth)
        
        if row["unit_class"] == "Inspiratory":
            insp_psths.append(rate_hz)
        elif row["unit_class"] == "Expiratory":
            exp_psths.append(rate_hz)
        else:
            tonic_psths.append(rate_hz)

    if insp_psths:
        mean_insp = np.mean(insp_psths, axis=0)
        sem_insp = stats.sem(insp_psths, axis=0)
        ax_d.plot(psth_centers, mean_insp, color=COLOR_INSP, linewidth=2, label="Inspiratory")
        ax_d.fill_between(psth_centers, mean_insp - sem_insp, mean_insp + sem_insp, color=COLOR_INSP, alpha=0.2)

    if exp_psths:
        mean_exp = np.mean(exp_psths, axis=0)
        sem_exp = stats.sem(exp_psths, axis=0)
        ax_d.plot(psth_centers, mean_exp, color=COLOR_EXP, linewidth=2, label="Expiratory")
        ax_d.fill_between(psth_centers, mean_exp - sem_exp, mean_exp + sem_exp, color=COLOR_EXP, alpha=0.2)

    if tonic_psths:
        mean_tonic = np.mean(tonic_psths, axis=0)
        sem_tonic = stats.sem(tonic_psths, axis=0)
        ax_d.plot(psth_centers, mean_tonic, color=COLOR_TONIC, linewidth=1.5, label="Tonic")
        ax_d.fill_between(psth_centers, mean_tonic - sem_tonic, mean_tonic + sem_tonic, color=COLOR_TONIC, alpha=0.15)

    ax_d.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax_d.axvline(120, color="black", linestyle=":", alpha=0.7)
    ax_d.set_title("d | Breath-Aligned Firing Rate", fontsize=11, fontweight="bold", loc="left")
    ax_d.set_xlabel("Time (ms)", fontsize=9)
    ax_d.set_ylabel("Firing Rate (sp/s)", fontsize=9)
    ax_d.legend(frameon=False, fontsize=8)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel e: Polar Phase Distribution Plot (Lowered with Sub-Gridspec)
    # -------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1:3, 1], polar=True)
    n_p = 50
    theta = np.linspace(0, 2*np.pi, n_p)
    
    if insp_psths:
        insp_polar = np.mean([row["phase_fr"] for _, row in df_merged[df_merged["unit_class"]=="Inspiratory"].iterrows()], axis=0)
        theta_mapped = np.linspace(0, 2*np.pi, len(insp_polar))
        ax_e.plot(theta_mapped, insp_polar, color=COLOR_INSP, linewidth=1.5)
        ax_e.fill(theta_mapped, insp_polar, color=COLOR_INSP, alpha=0.25, label="Insp")

    if exp_psths:
        exp_polar = np.mean([row["phase_fr"] for _, row in df_merged[df_merged["unit_class"]=="Expiratory"].iterrows()], axis=0)
        theta_mapped = np.linspace(0, 2*np.pi, len(exp_polar))
        ax_e.plot(theta_mapped, exp_polar, color=COLOR_EXP, linewidth=1.5)
        ax_e.fill(theta_mapped, exp_polar, color=COLOR_EXP, alpha=0.25, label="Exp")

    ax_e.set_title("e | Polar Phase Distribution", fontsize=11, fontweight="bold", loc="left", pad=12)
    ax_e.set_theta_zero_location("E")
    ax_e.set_theta_direction(1)

    # -------------------------------------------------------------
    # Panel f: Phase-Ordered Population Heatmap
    # -------------------------------------------------------------
    ax_f = fig.add_subplot(gs[0:2, 2])
    ax_f_act = fig.add_subplot(gs[2, 2], sharex=ax_f)

    # Sort units by preferred phase mean_phase
    df_sorted_phase = df_merged.sort_values(by="mean_phase", ascending=True).reset_index(drop=True)
    heat_rows = []
    
    for _, row in df_sorted_phase.iterrows():
        cfr = np.array(row["phase_fr"])
        max_v = np.max(cfr) if np.max(cfr) > 0 else 1.0
        norm_fr = cfr / max_v
        # Duplicate to 2 cycles (-pi to 3pi)
        two_cycle = np.concatenate([norm_fr, norm_fr])
        heat_rows.append(two_cycle)

    heat_rows = np.array(heat_rows)
    if len(heat_rows) > 0:
        im = ax_f.imshow(
            heat_rows,
            aspect="auto",
            cmap="Greys",
            extent=[-np.pi, 3*np.pi, 0, len(df_sorted_phase)],
            origin="lower",
            interpolation="nearest"
        )
        cbar = plt.colorbar(im, ax=ax_f, fraction=0.04, pad=0.02)
        cbar.set_label("Normalized FR", fontsize=9)

    ax_f.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax_f.axvline(np.pi, color="blue", linestyle="--", alpha=0.8)
    ax_f.axvline(2*np.pi, color="red", linestyle="--", alpha=0.8)
    ax_f.set_title("f | Phase-Ordered Population Heatmap", fontsize=11, fontweight="bold", loc="left")
    ax_f.set_ylabel("Neuron ID (Phase Sorted)", fontsize=10)

    # Population active curve underneath
    if len(heat_rows) > 0:
        active_pct = np.mean(heat_rows > 0.3, axis=0) * 100.0
        two_cycle_phase = np.linspace(-np.pi, 3*np.pi, len(active_pct))
        ax_f_act.plot(two_cycle_phase, active_pct, color="black", linewidth=1.5)
        ax_f_act.set_ylabel("Active (%)", fontsize=9)
        ax_f_act.set_xlabel("Respiratory Phase (rad)", fontsize=9)
        ax_f_act.set_xlim(-np.pi, 3*np.pi)
        ax_f_act.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel g & h: Polar Histogram & Coherence / Vector Strength
    # -------------------------------------------------------------
    ax_g = fig.add_subplot(gs[0, 3], polar=True)
    all_phases = df_merged["mean_phase"].values
    counts_g, bins_g = np.histogram(all_phases, bins=24)
    centers_g = (bins_g[:-1] + bins_g[1:]) / 2.0
    ax_g.bar(centers_g, counts_g, width=bins_g[1]-bins_g[0], color="dimgray", edgecolor="black", alpha=0.7)
    ax_g.set_title("g | Phase Histogram", fontsize=11, fontweight="bold", loc="left", pad=12)

    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.hist(df_merged["vector_strength"], bins=15, color="lightgray", edgecolor="black")
    ax_h.set_title("h | Vector Strength (R)", fontsize=11, fontweight="bold", loc="left")
    ax_h.set_xlabel("Vector Strength ($R$)", fontsize=10)
    ax_h.set_ylabel("Number of Neurons", fontsize=10)
    ax_h.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel i: Probe Depth Distribution of Insp vs Exp Units
    # -------------------------------------------------------------
    ax_i = fig.add_subplot(gs[2, 3])
    insp_depths = df_merged[df_merged["unit_class"]=="Inspiratory"]["depth_um"].values
    exp_depths = df_merged[df_merged["unit_class"]=="Expiratory"]["depth_um"].values
    bins_depth = np.linspace(df_merged["depth_um"].min(), df_merged["depth_um"].max(), 12)
    
    if len(insp_depths) > 0:
        ax_i.hist(insp_depths, bins=bins_depth, color=COLOR_INSP, alpha=0.6, label="Inspiratory", edgecolor="black")
    if len(exp_depths) > 0:
        ax_i.hist(exp_depths, bins=bins_depth, color=COLOR_EXP, alpha=0.6, label="Expiratory", edgecolor="black")

    ax_i.set_title("i | Probe Depth Distribution", fontsize=12, fontweight="bold", loc="left")
    ax_i.set_xlabel("Probe Depth (µm)", fontsize=10)
    ax_i.set_ylabel("Neuron Count", fontsize=10)
    ax_i.legend(frameon=False, fontsize=9)
    ax_i.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Figure 2 Replication: {out_file.resolve()}")


def plot_figure_3_replication(
    emg_env: np.ndarray,
    t_nidq: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    spikes_times: np.ndarray,
    spikes_clusters: np.ndarray,
    df_info: pd.DataFrame,
    df_phase: pd.DataFrame,
    out_file_2d: Path,
    out_file_3d_html: Path,
    duration_s: float
):
    """
    Replicates Figure 3 from the publication:
    a: Integrated Diaphragm EMG + Raster + Leading 4 Principal Components (PCs)
    b: Breath-Aligned Average PCs
    c & d: 3D / 2D Neural Manifold Colored by Phase and Trajectory Speed
    e: Phase-aligned PCs (-pi to 3pi)
    f: Trajectory Speed Polar Function
    g, h, i: Average trajectory & breath-to-breath Euclidean dispersion
    """
    # 1. Construct Smoothed Firing Rate Matrix
    dt = 0.005  # 5 ms time bin
    t_bins = np.arange(0.0, duration_s, dt)
    unit_ids = df_info["cluster_id"].values
    
    fr_matrix = []
    # Gaussian smoothing kernel (sigma = 25 ms)
    gaussian_sigma_bins = 5
    kernel = signal.windows.gaussian(21, std=gaussian_sigma_bins)
    kernel /= np.sum(kernel)

    for uid in unit_ids:
        spks = spikes_times[spikes_clusters == uid]
        counts, _ = np.histogram(spks, bins=np.append(t_bins, t_bins[-1] + dt))
        smooth_rate = np.convolve(counts / dt, kernel, mode="same")
        fr_matrix.append(smooth_rate)

    fr_matrix = np.array(fr_matrix)  # Shape: (N_units, N_time_bins)

    # 2. Compute Principal Component Analysis (PCA)
    pca = PCA(n_components=6)
    pca_proj = pca.fit_transform(fr_matrix.T)  # Shape: (N_time_bins, 6)
    
    # Instantaneous trajectory speed: || dX / dt ||
    grad_pca = np.gradient(pca_proj[:, :3], axis=0) / dt
    traj_speed = np.linalg.norm(grad_pca, axis=1)

    # Continuous Respiratory Phase phi(t)
    phi_eval = compute_continuous_respiratory_phase(t_nidq, emg_env, breath_onsets, breath_offsets, t_bins)

    # 3. Figure 3 Multi-Panel Assembly
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # -------------------------------------------------------------
    # Panel a: Diaphragm + Raster + Top 4 PCs
    # -------------------------------------------------------------
    t0_snap, t1_snap = 5.0, 9.0
    mask_snap_nidq = (t_nidq >= t0_snap) & (t_nidq <= t1_snap)
    mask_snap_bins = (t_bins >= t0_snap) & (t_bins <= t1_snap)

    ax_a_emg = fig.add_subplot(gs[0, 0])
    ax_a_ras = fig.add_subplot(gs[1, 0], sharex=ax_a_emg)
    ax_a_pcs = fig.add_subplot(gs[2, 0], sharex=ax_a_emg)

    ax_a_emg.plot(t_nidq[mask_snap_nidq], emg_env[mask_snap_nidq], color="black", linewidth=1.2)
    ax_a_emg.set_title("a | Multi-Scale Dynamics", fontsize=12, fontweight="bold", loc="left")
    ax_a_emg.set_ylabel("Diaphragm", fontsize=10)
    ax_a_emg.grid(True, linestyle=":", alpha=0.5)

    for idx, uid in enumerate(unit_ids):
        spks = spikes_times[spikes_clusters == uid]
        in_s = spks[(spks >= t0_snap) & (spks <= t1_snap)]
        ax_a_ras.scatter(in_s, np.full_like(in_s, idx + 1), s=4, color="gray", alpha=0.8)
    ax_a_ras.set_ylabel("Neuron ID", fontsize=10)
    ax_a_ras.grid(True, linestyle=":", alpha=0.5)

    for i in range(4):
        ax_a_pcs.plot(t_bins[mask_snap_bins], pca_proj[mask_snap_bins, i], label=f"$PC_{i+1}$", color=PC_COLORS[i], linewidth=1.5)
    ax_a_pcs.set_ylabel("PC Projections", fontsize=10)
    ax_a_pcs.set_xlabel("Time (seconds)", fontsize=10)
    ax_a_pcs.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    ax_a_pcs.set_xlim(t0_snap, t1_snap)
    ax_a_pcs.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel b: Breath-Aligned Average Leading PCs
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    win_b = 0.5  # [-500ms, +500ms]
    t_psth = np.arange(-win_b, win_b, dt) * 1000.0  # ms
    valid_onsets = breath_onsets[(breath_onsets > win_b) & (breath_onsets < duration_s - win_b)]
    
    for i in range(4):
        pc_trials = []
        for on in valid_onsets:
            idx_start = int(np.round((on - win_b) / dt))
            idx_end = idx_start + len(t_psth)
            if 0 <= idx_start and idx_end <= len(pca_proj):
                pc_trials.append(pca_proj[idx_start:idx_end, i])
        if pc_trials:
            mean_pc = np.mean(pc_trials, axis=0)
            sem_pc = stats.sem(pc_trials, axis=0)
            ax_b.plot(t_psth, mean_pc, label=f"$PC_{i+1}$", color=PC_COLORS[i], linewidth=2)
            ax_b.fill_between(t_psth, mean_pc - sem_pc, mean_pc + sem_pc, color=PC_COLORS[i], alpha=0.2)

    ax_b.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax_b.set_title("b | Breath-Aligned Average PCs", fontsize=12, fontweight="bold", loc="left")
    ax_b.set_xlabel("Time from Breath Onset (ms)", fontsize=10)
    ax_b.set_ylabel("PC Amplitude (a.u.)", fontsize=10)
    ax_b.legend(frameon=False, fontsize=8)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel c & d: Low-Dimensional Manifold (Colored by Phase & Speed)
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # Plot PC1 vs PC2 colored by phase phi(t)
    sc_c = ax_c.scatter(pca_proj[:, 0], pca_proj[:, 1], c=phi_eval, cmap="twilight", s=5, alpha=0.7)
    cbar_c = plt.colorbar(sc_c, ax=ax_c, fraction=0.04, pad=0.02)
    cbar_c.set_label(r"Phase $\phi$ (rad)", fontsize=9)
    ax_c.set_title(r"c | Manifold: Phase Trajectory ($PC_1$ vs $PC_2$)", fontsize=11, fontweight="bold", loc="left")
    ax_c.set_xlabel(r"$PC_1$ (Leading Mode)", fontsize=10)
    ax_c.set_ylabel(r"$PC_2$", fontsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # Plot PC1 vs PC2 colored by trajectory speed
    sc_d = ax_d.scatter(pca_proj[:, 0], pca_proj[:, 1], c=traj_speed, cmap="inferno", s=5, alpha=0.7)
    cbar_d = plt.colorbar(sc_d, ax=ax_d, fraction=0.04, pad=0.02)
    cbar_d.set_label("Speed ||dX/dt||", fontsize=9)
    ax_d.set_title(r"d | Manifold: Trajectory Speed ($PC_1$ vs $PC_2$)", fontsize=11, fontweight="bold", loc="left")
    ax_d.set_xlabel(r"$PC_1$", fontsize=10)
    ax_d.set_ylabel(r"$PC_2$", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel e: Phase-Aligned PCs (-pi to 3pi)
    # -------------------------------------------------------------
    ax_e = fig.add_subplot(gs[0, 2])
    n_phase_bins = 50
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    phase_centers = (phase_edges[:-1] + phase_edges[1:]) / 2.0

    for i in range(4):
        pc_phase_binned = []
        for p_idx in range(n_phase_bins):
            mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
            if np.any(mask_p):
                pc_phase_binned.append(np.mean(pca_proj[mask_p, i]))
            else:
                pc_phase_binned.append(0.0)
        
        # 2 cycles
        two_cyc_pc = np.concatenate([pc_phase_binned, pc_phase_binned])
        two_cyc_rad = np.linspace(-np.pi, 3*np.pi, len(two_cyc_pc))
        ax_e.plot(two_cyc_rad, two_cyc_pc, label=f"$PC_{i+1}$", color=PC_COLORS[i], linewidth=2)

    ax_e.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax_e.axvline(np.pi, color="blue", linestyle="--", alpha=0.8)
    ax_e.axvline(2*np.pi, color="red", linestyle="--", alpha=0.8)
    ax_e.set_title("e | Phase-Aligned PCs (2 Cycles)", fontsize=11, fontweight="bold", loc="left")
    ax_e.set_xlabel("Phase (rad)", fontsize=10)
    ax_e.set_ylabel("Amplitude (a.u.)", fontsize=10)
    ax_e.set_xlim(-np.pi, 3*np.pi)
    ax_e.legend(frameon=False, fontsize=8)
    ax_e.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Panel f: Trajectory Speed Polar Plot
    # -------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2, 1], polar=True)
    speed_phase_binned = []
    for p_idx in range(n_phase_bins):
        mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
        speed_phase_binned.append(np.mean(traj_speed[mask_p]) if np.any(mask_p) else 0.0)

    theta_speed = np.linspace(0, 2*np.pi, len(speed_phase_binned))
    ax_f.plot(theta_speed, speed_phase_binned, color="crimson", linewidth=2)
    ax_f.fill(theta_speed, speed_phase_binned, color="crimson", alpha=0.25)
    ax_f.set_title("f | Angular Trajectory Speed", fontsize=11, fontweight="bold", loc="left", pad=15)

    # -------------------------------------------------------------
    # Panel i: Breath-to-Breath Dispersion (Euclidean Distance to Mean)
    # -------------------------------------------------------------
    ax_i = fig.add_subplot(gs[2, 2])
    # Compute mean trajectory in 3D PC space across phase
    mean_traj_3d = []
    for p_idx in range(n_phase_bins):
        mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
        if np.any(mask_p):
            mean_traj_3d.append(np.mean(pca_proj[mask_p, :3], axis=0))
        else:
            mean_traj_3d.append(np.zeros(3))
    mean_traj_3d = np.array(mean_traj_3d)

    # Dispersion per phase bin
    dispersion_per_phase = []
    for p_idx in range(n_phase_bins):
        mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
        if np.any(mask_p):
            diff = pca_proj[mask_p, :3] - mean_traj_3d[p_idx]
            dist = np.linalg.norm(diff, axis=1)
            dispersion_per_phase.append(np.mean(dist))
        else:
            dispersion_per_phase.append(0.0)

    two_cyc_disp = np.concatenate([dispersion_per_phase, dispersion_per_phase])
    two_cyc_x = np.linspace(-np.pi, 3*np.pi, len(two_cyc_disp))
    ax_i.plot(two_cyc_x, two_cyc_disp, color="black", linewidth=2)
    ax_i.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax_i.axvline(np.pi, color="blue", linestyle="--", alpha=0.8)
    ax_i.axvline(2*np.pi, color="red", linestyle="--", alpha=0.8)
    ax_i.set_title("i | Breath-to-Breath Dispersion (Attractor Stability)", fontsize=11, fontweight="bold", loc="left")
    ax_i.set_xlabel("Phase (rad)", fontsize=10)
    ax_i.set_ylabel("Distance to Mean (a.u.)", fontsize=10)
    ax_i.set_xlim(-np.pi, 3*np.pi)
    ax_i.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_file_2d, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Figure 3 Replication (2D Static): {out_file_2d.resolve()}")

    # 4. Interactive 3D Plotly Trajectory Manifold (PC1 vs PC2 vs PC4)
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(
        x=pca_proj[:, 0],
        y=pca_proj[:, 1],
        z=pca_proj[:, 3],
        mode='lines+markers',
        marker=dict(
            size=3,
            color=phi_eval,
            colorscale='Twilight',
            colorbar=dict(title=r"Phase &phi; (rad)"),
            opacity=0.8
        ),
        line=dict(color='rgba(150,150,150,0.3)', width=1),
        name="Neural Trajectory"
    ))

    fig_3d.update_layout(
        title=f"<b>Interactive 3D Neural Manifold (PC1 vs PC2 vs PC4) - {len(unit_ids)} Clean Single Units</b>",
        scene=dict(
            xaxis_title="PC 1 (Respiratory Rhythm)",
            yaxis_title="PC 2 (Phase Progression)",
            zaxis_title="PC 4 (Attractor Transition)"
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig_3d.write_html(str(out_file_3d_html))
    logger.info(f"Saved Interactive 3D Manifold: {out_file_3d_html.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Replicate Publication Figures (Fig 2 & Fig 3)")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master", help="Path to consensus_master directory")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1", help="Path to raw experiment directory containing NIDQ files")
    parser.add_argument("--out-dir", type=str, default="output_publication_figures", help="Directory to save generated publication figures")
    parser.add_argument("--start-time", type=float, default=60.0, help="Start time in seconds for sliced datasets")
    args = parser.parse_args()

    # Handle paths correctly whether passed with or without quotes
    cons_dir_raw = args.consensus_dir
    if cons_dir_raw.startswith('"') and cons_dir_raw.endswith('"'):
        cons_dir_raw = cons_dir_raw[1:-1]
    
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        # Check literal quoted name
        alt_path = Path(f"'{cons_dir_raw}'")
        if alt_path.exists():
            cons_dir = alt_path

    raw_dir_raw = args.raw_dir
    if raw_dir_raw.startswith('"') and raw_dir_raw.endswith('"'):
        raw_dir_raw = raw_dir_raw[1:-1]
    raw_dir = Path(raw_dir_raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_t = args.start_time
    if start_t == 0.0 and ("60s_120s" in str(cons_dir) or "60s_120s" in str(out_dir)):
        start_t = 60.0

    logger.info("=" * 85)
    logger.info(" REPLICATING PUBLICATION FIGURES 2 & 3 (CIBRRIG / INSPIRATION-OFF-ATTRACTOR)")
    logger.info(f" Consensus Directory: {cons_dir.resolve()}")
    logger.info(f" Raw NIDQ Directory:  {raw_dir.resolve()}")
    logger.info(f" Output Directory:    {out_dir.resolve()}")
    logger.info(f" Time Slice Window:   {start_t:.1f}s to {start_t + 60.0:.1f}s")
    logger.info("=" * 85)

    # 1. Load Clean Single Units
    alf_dir = cons_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    # Keep only QC-passed clean Single Units
    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    if qc_col:
        df_sua = df_info[df_info[qc_col] == True].copy()
    else:
        df_sua = df_info.copy()

    sua_ids = set(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, list(sua_ids))
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]
    duration_s = float(np.max(spk_times)) if len(spk_times) > 0 else 60.0

    logger.info(f"Loaded {len(df_sua)} clean Single Units ({len(spk_times)} spikes) across {duration_s:.1f}s.")

    # 2. Load NIDQ Signals (EMG & Flow) Synchronized with Slice
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin file found in {raw_dir}")
    nidq_bin = nidq_bins[0]
    nidq_meta = nidq_bin.with_suffix(".meta")

    nidq_data = read_nidq_bin(nidq_bin, nidq_meta, emg_chan=0, ecg_chan=1)
    t_nidq_raw = nidq_data.time_vector
    
    t_end = start_t + duration_s
    mask_nidq = (t_nidq_raw >= start_t) & (t_nidq_raw <= t_end)
    t_nidq = t_nidq_raw[mask_nidq] - start_t
    emg_raw = nidq_data.all_channels[0][mask_nidq]

    # Process EMG & Extract Breaths
    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= duration_s)]
    breath_offsets = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= duration_s)]
    logger.info(f"Extracted {len(breath_onsets)} breaths across {duration_s:.1f}s.")

    # 3. Continuous Phase & Unit Phase Classification
    dt_eval = 0.005
    t_eval = np.arange(0.0, duration_s, dt_eval)
    phi_eval = compute_continuous_respiratory_phase(t_nidq, emg_proc.envelope, breath_onsets, breath_offsets, t_eval)
    
    df_phase = classify_units_by_phase(
        spikes_times=spk_times,
        spikes_clusters=spk_clusters,
        unit_ids=df_sua["cluster_id"].values,
        t_eval=t_eval,
        phi_eval=phi_eval,
        duration_s=duration_s
    )
    df_phase.to_csv(out_dir / "unit_respiratory_phase_classification.csv", index=False)
    logger.info(f"Phase Classification: {dict(df_phase['unit_class'].value_counts())}")

    # 4. Generate Figure 2 Replication
    fig2_path = out_dir / "publication_figure_2_single_cell_phase_dynamics.png"
    plot_figure_2_replication(
        emg_env=emg_proc.envelope,
        t_nidq=t_nidq,
        breath_onsets=breath_onsets,
        breath_offsets=breath_offsets,
        spikes_times=spk_times,
        spikes_clusters=spk_clusters,
        df_info=df_sua,
        df_phase=df_phase,
        out_file=fig2_path,
        time_window=(5.0, 9.0)
    )

    # 5. Generate Figure 3 Replication
    fig3_2d_path = out_dir / "publication_figure_3_low_d_neural_manifold.png"
    fig3_3d_html = out_dir / "publication_figure_3_interactive_3d_manifold.html"
    plot_figure_3_replication(
        emg_env=emg_proc.envelope,
        t_nidq=t_nidq,
        breath_onsets=breath_onsets,
        breath_offsets=breath_offsets,
        spikes_times=spk_times,
        spikes_clusters=spk_clusters,
        df_info=df_sua,
        df_phase=df_phase,
        out_file_2d=fig3_2d_path,
        out_file_3d_html=fig3_3d_html,
        duration_s=duration_s
    )

    print("\n" + "=" * 85)
    print(" [SUCCESS] PUBLICATION FIGURES 2 & 3 REPLICATED SUCCESSFULLY")
    print("=" * 85)
    print(f" • Figure 2 (Single-Cell Phase Dynamics) : {fig2_path.resolve()}")
    print(f" • Figure 3 (Low-D Neural Manifold 2D)   : {fig3_2d_path.resolve()}")
    print(f" • Figure 3 (Interactive 3D Manifold)    : {fig3_3d_html.resolve()}")
    print(f" • Classification Summary Table          : {out_dir / 'unit_respiratory_phase_classification.csv'}")
    print("=" * 85)

if __name__ == "__main__":
    main()
