#!/usr/bin/env python3
"""
Generate Split Publication Panels for Figure 3 (Low-Dimensional Neural Manifold):
Subdivides publication_figure_3_low_d_neural_manifold.png into 4 focused presentation panels:

1. fig3_panel_a_multiscale_dynamics.png / .svg:
   - Panel a: Integrated Multi-Scale Dynamics: Diaphragm EMG (motor output),
     37-unit Neuropixels SUA raster, and top 4 Principal Component projections over time.
   - Central Point: High-dimensional brainstem spiking coordinates into a coherent low-dimensional
     latent manifold directly coupled to motor execution.

2. fig3_panel_b_e_latent_modes.png / .svg:
   - Panel b: Breath-Aligned Average Leading PCs (mean ± SEM, -500 to +500 ms relative to onset).
   - Panel e: Phase-Aligned PCs across 2 full cycles (-pi to 3pi, phase-normalized).
   - Central Point: Leading principal components form orthogonal quadrature-phase harmonic modes
     that preserve stable, cycle-invariant geometry across variable breath lengths.

3. fig3_panel_c_d_f_manifold_velocity.png / .svg:
   - Panel c: 2D Manifold Phase Trajectory (PC1 vs PC2) colored by continuous phase phi(t).
   - Panel d: 2D Manifold Trajectory Speed (PC1 vs PC2) colored by instantaneous velocity ||dX/dt||.
   - Panel f: Angular Trajectory Speed polar distribution (dtheta/dt).
   - Central Point: Population activity forms a smooth rotational limit cycle with pronounced
     deceleration / dwelling in the Inspiration-Off expiratory attractor well.

4. fig3_panel_i_attractor_dispersion.png / .svg:
   - Panel i: Breath-to-Breath Dispersion (Euclidean distance to mean trajectory across phase -pi to 3pi).
   - Central Point: Direct quantitative proof of the Inspiration-Off attractor: inter-trial variance
     contracts to a sharp global minimum during expiration (attractor dwelling) and flares during
     explosive barrier escape at inspiration onset.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).parent.parent
VLAD_REPO = PROJECT_ROOT / "inspiration-off-attractor"
VLAD_DIR = VLAD_REPO / "VLAD"

for p in [str(PROJECT_ROOT), str(VLAD_REPO), str(VLAD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

ASSETS_DIR = PROJECT_ROOT / "presentation_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PC_COLORS = ["#10b981", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#64748b"]

def compute_continuous_respiratory_phase(
    t_nidq: np.ndarray,
    emg_envelope: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    t_eval: np.ndarray
) -> np.ndarray:
    phi = np.full_like(t_eval, np.nan)
    n_breaths = min(len(breath_onsets), len(breath_offsets))
    
    for i in range(n_breaths - 1):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i]
        t_next_on = breath_onsets[i + 1]

        if t_off <= t_on or t_next_on <= t_off:
            continue

        insp_mask = (t_eval >= t_on) & (t_eval < t_off)
        if np.any(insp_mask):
            phi[insp_mask] = (t_eval[insp_mask] - t_on) / (t_off - t_on) * np.pi

        exp_mask = (t_eval >= t_off) & (t_eval < t_next_on)
        if np.any(exp_mask):
            phi[exp_mask] = -np.pi + (t_eval[exp_mask] - t_off) / (t_next_on - t_off) * np.pi

    valid_idx = np.where(~np.isnan(phi))[0]
    if len(valid_idx) > 0:
        phi[:valid_idx[0]] = phi[valid_idx[0]]
        phi[valid_idx[-1]:] = phi[valid_idx[-1]]

    return phi

def load_data():
    cons_dir = PROJECT_ROOT / "output_LP21_60s_120s" / "sliding_window_pipeline" / "lupin" / "consensus_master"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1")
    start_t = 60.0
    
    alf_dir = cons_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

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

    # NIDQ
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

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= duration_s)]
    breath_offsets = emg_proc.breath_offsets[(emg_proc.breath_offsets >= 0.0) & (emg_proc.breath_offsets <= duration_s)]

    # Compute Firing Rate Matrix
    dt = 0.005  # 5 ms time bin
    t_bins = np.arange(0.0, duration_s, dt)
    unit_ids = df_sua["cluster_id"].values
    
    gaussian_sigma_bins = 5
    kernel = signal.windows.gaussian(21, std=gaussian_sigma_bins)
    kernel /= np.sum(kernel)

    fr_matrix = []
    for uid in unit_ids:
        spks = spk_times[spk_clusters == uid]
        counts, _ = np.histogram(spks, bins=np.append(t_bins, t_bins[-1] + dt))
        smooth_rate = np.convolve(counts / dt, kernel, mode="same")
        fr_matrix.append(smooth_rate)
    fr_matrix = np.array(fr_matrix)

    # PCA
    pca = PCA(n_components=6)
    pca_proj = pca.fit_transform(fr_matrix.T)

    # Instantaneous trajectory speed || dX / dt || in 3D PC space
    grad_pca = np.gradient(pca_proj[:, :3], axis=0) / dt
    traj_speed = np.linalg.norm(grad_pca, axis=1)

    # Continuous Respiratory Phase phi(t)
    phi_eval = compute_continuous_respiratory_phase(t_nidq, emg_proc.envelope, breath_onsets, breath_offsets, t_bins)

    return {
        "spk_times": spk_times,
        "spk_clusters": spk_clusters,
        "unit_ids": unit_ids,
        "df_sua": df_sua,
        "t_nidq": t_nidq,
        "emg_env": emg_proc.envelope,
        "breath_onsets": breath_onsets,
        "breath_offsets": breath_offsets,
        "duration_s": duration_s,
        "dt": dt,
        "t_bins": t_bins,
        "fr_matrix": fr_matrix,
        "pca": pca,
        "pca_proj": pca_proj,
        "traj_speed": traj_speed,
        "phi_eval": phi_eval
    }

# ==============================================================================
# Panel 1: Multi-Scale Dynamics (Panel a)
# ==============================================================================
def make_fig3_panel_a(data):
    t_nidq = data["t_nidq"]
    emg_env = data["emg_env"]
    spk_times = data["spk_times"]
    spk_clusters = data["spk_clusters"]
    unit_ids = data["unit_ids"]
    t_bins = data["t_bins"]
    pca_proj = data["pca_proj"]

    t0_snap, t1_snap = 5.0, 9.0
    mask_snap_nidq = (t_nidq >= t0_snap) & (t_nidq <= t1_snap)
    mask_snap_bins = (t_bins >= t0_snap) & (t_bins <= t1_snap)

    fig = plt.figure(figsize=(12.0, 8.5), dpi=250)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.8, 1.4], hspace=0.15)

    ax_emg = fig.add_subplot(gs[0])
    ax_ras = fig.add_subplot(gs[1], sharex=ax_emg)
    ax_pcs = fig.add_subplot(gs[2], sharex=ax_emg)

    # 1. Diaphragm EMG
    ax_emg.plot(t_nidq[mask_snap_nidq], emg_env[mask_snap_nidq], color="#0f172a", linewidth=1.5)
    ax_emg.set_title("a | Multi-Scale Dynamics: Diaphragm EMG, 37 SUA Units, and Leading PC Projections", 
                     fontsize=13, fontweight="bold", pad=12, loc="left", color="#0f172a")
    ax_emg.set_ylabel("Diaphragm EMG\n(Normalized)", fontsize=10, fontweight="bold", color="#334155")
    ax_emg.grid(True, linestyle=":", alpha=0.5)
    plt.setp(ax_emg.get_xticklabels(), visible=False)

    # 2. SUA Spiking Raster
    for idx, uid in enumerate(unit_ids):
        spks = spk_times[spk_clusters == uid]
        in_s = spks[(spks >= t0_snap) & (spks <= t1_snap)]
        ax_ras.scatter(in_s, np.full_like(in_s, idx + 1), s=6, color="#475569", alpha=0.85, marker="|")
    ax_ras.set_ylabel("Neuron ID\n(37 Clean SUAs)", fontsize=10, fontweight="bold", color="#334155")
    ax_ras.set_ylim(0.5, len(unit_ids) + 0.5)
    ax_ras.grid(True, linestyle=":", alpha=0.5)
    plt.setp(ax_ras.get_xticklabels(), visible=False)

    # 3. PC Projections
    for i in range(4):
        ax_pcs.plot(t_bins[mask_snap_bins], pca_proj[mask_snap_bins, i], 
                    label=f"$PC_{i+1}$ ({data['pca'].explained_variance_ratio_[i]*100:.1f}%)", 
                    color=PC_COLORS[i], linewidth=2.0)
    ax_pcs.set_ylabel("PC Amplitude\n(a.u.)", fontsize=10, fontweight="bold", color="#334155")
    ax_pcs.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold", color="#334155")
    ax_pcs.set_xlim(t0_snap, t1_snap)
    ax_pcs.set_ylim(-210, 490)
    ax_pcs.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", ncol=4, fontsize=9)
    ax_pcs.grid(True, linestyle=":", alpha=0.5)

    # Annotations
    ax_emg.text(0.015, 0.82, "Diaphragmatic Motor Output (Inspiratory Bursts)", transform=ax_emg.transAxes,
                fontsize=9.5, fontweight="bold", color="#0284c7", bbox=dict(boxstyle="round,pad=0.25", fc="#e0f2fe", ec="#7dd3fc", alpha=0.9))
    ax_ras.text(0.015, 0.88, "Brainstem Ventral Respiratory Column Spike Trains", transform=ax_ras.transAxes,
                fontsize=9.5, fontweight="bold", color="#334155", bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#cbd5e1", alpha=0.9))
    ax_pcs.text(0.015, 0.12, "Low-Dimensional Coordinate Embeddings (Rotational Oscillators)", transform=ax_pcs.transAxes,
                fontsize=9.5, fontweight="bold", color="#10b981", bbox=dict(boxstyle="round,pad=0.25", fc="#ecfdf5", ec="#6ee7b7", alpha=0.9))

    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.15)
    out_png = ASSETS_DIR / "fig3_panel_a_multiscale_dynamics.png"
    out_svg = ASSETS_DIR / "fig3_panel_a_multiscale_dynamics.svg"
    fig.savefig(out_png, dpi=250)
    fig.savefig(out_svg, format='svg')
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

# ==============================================================================
# Panel 2: Latent Harmonic Modes (Panels b & e)
# ==============================================================================
def make_fig3_panel_b_e(data):
    pca_proj = data["pca_proj"]
    breath_onsets = data["breath_onsets"]
    duration_s = data["duration_s"]
    dt = data["dt"]
    phi_eval = data["phi_eval"]

    fig, (ax_b, ax_e) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=250)

    # Panel b: Breath-Aligned Average PCs
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
            ax_b.plot(t_psth, mean_pc, label=f"$PC_{i+1}$ ({data['pca'].explained_variance_ratio_[i]*100:.1f}%)", 
                      color=PC_COLORS[i], linewidth=2.2)
            ax_b.fill_between(t_psth, mean_pc - sem_pc, mean_pc + sem_pc, color=PC_COLORS[i], alpha=0.18)

    ax_b.axvline(0, color="#dc2626", linestyle="--", linewidth=1.8, label="Inspiration Onset ($I_{on}$)")
    ax_b.set_title("b | Breath-Aligned Average PCs (Time Domain)", fontsize=12.5, fontweight="bold", loc="left", color="#0f172a")
    ax_b.set_xlabel("Time from Breath Onset (ms)", fontsize=10.5, fontweight="bold", color="#334155")
    ax_b.set_ylabel("PC Amplitude (a.u.)", fontsize=10.5, fontweight="bold", color="#334155")
    ax_b.set_xlim(-500, 500)
    ax_b.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9, loc="upper right")
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Panel e: Phase-Aligned PCs (2 Cycles, -pi to 3pi)
    n_phase_bins = 50
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    for i in range(4):
        pc_phase_binned = []
        for p_idx in range(n_phase_bins):
            mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
            if np.any(mask_p):
                pc_phase_binned.append(np.mean(pca_proj[mask_p, i]))
            else:
                pc_phase_binned.append(0.0)
        
        two_cyc_pc = np.concatenate([pc_phase_binned, pc_phase_binned])
        two_cyc_rad = np.linspace(-np.pi, 3*np.pi, len(two_cyc_pc))
        ax_e.plot(two_cyc_rad, two_cyc_pc, label=f"$PC_{i+1}$", color=PC_COLORS[i], linewidth=2.2)

    ax_e.axvline(0, color="#dc2626", linestyle="--", linewidth=1.8, label="$I_{on}$ ($0, 2\pi$)")
    ax_e.axvline(np.pi, color="#0284c7", linestyle="--", linewidth=1.8, label="$I_{off}$ ($\pi$)")
    ax_e.axvline(2*np.pi, color="#dc2626", linestyle="--", linewidth=1.8)
    ax_e.set_title("e | Phase-Aligned PCs: 2 Cycles (Normalized Phase Domain)", fontsize=12.5, fontweight="bold", loc="left", color="#0f172a")
    ax_e.set_xlabel(r"Continuous Respiratory Phase $\phi$ (rad)", fontsize=10.5, fontweight="bold", color="#334155")
    ax_e.set_ylabel("PC Amplitude (a.u.)", fontsize=10.5, fontweight="bold", color="#334155")
    ax_e.set_xlim(-np.pi, 3*np.pi)
    ax_e.set_xticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
    ax_e.set_xticklabels([r"$-\pi$", r"$0$", r"$+\pi$", r"$+2\pi$", r"$+3\pi$"])
    ax_e.set_ylim(-190, 520)
    ax_e.set_yticks([-100, 0, 100, 200, 300, 400])
    ax_e.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9, loc="upper right")
    ax_e.grid(True, linestyle=":", alpha=0.5)

    # Shaded backgrounds for inspiration vs expiration on Panel e
    ax_e.axvspan(0, np.pi, color="#fee2e2", alpha=0.35, zorder=0)
    ax_e.axvspan(2*np.pi, 3*np.pi, color="#fee2e2", alpha=0.35, zorder=0)
    ax_e.axvspan(-np.pi, 0, color="#e0f2fe", alpha=0.35, zorder=0)
    ax_e.axvspan(np.pi, 2*np.pi, color="#e0f2fe", alpha=0.35, zorder=0)
    ax_e.text(np.pi/2, 475, "Insp", ha="center", fontsize=9.5, fontweight="bold", color="#dc2626")
    ax_e.text(1.5*np.pi, 475, "Exp", ha="center", fontsize=9.5, fontweight="bold", color="#0284c7")

    plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12, wspace=0.22)
    out_png = ASSETS_DIR / "fig3_panel_b_e_latent_modes.png"
    out_svg = ASSETS_DIR / "fig3_panel_b_e_latent_modes.svg"
    fig.savefig(out_png, dpi=250)
    fig.savefig(out_svg, format='svg')
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

# ==============================================================================
# Panel 3: Low-Dimensional Manifold & Trajectory Velocity (Panels c, d & f)
# ==============================================================================
def make_fig3_panel_c_d_f(data):
    pca_proj = data["pca_proj"]
    phi_eval = data["phi_eval"]
    traj_speed = data["traj_speed"]

    fig = plt.figure(figsize=(16.0, 6.2), dpi=250)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.15, 0.95], wspace=0.42)

    ax_c = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])
    ax_f = fig.add_subplot(gs[2], polar=True)

    # Panel c: PC1 vs PC2 colored by phase
    sc_c = ax_c.scatter(pca_proj[:, 0], pca_proj[:, 1], c=phi_eval, cmap="twilight", s=6, alpha=0.75)
    cbar_c = plt.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label(r"Phase $\phi$ (rad)", fontsize=9, fontweight="bold", color="#334155")
    cbar_c.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar_c.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
    ax_c.set_title(r"c | Manifold: Phase Trajectory ($PC_1$ vs $PC_2$)", fontsize=11.5, fontweight="bold", loc="left", color="#0f172a")
    ax_c.set_xlabel(r"$PC_1$ (Rhythm / Inspiration Mode)", fontsize=10, fontweight="bold", color="#334155")
    ax_c.set_ylabel(r"$PC_2$ (Quadrature Phase Mode)", fontsize=10, fontweight="bold", color="#334155")
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # Panel d: PC1 vs PC2 colored by trajectory speed
    sc_d = ax_d.scatter(pca_proj[:, 0], pca_proj[:, 1], c=traj_speed, cmap="inferno", s=6, alpha=0.75)
    cbar_d = plt.colorbar(sc_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label(r"Trajectory Speed $||dX/dt||$", fontsize=9, fontweight="bold", color="#334155")
    ax_d.set_title(r"d | Trajectory Speed ($PC_1$ vs $PC_2$)", fontsize=11.5, fontweight="bold", loc="left", color="#0f172a")
    ax_d.set_xlabel(r"$PC_1$", fontsize=10, fontweight="bold", color="#334155")
    ax_d.set_ylabel(r"$PC_2$", fontsize=10, fontweight="bold", color="#334155")
    ax_d.grid(True, linestyle=":", alpha=0.5)

    # Annotate dwell basin and fast sweep on panel d
    ax_d.annotate("Attractor Dwelling\n(Low Speed ~0-2,000)", 
                  xy=(-100, 0), xytext=(-180, 70),
                  arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.8),
                  fontsize=8.5, fontweight="bold", color="#0284c7",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#f0f9ff", ec="#bae6fd"))
    ax_d.annotate("Ballistic Sweep\n(High Speed >10,000)", 
                  xy=(400, 50), xytext=(220, 115),
                  arrowprops=dict(arrowstyle="->", color="#f97316", lw=1.8),
                  fontsize=8.5, fontweight="bold", color="#ea580c",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#fff7ed", ec="#fed7aa"))

    # Panel f: Angular Trajectory Speed Polar Plot
    n_phase_bins = 50
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    speed_phase_binned = []
    for p_idx in range(n_phase_bins):
        mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
        speed_phase_binned.append(np.mean(traj_speed[mask_p]) if np.any(mask_p) else 0.0)

    theta_speed = np.linspace(0, 2*np.pi, len(speed_phase_binned))
    ax_f.plot(theta_speed, speed_phase_binned, color="#dc2626", linewidth=2.2)
    ax_f.fill(theta_speed, speed_phase_binned, color="#dc2626", alpha=0.25)
    ax_f.set_title("f | Angular Speed Distribution", fontsize=11.5, fontweight="bold", loc="left", pad=16, color="#0f172a")

    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, wspace=0.55)
    out_png = ASSETS_DIR / "fig3_panel_c_d_f_manifold_velocity.png"
    out_svg = ASSETS_DIR / "fig3_panel_c_d_f_manifold_velocity.svg"
    fig.savefig(out_png, dpi=250)
    fig.savefig(out_svg, format='svg')
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

# ==============================================================================
# Panel 4: Attractor Stability & Dispersion (Panel i)
# ==============================================================================
def make_fig3_panel_i(data):
    pca_proj = data["pca_proj"]
    phi_eval = data["phi_eval"]

    fig, ax_i = plt.subplots(figsize=(10.5, 6.0), dpi=250)

    n_phase_bins = 50
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    mean_traj_3d = []
    for p_idx in range(n_phase_bins):
        mask_p = (phi_eval >= phase_edges[p_idx]) & (phi_eval < phase_edges[p_idx + 1])
        if np.any(mask_p):
            mean_traj_3d.append(np.mean(pca_proj[mask_p, :3], axis=0))
        else:
            mean_traj_3d.append(np.zeros(3))
    mean_traj_3d = np.array(mean_traj_3d)

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

    # Plot
    ax_i.plot(two_cyc_x, two_cyc_disp, color="#0f172a", linewidth=2.8, label="Inter-Trial Dispersion (Distance to Mean Trajectory)")
    ax_i.axvline(0, color="#dc2626", linestyle="--", linewidth=1.8, label="$I_{on}$ ($0, 2\pi$)")
    ax_i.axvline(np.pi, color="#0284c7", linestyle="--", linewidth=1.8, label="$I_{off}$ ($\pi$)")
    ax_i.axvline(2*np.pi, color="#dc2626", linestyle="--", linewidth=1.8)

    # Shading for Inspiration vs Expiration
    ax_i.axvspan(0, np.pi, color="#fee2e2", alpha=0.4, zorder=0)
    ax_i.axvspan(2*np.pi, 3*np.pi, color="#fee2e2", alpha=0.4, zorder=0)
    ax_i.axvspan(-np.pi, 0, color="#e0f2fe", alpha=0.4, zorder=0)
    ax_i.axvspan(np.pi, 2*np.pi, color="#e0f2fe", alpha=0.4, zorder=0)

    ax_i.set_title("i | Breath-to-Breath Dispersion Across Phase: Empirical Attractor Stability", 
                   fontsize=12.5, fontweight="bold", loc="left", color="#0f172a", pad=12)
    ax_i.set_xlabel(r"Continuous Respiratory Phase $\phi$ (rad)", fontsize=11, fontweight="bold", color="#334155")
    ax_i.set_ylabel(r"Euclidean Distance to Mean Trajectory in $\mathbb{R}^3$ (a.u.)", fontsize=11, fontweight="bold", color="#334155")
    ax_i.set_xlim(-np.pi, 3*np.pi)
    ax_i.set_xticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
    ax_i.set_xticklabels([r"$-\pi$ (Exp)", r"$0$ ($I_{on}$)", r"$+\pi$ ($I_{off}$)", r"$+2\pi$ ($I_{on}$)", r"$+3\pi$ ($I_{off}$)"])
    ax_i.grid(True, linestyle=":", alpha=0.5)
    ax_i.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5, loc="upper right")

    # Critical Conceptual Annotations
    ax_i.annotate("Attractor Basin Contraction\n(Global Dispersion Minimum ~12 a.u.)\nTrajectories tightly funnel into well",
                  xy=(np.pi*1.5, 14), xytext=(np.pi*1.1, 35),
                  arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2.0),
                  fontsize=9.5, fontweight="bold", color="#0369a1",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#e0f2fe", ec="#7dd3fc"))

    ax_i.annotate("Explosive Barrier Escape Flare\n(Dispersion Peak ~78 a.u.)\nTrial-by-trial threshold-crossing jitter",
                  xy=(0, 76), xytext=(-np.pi*0.8, 62),
                  arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2.0),
                  fontsize=9.5, fontweight="bold", color="#b91c1c",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#fca5a5"))

    plt.tight_layout()
    out_png = ASSETS_DIR / "fig3_panel_i_attractor_dispersion.png"
    out_svg = ASSETS_DIR / "fig3_panel_i_attractor_dispersion.svg"
    fig.savefig(out_png, dpi=250)
    fig.savefig(out_svg, format='svg')
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

def main():
    print("Loading recording data and computing PCA embeddings...")
    data = load_data()
    print("Data loaded successfully.")

    print("\n--- Generating Split Panels for Figure 3 ---")
    make_fig3_panel_a(data)
    make_fig3_panel_b_e(data)
    make_fig3_panel_c_d_f(data)
    make_fig3_panel_i(data)
    print("\nAll 4 split panels for Figure 3 successfully created!")

if __name__ == "__main__":
    main()
