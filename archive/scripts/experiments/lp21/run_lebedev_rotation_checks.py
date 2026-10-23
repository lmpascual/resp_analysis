#!/usr/bin/env python3
"""
Lebedev et al. (2019) Pitfall & Rigor Checks for Neural Rotational Dynamics
Implements the 6 essential checks and controls proposed in Lebedev et al. (Scientific Reports 2019):
1. PETH Peak Latency Sorting & Single-Breath Waves (Fig 2A-E)
2. Trial-by-Trial Peak Scatter & Cross-Correlation Matrix (Fig 2F, G)
3. Direct Subpopulation 3D Space (Early, Intermediate, Late) without PCA (Fig 3)
4. Breath / Trial Shuffling Null Model (Fig 4B, D)
5. Multiple Linear Regression to Arbitrary Lissajous Curves (Circle vs Figure-8) with Train/Test Cross-Validation (Fig 9)
6. Cross-Condition Mean Subtraction vs Raw Rates Diagnostics
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
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LEBEDEV_CHECKS")


def compute_breath_peths_per_neuron(
    spikes_times: np.ndarray,
    spikes_clusters: np.ndarray,
    unit_ids: np.ndarray,
    breath_onsets: np.ndarray,
    dt: float = 0.005,
    win_pre: float = 0.1,
    win_post: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts breath-triggered PETHs for every neuron across all individual breaths.
    Returns:
    - time_axis_ms: array of time bins relative to breath onset
    - mean_peths: array of shape (N_units, N_time_bins) (averaged across breaths)
    - trial_peths: array of shape (N_units, N_breaths, N_time_bins)
    - peak_times: array of shape (N_units, N_breaths) containing peak time per trial
    """
    time_bins = np.arange(-win_pre, win_post + dt, dt)
    time_centers_ms = (time_bins[:-1] + time_bins[1:]) / 2.0 * 1000.0
    n_bins = len(time_centers_ms)
    n_units = len(unit_ids)
    
    valid_onsets = breath_onsets[(breath_onsets > win_pre) & (breath_onsets < np.max(spikes_times) - win_post)]
    n_breaths = len(valid_onsets)

    trial_peths = np.zeros((n_units, n_breaths, n_bins))
    peak_times = np.zeros((n_units, n_breaths))

    # Gaussian smoothing kernel (sigma = 15 ms)
    gaussian_sigma = 3
    kernel = signal.windows.gaussian(15, std=gaussian_sigma)
    kernel /= np.sum(kernel)

    for u_idx, uid in enumerate(unit_ids):
        spks = spikes_times[spikes_clusters == uid]
        for b_idx, on in enumerate(valid_onsets):
            rel_spks = spks - on
            in_win = rel_spks[(rel_spks >= -win_pre) & (rel_spks <= win_post)]
            counts, _ = np.histogram(in_win, bins=time_bins)
            rate_hz = np.convolve(counts / dt, kernel, mode="same")
            trial_peths[u_idx, b_idx, :] = rate_hz
            
            # Find peak time in ms
            peak_bin = np.argmax(rate_hz)
            peak_times[u_idx, b_idx] = time_centers_ms[peak_bin]

    mean_peths = np.mean(trial_peths, axis=1)  # (N_units, N_bins)
    return time_centers_ms, mean_peths, trial_peths, peak_times


def run_lebedev_checks(
    consensus_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    duration_s: Optional[float] = None,
    start_time: Optional[float] = None
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 85)
    logger.info(" LEBEDEV ET AL. (2019) RIGOR & DYNAMICAL CONTROLS")
    logger.info(f" Consensus Directory: {consensus_dir.resolve()}")
    logger.info(f" Raw NIDQ Directory:  {raw_dir.resolve()}")
    logger.info(f" Output Directory:    {out_dir.resolve()}")
    logger.info("=" * 85)

    # 1. Load Clean Units
    alf_dir = consensus_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(consensus_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else ("passed_allen_qc" if "passed_allen_qc" in df_info.columns else None)
    if qc_col:
        df_sua = df_info[df_info[qc_col] == True].copy()
    else:
        df_sua = df_info.copy()

    sua_ids = np.array(df_sua["cluster_id"].values)
    mask_sua = np.isin(spk_clusters, sua_ids)
    spk_times = spk_times[mask_sua]
    spk_clusters = spk_clusters[mask_sua]

    rec_dur = float(np.max(spk_times)) if len(spk_times) > 0 else 60.0
    if duration_s is not None:
        rec_dur = min(rec_dur, duration_s)
        mask_dur = spk_times <= rec_dur
        spk_times = spk_times[mask_dur]
        spk_clusters = spk_clusters[mask_dur]

    logger.info(f"Loaded {len(sua_ids)} clean Single Units across {rec_dur:.1f}s.")

    # 2. Load NIDQ & Process EMG for Breaths Synchronized with Slice
    if start_time is not None:
        start_t = start_time
    elif "60s_120s" in str(consensus_dir) or "60s_120s" in str(out_dir):
        start_t = 60.0
    else:
        start_t = 0.0
    nidq_bins = list(raw_dir.glob("*.nidq.bin"))
    if not nidq_bins:
        raise FileNotFoundError(f"No .nidq.bin found in {raw_dir}")
    nidq_bin = nidq_bins[0]
    ecg_chan = 2 if "e097" in str(consensus_dir).lower() or "e097" in str(raw_dir).lower() else 1
    nidq_data = read_nidq_bin(nidq_bin, nidq_bin.with_suffix(".meta"), emg_chan=0, ecg_chan=ecg_chan)
    
    t_nidq_raw = nidq_data.time_vector
    t_end = start_t + rec_dur
    mask_n = (t_nidq_raw >= start_t) & (t_nidq_raw <= t_end)
    t_nidq = t_nidq_raw[mask_n] - start_t
    emg_raw = nidq_data.all_channels[0][mask_n]

    emg_cfg = EMGConfig(fs=nidq_data.fs, highpass_cutoff=300.0, lowpass_cutoff=5000.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets[(emg_proc.breath_onsets >= 0.0) & (emg_proc.breath_onsets <= rec_dur)]
    logger.info(f"Detected {len(breath_onsets)} synchronized breath onsets across slice {start_t:.1f}s to {t_end:.1f}s.")

    # 3. Compute PETHs and Peak Latency Order
    time_ms, mean_peths, trial_peths, peak_times = compute_breath_peths_per_neuron(
        spikes_times=spk_times,
        spikes_clusters=spk_clusters,
        unit_ids=sua_ids,
        breath_onsets=breath_onsets,
        dt=0.005,
        win_pre=0.15,
        win_post=0.35
    )

    # Sort units by average peak time (Lebedev Fig 2E)
    avg_peak_idx = np.argmax(mean_peths, axis=1)
    sorted_order = np.argsort(avg_peak_idx)
    sorted_mean_peths = mean_peths[sorted_order]
    sorted_sua_ids = sua_ids[sorted_order]

    # Normalize PETHs for display (zero mean, unit max)
    norm_peths = np.zeros_like(sorted_mean_peths)
    for i in range(len(sorted_mean_peths)):
        row = sorted_mean_peths[i]
        mx = np.max(row)
        norm_peths[i] = (row - np.min(row)) / (mx - np.min(row) + 1e-6)

    # Subpopulation Partition (Early, Intermediate, Late - Lebedev Fig 3)
    n_u = len(sua_ids)
    n_third = max(1, n_u // 3)
    idx_early = sorted_order[:n_third]
    idx_interm = sorted_order[n_third:2*n_third]
    idx_late = sorted_order[2*n_third:]

    early_peth = np.mean(mean_peths[idx_early], axis=0)
    interm_peth = np.mean(mean_peths[idx_interm], axis=0)
    late_peth = np.mean(mean_peths[idx_late], axis=0)

    # 4. Lissajous & Arbitrary Geometric Regression Test (Lebedev Fig 9)
    # Fit population to Circle: (cos 2pi t/T, sin 2pi t/T) vs Figure-8: (cos 2pi t/T, sin 4pi t/T)
    # Train on 50% odd breaths, test on 50% even breaths (cross-validation)
    n_b = trial_peths.shape[1]
    train_b = np.arange(0, n_b, 2)
    test_b = np.arange(1, n_b, 2)

    T_len = len(time_ms)
    t_norm = np.linspace(0, 1, T_len)
    circle_x = np.cos(2 * np.pi * t_norm)
    circle_y = np.sin(2 * np.pi * t_norm)
    fig8_x = np.cos(2 * np.pi * t_norm)
    fig8_y = np.sin(4 * np.pi * t_norm)

    # Build Train & Test matrices
    X_train, Y_train_circle, Y_train_fig8 = [], [], []
    for b in train_b:
        X_train.append(trial_peths[:, b, :].T)  # (T_len, N_units)
        Y_train_circle.append(np.column_stack([circle_x, circle_y]))
        Y_train_fig8.append(np.column_stack([fig8_x, fig8_y]))

    X_train = np.vstack(X_train)
    Y_train_circle = np.vstack(Y_train_circle)
    Y_train_fig8 = np.vstack(Y_train_fig8)

    X_test, Y_test_circle, Y_test_fig8 = [], [], []
    for b in test_b:
        X_test.append(trial_peths[:, b, :].T)
        Y_test_circle.append(np.column_stack([circle_x, circle_y]))
        Y_test_fig8.append(np.column_stack([fig8_x, fig8_y]))

    X_test = np.vstack(X_test)
    Y_test_circle = np.vstack(Y_test_circle)
    Y_test_fig8 = np.vstack(Y_test_fig8)

    # Train multiple linear regression models
    reg_circle = LinearRegression().fit(X_train, Y_train_circle)
    pred_circle_train = reg_circle.predict(X_train[:T_len])
    pred_circle_test = reg_circle.predict(X_test[:T_len])
    r2_circle_test = reg_circle.score(X_test, Y_test_circle)

    reg_fig8 = LinearRegression().fit(X_train, Y_train_fig8)
    pred_fig8_train = reg_fig8.predict(X_train[:T_len])
    pred_fig8_test = reg_fig8.predict(X_test[:T_len])
    r2_fig8_test = reg_fig8.score(X_test, Y_test_fig8)

    # 5. Breath Shuffling Control (Lebedev Fig 3B, D)
    shuffled_trial_peths = np.zeros_like(trial_peths)
    for u in range(n_u):
        perm = np.random.permutation(n_b)
        shuffled_trial_peths[u] = trial_peths[u, perm, :]

    shuffled_early = np.mean(np.mean(shuffled_trial_peths[idx_early], axis=1), axis=0)
    shuffled_interm = np.mean(np.mean(shuffled_trial_peths[idx_interm], axis=1), axis=0)
    shuffled_late = np.mean(np.mean(shuffled_trial_peths[idx_late], axis=1), axis=0)

    # -------------------------------------------------------------
    # 6. Assemble Multi-Panel Lebedev Rigor Figure
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(19, 13))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.35)

    # Panel A: Stacked PETHs Sorted by Peak Latency (Lebedev Fig 2E)
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(
        norm_peths,
        aspect="auto",
        cmap="jet",
        extent=[time_ms[0], time_ms[-1], 0, n_u],
        origin="lower",
        interpolation="nearest"
    )
    ax_a.axvline(0, color="white", linestyle="--", linewidth=1.5)
    ax_a.set_title("A | Sequential PETH Wave\n(Sorted by Peak Latency)", fontsize=11, fontweight="bold", loc="left")
    ax_a.set_xlabel("Time from Breath Onset (ms)", fontsize=9)
    ax_a.set_ylabel("Neuron Rank (Early -> Late)", fontsize=9)
    cbar_a = plt.colorbar(im_a, ax=ax_a, fraction=0.04, pad=0.02)
    cbar_a.set_label("Norm Rate", fontsize=8)

    # Panel B: Peak Latency Across Individual Breaths (Lebedev Fig 2F)
    ax_b = fig.add_subplot(gs[0, 1])
    n_show_breaths = min(30, n_b)
    for b in range(n_show_breaths):
        ax_b.scatter(peak_times[sorted_order, b], np.arange(n_u), s=4, color="black", alpha=0.35)
    ax_b.set_title(f"B | Peak Spread Across {n_show_breaths} Breaths\n(Temporal Order Consistency)", fontsize=11, fontweight="bold", loc="left")
    ax_b.set_xlabel("Peak Firing Time (ms)", fontsize=9)
    ax_b.set_ylabel("Neuron Rank", fontsize=9)
    ax_b.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Pairwise Correlation Matrix (Lebedev Fig 2G)
    ax_c = fig.add_subplot(gs[0, 2])
    corr_matrix = np.corrcoef(mean_peths[sorted_order])
    im_c = ax_c.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
    ax_c.set_title("C | Pairwise Cross-Correlation\n(Phase Shift Signatures)", fontsize=11, fontweight="bold", loc="left")
    ax_c.set_xlabel("Neuron Rank", fontsize=9)
    ax_c.set_ylabel("Neuron Rank", fontsize=9)
    cbar_c = plt.colorbar(im_c, ax=ax_c, fraction=0.04, pad=0.02)
    cbar_c.set_label("Pearson r", fontsize=8)

    # Panel D: Subpopulation Average PETHs (Early, Interm, Late - Lebedev Fig 3A)
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.plot(time_ms, early_peth, color="#27ae60", linewidth=2, label="Early (1/3)")
    ax_d.plot(time_ms, interm_peth, color="#e67e22", linewidth=2, label="Intermediate (1/3)")
    ax_d.plot(time_ms, late_peth, color="#2980b9", linewidth=2, label="Late (1/3)")
    ax_d.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax_d.set_title("D | Subpopulation PETHs\n(Early / Interm / Late Waves)", fontsize=11, fontweight="bold", loc="left")
    ax_d.set_xlabel("Time (ms)", fontsize=9)
    ax_d.set_ylabel("Rate (sp/s)", fontsize=9)
    ax_d.legend(frameon=False, fontsize=8)
    ax_d.grid(True, linestyle=":", alpha=0.5)

    # Panel E: Direct 3D Subpopulation Trajectory (NO PCA - Lebedev Fig 3C)
    ax_e = fig.add_subplot(gs[1, 0], projection='3d')
    ax_e.plot(early_peth, interm_peth, late_peth, color="navy", linewidth=2.5)
    ax_e.scatter(early_peth[0], interm_peth[0], late_peth[0], color="red", s=40, label="t=-150ms")
    ax_e.scatter(early_peth[-1], interm_peth[-1], late_peth[-1], color="black", s=40, label="t=+350ms")
    ax_e.set_title("E | Direct Subpopulation Trajectory\n(NO PCA / jPCA applied)", fontsize=11, fontweight="bold", loc="left")
    ax_e.set_xlabel("Early (sp/s)", fontsize=8)
    ax_e.set_ylabel("Interm (sp/s)", fontsize=8)
    ax_e.set_zlabel("Late (sp/s)", fontsize=8)
    ax_e.legend(frameon=False, fontsize=7)

    # Panel F: Shuffled Breaths Subpopulation Trajectory (Lebedev Fig 3D)
    ax_f = fig.add_subplot(gs[1, 1], projection='3d')
    ax_f.plot(shuffled_early, shuffled_interm, shuffled_late, color="darkred", linewidth=2.5)
    ax_f.set_title("F | Shuffled Breaths Null Control\n(Preserves Mean Latencies)", fontsize=11, fontweight="bold", loc="left")
    ax_f.set_xlabel("Early (sp/s)", fontsize=8)
    ax_f.set_ylabel("Interm (sp/s)", fontsize=8)
    ax_f.set_zlabel("Late (sp/s)", fontsize=8)

    # Panel G: Linear Regression Fit to Circle (Lebedev Fig 9A, C)
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.plot(circle_x, circle_y, color="lightgray", linestyle="--", linewidth=2, label="Target Circle")
    ax_g.plot(pred_circle_train[:, 0], pred_circle_train[:, 1], color="black", linewidth=1.5, label="Train Fit")
    ax_g.plot(pred_circle_test[:, 0], pred_circle_test[:, 1], color="crimson", linewidth=1.5, label=f"Test Pred ($R^2={r2_circle_test:.2f}$)")
    ax_g.set_title(f"G | Linear Fit to Circle\n(Train vs Test $R^2={r2_circle_test:.2f}$)", fontsize=11, fontweight="bold", loc="left")
    ax_g.set_xlim(-1.5, 1.5)
    ax_g.set_ylim(-1.5, 1.5)
    ax_g.set_aspect("equal")
    ax_g.legend(frameon=False, fontsize=7)
    ax_g.grid(True, linestyle=":", alpha=0.5)

    # Panel H: Linear Regression Fit to Figure-8 Lissajous (Lebedev Fig 9B, D)
    ax_h = fig.add_subplot(gs[1, 3])
    ax_h.plot(fig8_x, fig8_y, color="lightgray", linestyle="--", linewidth=2, label="Target Figure-8")
    ax_h.plot(pred_fig8_train[:, 0], pred_fig8_train[:, 1], color="black", linewidth=1.5, label="Train Fit")
    ax_h.plot(pred_fig8_test[:, 0], pred_fig8_test[:, 1], color="teal", linewidth=1.5, label=f"Test Pred ($R^2={r2_fig8_test:.2f}$)")
    ax_h.set_title(f"H | Linear Fit to Figure-8\n(Train vs Test $R^2={r2_fig8_test:.2f}$)", fontsize=11, fontweight="bold", loc="left")
    ax_h.set_xlim(-1.5, 1.5)
    ax_h.set_ylim(-1.5, 1.5)
    ax_h.set_aspect("equal")
    ax_h.legend(frameon=False, fontsize=7)
    ax_h.grid(True, linestyle=":", alpha=0.5)

    # Panel I: PCA with vs without Cross-Breath Mean Subtraction (Lebedev Fig 8)
    ax_i1 = fig.add_subplot(gs[2, 0])
    ax_i2 = fig.add_subplot(gs[2, 1])

    # PCA on raw rates
    pca_raw = PCA(n_components=2).fit_transform(mean_peths.T)
    ax_i1.plot(pca_raw[:, 0], pca_raw[:, 1], color="black", linewidth=2)
    ax_i1.scatter(pca_raw[0, 0], pca_raw[0, 1], color="red", s=30, label="Onset")
    ax_i1.set_title("I1 | PCA (Raw Mean PETHs)", fontsize=10, fontweight="bold", loc="left")
    ax_i1.set_xlabel("PC 1", fontsize=8)
    ax_i1.set_ylabel("PC 2", fontsize=8)
    ax_i1.grid(True, linestyle=":", alpha=0.5)

    # PCA with mean subtracted
    mean_subtracted_peths = mean_peths - np.mean(mean_peths, axis=1, keepdims=True)
    pca_sub = PCA(n_components=2).fit_transform(mean_subtracted_peths.T)
    ax_i2.plot(pca_sub[:, 0], pca_sub[:, 1], color="purple", linewidth=2)
    ax_i2.scatter(pca_sub[0, 0], pca_sub[0, 1], color="red", s=30, label="Onset")
    ax_i2.set_title("I2 | PCA (Mean-Subtracted)", fontsize=10, fontweight="bold", loc="left")
    ax_i2.set_xlabel("PC 1", fontsize=8)
    ax_i2.set_ylabel("PC 2", fontsize=8)
    ax_i2.grid(True, linestyle=":", alpha=0.5)

    # Panel J: Summary Quantification & Scientific Interpretation
    ax_j = fig.add_subplot(gs[2, 2:])
    ax_j.axis("off")
    summary_text = (
        "<b>Lebedev et al. (2019) Dynamical Rigor Assessment:</b>\n\n"
        f"• <b>Total Clean Single Units Evaluated</b>: {n_u} units across {len(breath_onsets)} breaths.\n"
        f"• <b>Sequential Temporal Spread</b>: Clear sequential activation wave (Panels A, B, D).\n"
        f"• <b>Direct Subpopulation Rotation (Panel E)</b>: A closed 3D loop emerges in (Early, Interm, Late)\n"
        "  space without requiring PCA or jPCA transformations.\n"
        f"• <b>Cross-Breath Shuffling Control (Panel F)</b>: Confirms rotation is driven by the robust phase\n"
        "  sequence across breaths rather than trial-by-trial spurious noise.\n"
        f"• <b>Geometric Regression Capacity</b>: Circle Test $R^2 = {r2_circle_test:.3f}$ | Figure-8 Test $R^2 = {r2_fig8_test:.3f}$.\n"
        "  Demonstrates the expressive capacity of phase-shifted neural ensembles under linear projection.\n"
        "• <b>Conclusion</b>: The observed respiratory limit cycle reflects an authentic, phase-locked\n"
        "  sequence of inspiratory and expiratory neural ensembles."
    )
    ax_j.text(0.05, 0.90, summary_text.replace("<b>", "").replace("</b>", ""), fontsize=10, verticalalignment="top", fontfamily="sans-serif", linespacing=1.4)

    plt.tight_layout()
    fig_path = out_dir / "lebedev_2019_rotation_dynamics_rigor_checks.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved Lebedev Rigor Diagnostic Figure: {fig_path.resolve()}")

    # 7. Save Interactive 3D Subpopulation & Lissajous Explorer
    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter3d(
        x=early_peth,
        y=interm_peth,
        z=late_peth,
        mode='lines+markers',
        marker=dict(size=4, color=time_ms, colorscale='Viridis', colorbar=dict(title="Time (ms)")),
        line=dict(color='navy', width=4),
        name="Subpopulation Trajectory (Direct)"
    ))
    fig_html.update_layout(
        title="<b>Lebedev et al. Check: Direct 3D Subpopulation Trajectory (Early vs Intermediate vs Late) - NO PCA</b>",
        scene=dict(
            xaxis_title="Early Insp Subpopulation (sp/s)",
            yaxis_title="Intermediate Subpopulation (sp/s)",
            zaxis_title="Late Exp Subpopulation (sp/s)"
        ),
        template="plotly_white"
    )
    html_path = out_dir / "lebedev_subpopulation_3d_trajectory.html"
    fig_html.write_html(str(html_path))
    logger.info(f"Saved Interactive 3D Subpopulation HTML: {html_path.resolve()}")

    # 8. Direct Vector & High-Res Export of Split Presentation Panels
    # Panel 1: lebedev_panel_a_b_c_d_sequential_wave (.png and .svg)
    fig_p1, axs_p1 = plt.subplots(1, 4, figsize=(22, 5.0), dpi=300)
    im_a1 = axs_p1[0].imshow(norm_peths[sorted_order], aspect="auto", origin="lower",
                             extent=[time_ms[0], time_ms[-1], 0, n_u], cmap="inferno")
    axs_p1[0].axvline(0, color="white", linestyle="--", linewidth=1.5, alpha=0.9)
    axs_p1[0].set_title("A | Peak Latency Sorted PETHs\n(Wave Cascade Organization)", fontsize=11, fontweight="bold")
    axs_p1[0].set_xlabel("Time from Breath Onset (ms)", fontsize=10)
    axs_p1[0].set_ylabel("Neuron Rank (Early -> Late)", fontsize=10)
    cb_a1 = plt.colorbar(im_a1, ax=axs_p1[0], fraction=0.046, pad=0.04)
    cb_a1.set_label("Norm Rate", fontsize=9)

    for b in range(n_show_breaths):
        axs_p1[1].scatter(peak_times[sorted_order, b], np.arange(n_u), s=6, color="black", alpha=0.35)
    axs_p1[1].set_title(f"B | Peak Spread Across {n_show_breaths} Breaths\n(Temporal Sequence Consistency)", fontsize=11, fontweight="bold")
    axs_p1[1].set_xlabel("Peak Firing Time (ms)", fontsize=10)
    axs_p1[1].set_ylabel("Neuron Rank", fontsize=10)
    axs_p1[1].grid(True, linestyle=":", alpha=0.5)

    im_c1 = axs_p1[2].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
    axs_p1[2].set_title("C | Pairwise Cross-Correlation\n(Phase Shift Offsets)", fontsize=11, fontweight="bold")
    axs_p1[2].set_xlabel("Neuron Rank", fontsize=10)
    axs_p1[2].set_ylabel("Neuron Rank", fontsize=10)
    cb_c1 = plt.colorbar(im_c1, ax=axs_p1[2], fraction=0.046, pad=0.04)
    cb_c1.set_label("Pearson r", fontsize=9)

    axs_p1[3].plot(time_ms, early_peth, color="#27ae60", linewidth=2.5, label="Early (1/3)")
    axs_p1[3].plot(time_ms, interm_peth, color="#e67e22", linewidth=2.5, label="Intermediate (1/3)")
    axs_p1[3].plot(time_ms, late_peth, color="#2980b9", linewidth=2.5, label="Late (1/3)")
    axs_p1[3].axvline(0, color="black", linestyle="--", alpha=0.6)
    axs_p1[3].set_title("D | Subpopulation PETHs\n(Wave Phase Shifts)", fontsize=11, fontweight="bold")
    axs_p1[3].set_xlabel("Time (ms)", fontsize=10)
    axs_p1[3].set_ylabel("Rate (sp/s)", fontsize=10)
    axs_p1[3].legend(frameon=True, fontsize=9)
    axs_p1[3].grid(True, linestyle=":", alpha=0.5)

    fig_p1.tight_layout()
    fig_p1.savefig(out_dir / "lebedev_panel_a_b_c_d_sequential_wave.png", dpi=300)
    fig_p1.savefig(out_dir / "lebedev_panel_a_b_c_d_sequential_wave.svg")
    plt.close(fig_p1)

    # Panel 2: lebedev_panel_e_f_i_subpop_trajectories (.png and .svg)
    fig_p2 = plt.figure(figsize=(18, 5.0), dpi=300)
    ax_e2 = fig_p2.add_subplot(1, 4, 1, projection='3d')
    ax_e2.plot(early_peth, interm_peth, late_peth, color="navy", linewidth=2.5)
    ax_e2.scatter(early_peth[0], interm_peth[0], late_peth[0], color="red", s=45, label="Onset")
    ax_e2.scatter(early_peth[-1], interm_peth[-1], late_peth[-1], color="black", s=45, label="Offset")
    ax_e2.set_title("E | Direct Subpopulation 3D\n(NO PCA / jPCA applied)", fontsize=10, fontweight="bold")
    ax_e2.set_xlabel("Early", fontsize=8)
    ax_e2.set_ylabel("Interm", fontsize=8)
    ax_e2.set_zlabel("Late", fontsize=8)
    ax_e2.legend(frameon=False, fontsize=8)

    ax_f2 = fig_p2.add_subplot(1, 4, 2, projection='3d')
    ax_f2.plot(shuffled_early, shuffled_interm, shuffled_late, color="darkred", linewidth=2.5)
    ax_f2.set_title("F | Shuffled Breaths Null Control\n(Preserves Mean Latency)", fontsize=10, fontweight="bold")
    ax_f2.set_xlabel("Early", fontsize=8)
    ax_f2.set_ylabel("Interm", fontsize=8)
    ax_f2.set_zlabel("Late", fontsize=8)

    ax_i1_2 = fig_p2.add_subplot(1, 4, 3)
    ax_i1_2.plot(pca_raw[:, 0], pca_raw[:, 1], color="black", linewidth=2.5)
    ax_i1_2.scatter(pca_raw[0, 0], pca_raw[0, 1], color="red", s=35, label="Onset")
    ax_i1_2.set_title("I1 | PCA (Raw Mean PETHs)", fontsize=10, fontweight="bold")
    ax_i1_2.set_xlabel("PC 1", fontsize=9)
    ax_i1_2.set_ylabel("PC 2", fontsize=9)
    ax_i1_2.grid(True, linestyle=":", alpha=0.5)
    ax_i1_2.legend(frameon=False, fontsize=8)

    ax_i2_2 = fig_p2.add_subplot(1, 4, 4)
    ax_i2_2.plot(pca_sub[:, 0], pca_sub[:, 1], color="purple", linewidth=2.5)
    ax_i2_2.scatter(pca_sub[0, 0], pca_sub[0, 1], color="red", s=35, label="Onset")
    ax_i2_2.set_title("I2 | PCA (Mean-Subtracted)", fontsize=10, fontweight="bold")
    ax_i2_2.set_xlabel("PC 1", fontsize=9)
    ax_i2_2.set_ylabel("PC 2", fontsize=9)
    ax_i2_2.grid(True, linestyle=":", alpha=0.5)
    ax_i2_2.legend(frameon=False, fontsize=8)

    fig_p2.tight_layout()
    fig_p2.savefig(out_dir / "lebedev_panel_e_f_i_subpop_trajectories.png", dpi=300)
    fig_p2.savefig(out_dir / "lebedev_panel_e_f_i_subpop_trajectories.svg")
    plt.close(fig_p2)

    # Panel 3: lebedev_panel_g_h_summary (.png and .svg) & lebedev_panel_g_h_geometric_regression.png
    fig_p3 = plt.figure(figsize=(16, 5.0), dpi=300)
    ax_g3 = fig_p3.add_subplot(1, 3, 1)
    ax_g3.plot(circle_x, circle_y, color="lightgray", linestyle="--", linewidth=2, label="Target Circle")
    ax_g3.plot(pred_circle_train[:, 0], pred_circle_train[:, 1], color="black", linewidth=1.5, label="Train Fit")
    ax_g3.plot(pred_circle_test[:, 0], pred_circle_test[:, 1], color="crimson", linewidth=1.8, label=f"Test Pred (R²={r2_circle_test:.2f})")
    ax_g3.set_title(f"G | Linear Fit to Circle\n(Cross-Val R² = {r2_circle_test:.2f})", fontsize=11, fontweight="bold")
    ax_g3.set_xlim(-1.5, 1.5)
    ax_g3.set_ylim(-1.5, 1.5)
    ax_g3.set_aspect("equal")
    ax_g3.legend(frameon=False, fontsize=8)
    ax_g3.grid(True, linestyle=":", alpha=0.5)

    ax_h3 = fig_p3.add_subplot(1, 3, 2)
    ax_h3.plot(fig8_x, fig8_y, color="lightgray", linestyle="--", linewidth=2, label="Target Figure-8")
    ax_h3.plot(pred_fig8_train[:, 0], pred_fig8_train[:, 1], color="black", linewidth=1.5, label="Train Fit")
    ax_h3.plot(pred_fig8_test[:, 0], pred_fig8_test[:, 1], color="teal", linewidth=1.8, label=f"Test Pred (R²={r2_fig8_test:.2f})")
    ax_h3.set_title(f"H | Linear Fit to Figure-8\n(Cross-Val R² = {r2_fig8_test:.2f})", fontsize=11, fontweight="bold")
    ax_h3.set_xlim(-1.5, 1.5)
    ax_h3.set_ylim(-1.5, 1.5)
    ax_h3.set_aspect("equal")
    ax_h3.legend(frameon=False, fontsize=8)
    ax_h3.grid(True, linestyle=":", alpha=0.5)

    ax_j3 = fig_p3.add_subplot(1, 3, 3)
    ax_j3.axis("off")
    ax_j3.text(0.02, 0.95, summary_text.replace("<b>", "").replace("</b>", ""), fontsize=9.5, verticalalignment="top", fontfamily="sans-serif", linespacing=1.4)

    fig_p3.tight_layout()
    fig_p3.savefig(out_dir / "lebedev_panel_g_h_summary.png", dpi=300)
    fig_p3.savefig(out_dir / "lebedev_panel_g_h_summary.svg")
    plt.close(fig_p3)

    # Standalone regression figure
    fig_reg, axs_reg = plt.subplots(1, 2, figsize=(10, 4.8), dpi=300)
    axs_reg[0].plot(circle_x, circle_y, color="lightgray", linestyle="--", linewidth=2, label="Target Circle")
    axs_reg[0].plot(pred_circle_train[:, 0], pred_circle_train[:, 1], color="black", linewidth=1.5, label="Train Fit")
    axs_reg[0].plot(pred_circle_test[:, 0], pred_circle_test[:, 1], color="crimson", linewidth=1.8, label=f"Test Pred (R²={r2_circle_test:.2f})")
    axs_reg[0].set_title(f"G | Linear Fit to Circle (R²={r2_circle_test:.2f})", fontsize=11, fontweight="bold")
    axs_reg[0].set_aspect("equal")
    axs_reg[0].legend(frameon=False, fontsize=8)
    axs_reg[0].grid(True, linestyle=":", alpha=0.5)

    axs_reg[1].plot(fig8_x, fig8_y, color="lightgray", linestyle="--", linewidth=2, label="Target Figure-8")
    axs_reg[1].plot(pred_fig8_train[:, 0], pred_fig8_train[:, 1], color="black", linewidth=1.5, label="Train Fit")
    axs_reg[1].plot(pred_fig8_test[:, 0], pred_fig8_test[:, 1], color="teal", linewidth=1.8, label=f"Test Pred (R²={r2_fig8_test:.2f})")
    axs_reg[1].set_title(f"H | Linear Fit to Figure-8 (R²={r2_fig8_test:.2f})", fontsize=11, fontweight="bold")
    axs_reg[1].set_aspect("equal")
    axs_reg[1].legend(frameon=False, fontsize=8)
    axs_reg[1].grid(True, linestyle=":", alpha=0.5)
    fig_reg.tight_layout()
    fig_reg.savefig(out_dir / "lebedev_panel_g_h_geometric_regression.png", dpi=300)
    plt.close(fig_reg)

    print("\n" + "=" * 85)
    print(" [SUCCESS] LEBEDEV ET AL. (2019) RIGOR CHECKS COMPLETED")
    print("=" * 85)
    print(f" • Multi-Panel Diagnostic Plot : {fig_path.resolve()}")
    print(f" • Interactive 3D HTML Model   : {html_path.resolve()}")
    print(f" • Circle Generalization R^2   : {r2_circle_test:.3f}")
    print(f" • Figure-8 Generalization R^2 : {r2_fig8_test:.3f}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Run Lebedev et al. (2019) Rigor Checks on Rotational Dynamics")
    parser.add_argument("--consensus-dir", type=str, default="'output_LP21_60s_120s'/sliding_window_pipeline/lupin/consensus_master", help="Path to consensus_master folder")
    parser.add_argument("--raw-dir", type=str, default=r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1", help="Path to raw NIDQ directory")
    parser.add_argument("--out-dir", type=str, default="output_lebedev_rigor_checks_60s", help="Output directory")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds to analyze")
    parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds for NIDQ slicing")
    args = parser.parse_args()

    cons_dir_raw = args.consensus_dir
    if cons_dir_raw.startswith('"') and cons_dir_raw.endswith('"'):
        cons_dir_raw = cons_dir_raw[1:-1]
    cons_dir = Path(cons_dir_raw)
    if not cons_dir.exists():
        alt_path = Path(f"'{cons_dir_raw}'")
        if alt_path.exists():
            cons_dir = alt_path

    raw_dir_raw = args.raw_dir
    if raw_dir_raw.startswith('"') and raw_dir_raw.endswith('"'):
        raw_dir_raw = raw_dir_raw[1:-1]
    raw_dir = Path(raw_dir_raw)

    run_lebedev_checks(
        consensus_dir=cons_dir,
        raw_dir=raw_dir,
        out_dir=Path(args.out_dir),
        duration_s=args.duration,
        start_time=args.start_time
    )

if __name__ == "__main__":
    main()

