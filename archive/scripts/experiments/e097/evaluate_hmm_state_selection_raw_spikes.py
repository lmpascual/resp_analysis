"""Comprehensive Evaluation of HMM Model Order (State Numbers) on Raw Spiking Activity
without PCA Dimensionality Reduction.

Contrasts:
1. Baseline Eupnea Window (win1_baseline, 1740s - 1800s, 60s, pure eupneic breathing).
2. Anoxic Challenge Window (win2_anox1, 1800s - 1836s, 36s, eupnea -> apnea -> gasping).
Across Kilosort 4 and Lupin.

Computes:
- Model log-likelihood ln(L)
- Number of free parameters p
- Akaike Information Criterion (AIC = 2p - 2ln L)
- Bayesian Information Criterion (BIC = p*ln(T) - 2ln L)
- Out-of-sample test log-likelihood (80% train / 20% test)
- State emission profiles lambda_k across all single units sorted by VRC depth
- Trial-by-trial state decoding without PCA
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from hmmlearn import hmm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(r"c:\Users\lpascu\resp_analysis")
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAW_HMM_ORDER")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def load_window_raw_spikes(window_name: str, sorter_key: str, t_start: float, dur: float, dt: float = 0.010, max_units: int = 150):
    """Loads spike trains and bins into raw integer counts (dt = 10 ms)."""
    base_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / window_name / sorter_key
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

    # Sort units along VRC: Caudal entry (~2850+ um) to Rostral tip (0 um)
    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    if len(df_qc) > max_units:
        # If very high unit count (e.g. win1 KS4 with 315 units), select top max_units most active
        df_qc["n_spk_total"] = [np.sum(spk_clusters == cid) for cid in df_qc["cluster_id"]]
        df_qc = df_qc.sort_values(by="n_spk_total", ascending=False).head(max_units)
        df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)

    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values
    n_units = len(cids)

    time_bins = np.arange(0, dur + dt, dt)
    n_bins = len(time_bins) - 1
    counts = np.zeros((n_bins, n_units), dtype=np.int32)

    for u_idx, cid in enumerate(cids):
        c_spk = spk_times[spk_clusters == cid]
        if np.min(c_spk) > 1000.0:
            c_spk = c_spk - t_start
        counts[:, u_idx] = np.histogram(c_spk, bins=time_bins)[0][:n_bins]

    logger.info(f"Loaded {window_name} | {sorter_key}: {n_units} units, {n_bins} 10ms bins, {np.sum(counts)} spikes.")
    return {
        "window_name": window_name,
        "sorter_key": sorter_key,
        "n_units": n_units,
        "cids": cids,
        "depths": depths,
        "counts": counts,
        "dt": dt,
        "dur": dur,
        "n_bins": n_bins,
        "time_bins": time_bins,
        "bin_centers": (time_bins[:-1] + time_bins[1:]) / 2.0
    }


def evaluate_model_order(data: Dict[str, Any], k_range=range(2, 9)):
    """Systematically fits PoissonHMM for each K and calculates AIC, BIC, and Out-of-Sample Test LL."""
    counts = data["counts"]
    T, N = counts.shape
    window = data["window_name"]
    sorter = data["sorter_key"]

    # 80/20 train/test split (contiguous blocks to respect temporal structure)
    split_idx = int(T * 0.80)
    train_counts = counts[:split_idx]
    test_counts = counts[split_idx:]

    records = []
    models = {}

    for k in k_range:
        t0 = time.time()
        # Poisson HMM parameters: initial state (k-1), transition matrix (k*(k-1)), emission rates (k*N)
        p = (k - 1) + k * (k - 1) + k * N

        # 1. Fit on full data
        model_full = hmm.PoissonHMM(n_components=k, n_iter=50, random_state=42, tol=1e-2)
        model_full.fit(counts)
        log_lik = model_full.score(counts)
        aic = 2 * p - 2 * log_lik
        bic = p * np.log(T) - 2 * log_lik

        # 2. Out-of-sample cross-validation (train on 80%, test on 20%)
        test_ll_per_bin = np.nan
        try:
            model_cv = hmm.PoissonHMM(n_components=k, n_iter=40, random_state=42, tol=1e-2)
            model_cv.fit(train_counts)
            test_score = model_cv.score(test_counts)
            test_ll_per_bin = test_score / len(test_counts)
        except Exception as e:
            logger.warning(f"CV fit for K={k} encountered warning: {e}")

        elapsed = time.time() - t0
        logger.info(f"[{window} | {sorter}] K={k} | Params={p:5d} | LogLik={log_lik:10.1f} | AIC={aic:10.1f} | BIC={bic:10.1f} | Test-LL/bin={test_ll_per_bin:7.3f} ({elapsed:.2f}s)")

        rec = {
            "window": window,
            "sorter": sorter,
            "n_units": N,
            "k": k,
            "p": p,
            "log_lik": float(log_lik),
            "aic": float(aic),
            "bic": float(bic),
            "test_ll_per_bin": float(test_ll_per_bin),
            "elapsed": float(elapsed)
        }
        records.append(rec)
        models[k] = model_full

    df = pd.DataFrame(records)
    # Determine best K by BIC (minimum) and AIC (minimum)
    best_k_bic = int(df.loc[df["bic"].idxmin()]["k"])
    best_k_aic = int(df.loc[df["aic"].idxmin()]["k"])
    best_k_test = int(df.loc[df["test_ll_per_bin"].idxmax()]["k"]) if not df["test_ll_per_bin"].isna().all() else best_k_bic

    logger.info(f"[{window} | {sorter}] Optimal States: Best BIC = K={best_k_bic} | Best AIC = K={best_k_aic} | Best Test-LL = K={best_k_test}")

    return {
        "df": df,
        "models": models,
        "best_k_bic": best_k_bic,
        "best_k_aic": best_k_aic,
        "best_k_test": best_k_test
    }


def generate_model_order_evaluation_figures(eval_win1_ks4, eval_win1_lup, eval_win2_ks4, eval_win2_lup, out_dir: Path):
    """Generates publication-quality figure contrasting Baseline Eupnea vs Anoxic Transition state order curves."""
    fig = plt.figure(figsize=(18, 12), facecolor='#090d16')
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.22, left=0.07, right=0.95, top=0.92, bottom=0.08)

    datasets = [
        ("A. Baseline Eupnea Window (win1_baseline, 60s) - Kilosort 4", eval_win1_ks4["df"], eval_win1_ks4["best_k_bic"], '#38bdf8', gs[0, 0]),
        ("B. Baseline Eupnea Window (win1_baseline, 60s) - Lupin", eval_win1_lup["df"], eval_win1_lup["best_k_bic"], '#f59e0b', gs[0, 1]),
        ("C. Anoxic Transition Window (win2_anox1, 36s) - Kilosort 4", eval_win2_ks4["df"], eval_win2_ks4["best_k_bic"], '#10b981', gs[1, 0]),
        ("D. Anoxic Transition Window (win2_anox1, 36s) - Lupin", eval_win2_lup["df"], eval_win2_lup["best_k_bic"], '#ec4899', gs[1, 1]),
    ]

    for title, df, best_k, col, pos in datasets:
        ax = fig.add_subplot(pos, facecolor='#0f172a')
        k_vals = df["k"].values
        
        # Normalize BIC and AIC relative to K=2 for clear visual contrast
        bic_norm = (df["bic"].values - np.min(df["bic"].values)) / 1000.0
        aic_norm = (df["aic"].values - np.min(df["aic"].values)) / 1000.0

        # Primary Axis: BIC & AIC (relative to minimum, in 1k units)
        l1 = ax.plot(k_vals, bic_norm, marker='o', lw=2.0, color='#38bdf8', label='BIC (Δ / 1,000, lower is better)')
        l2 = ax.plot(k_vals, aic_norm, marker='s', lw=1.8, ls='--', color='#a855f7', label='AIC (Δ / 1,000, lower is better)')
        ax.set_xlabel("Number of Hidden States (K)", color='#f8fafc', fontsize=11, fontweight='bold')
        ax.set_ylabel("Information Criterion Penalty (x10³)", color='#f8fafc', fontsize=10, fontweight='bold')
        ax.tick_params(colors='#94a3b8', labelsize=9.5)
        ax.set_xticks(k_vals)

        # Highlight optimal K
        ax.axvline(best_k, color='#fbbf24', ls=':', lw=2.0, label=f'Optimal BIC (K = {best_k})')

        # Secondary Axis: Out-of-Sample Test Log-Likelihood per bin
        ax2 = ax.twinx()
        l3 = ax2.plot(k_vals, df["test_ll_per_bin"].values, marker='^', lw=1.8, color='#10b981', label='Test Log-Likelihood / bin (higher is better)')
        ax2.set_ylabel("Out-of-Sample Test LL / bin", color='#10b981', fontsize=10, fontweight='bold')
        ax2.tick_params(colors='#10b981', labelsize=9.5)

        ax.set_title(title, color='#f8fafc', fontsize=12, fontweight='bold', pad=10)

        # Combine legends
        lines = l1 + l2 + [plt.Line2D([0], [0], color='#fbbf24', ls=':', lw=2.0)] + l3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8.0)

    fig.suptitle("HMM Model Order Selection: Systematic Evaluation on Raw Spiking Activity (Without PCA)",
                  color='#f8fafc', fontsize=15, fontweight='bold', y=0.97)

    fig_path = out_dir / "hmm_model_order_selection_baseline_vs_anoxia.png"
    fig.savefig(fig_path, dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved Model Order figure to {fig_path}")


def generate_raw_spiking_state_profiles_and_rasters(eval_win2_ks4, data_win2_ks4, eval_win2_lup, data_win2_lup, out_dir: Path):
    """Visualizes the decoded state emission rate profiles (lambda_k) across all single units and trial rasters without PCA."""
    fig = plt.figure(figsize=(19, 13), facecolor='#090d16')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.22,
                            left=0.06, right=0.96, top=0.93, bottom=0.07)

    # 1. KS4 State Emission Rates (lambda_k in Hz) across all 92 units sorted by depth
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#0f172a')
    best_k = eval_win2_ks4["best_k_bic"]
    best_model = eval_win2_ks4["models"][best_k]
    # Lambdas are per 10 ms bin; convert to Hz by multiplying by 100
    lambdas_hz = best_model.lambdas_ * (1.0 / data_win2_ks4["dt"])  # Shape: (K, N)
    
    # Plot heatmap of emission rates
    im1 = ax1.imshow(lambdas_hz, aspect='auto', cmap='plasma', interpolation='nearest')
    ax1.set_xlabel("Single Units (Sorted Caudal to Rostral Depth)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax1.set_ylabel(f"HMM States (K={best_k})\nOptimal BIC", color='#f8fafc', fontsize=10, fontweight='bold')
    ax1.set_yticks(range(best_k))
    ax1.set_yticklabels([f"State {i+1}" for i in range(best_k)], color='#cbd5e1', fontsize=9.5)
    ax1.set_title(f"Kilosort 4 (92 Units): Raw Poisson Emission Rates (λ_k in Hz) across States (K={best_k})",
                  color='#38bdf8', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label("Rate (Hz)", color='#f8fafc', fontsize=9)
    cbar1.ax.tick_params(colors='#94a3b8')

    # 2. Lupin State Emission Rates across 120 units
    ax2 = fig.add_subplot(gs[1, 0], facecolor='#0f172a')
    best_k_lup = eval_win2_lup["best_k_bic"]
    best_model_lup = eval_win2_lup["models"][best_k_lup]
    lambdas_hz_lup = best_model_lup.lambdas_ * (1.0 / data_win2_lup["dt"])
    
    im2 = ax2.imshow(lambdas_hz_lup, aspect='auto', cmap='plasma', interpolation='nearest')
    ax2.set_xlabel("Single Units (Sorted Caudal to Rostral Depth)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax2.set_ylabel(f"HMM States (K={best_k_lup})\nOptimal BIC", color='#f8fafc', fontsize=10, fontweight='bold')
    ax2.set_yticks(range(best_k_lup))
    ax2.set_yticklabels([f"State {i+1}" for i in range(best_k_lup)], color='#cbd5e1', fontsize=9.5)
    ax2.set_title(f"Lupin (120 Units): Raw Poisson Emission Rates (λ_k in Hz) across States (K={best_k_lup})",
                  color='#f59e0b', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label("Rate (Hz)", color='#f8fafc', fontsize=9)
    cbar2.ax.tick_params(colors='#94a3b8')

    # 3. Decoded States Timeline on Raw Spiking for win2_anox1
    ax3 = fig.add_subplot(gs[0, 1], facecolor='#0f172a')
    states_ks4 = best_model.predict(data_win2_ks4["counts"])
    t_axis = data_win2_ks4["bin_centers"]
    colors = ['#10b981', '#f97316', '#38bdf8', '#ef4444', '#a855f7', '#fbbf24', '#06b6d4', '#64748b']

    for s_i in range(best_k):
        mask_s = (states_ks4 == s_i)
        ax3.fill_between(t_axis, 0, 1, where=mask_s, color=colors[s_i % len(colors)], step='mid', alpha=0.9,
                         label=f"State {s_i+1}")
    ax3.set_xlim(0, 36.0)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel("Time in Anoxic Window (s)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax3.set_ylabel("Decoded State", color='#f8fafc', fontsize=10, fontweight='bold')
    ax3.set_title(f"KS4 (92 Units): Decoded HMM States (K={best_k}) on Raw Spiking", color='#38bdf8', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8, ncol=best_k)

    # 4. Decoded States Timeline for Lupin on Raw Spiking
    ax4 = fig.add_subplot(gs[1, 1], facecolor='#0f172a')
    states_lup = best_model_lup.predict(data_win2_lup["counts"])
    for s_i in range(best_k_lup):
        mask_s = (states_lup == s_i)
        ax4.fill_between(t_axis, 0, 1, where=mask_s, color=colors[s_i % len(colors)], step='mid', alpha=0.9,
                         label=f"State {s_i+1}")
    ax4.set_xlim(0, 36.0)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_xlabel("Time in Anoxic Window (s)", color='#f8fafc', fontsize=10, fontweight='bold')
    ax4.set_ylabel("Decoded State", color='#f8fafc', fontsize=10, fontweight='bold')
    ax4.set_title(f"Lupin (120 Units): Decoded HMM States (K={best_k_lup}) on Raw Spiking", color='#f59e0b', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8, ncol=best_k_lup)

    fig.suptitle("Raw Population Spiking Emission Rates & Decoded States (No PCA Dimensionality Reduction)",
                  color='#f8fafc', fontsize=14, fontweight='bold', y=0.97)

    fig_path = out_dir / "raw_spiking_state_profiles_and_decoding.png"
    fig.savefig(fig_path, dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved Raw Spiking Profiles figure to {fig_path}")


def main():
    t0_main = time.time()
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / "win2_anox1" / "population_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading Raw Spiking Data for Baseline Eupnea and Anoxic Transition...")
    # Baseline window (1740s - 1800s, 60s)
    data_win1_ks4 = load_window_raw_spikes("win1_baseline", "kilosort4", 1740.0, 60.0, max_units=120)
    data_win1_lup = load_window_raw_spikes("win1_baseline", "lupin", 1740.0, 60.0, max_units=120)

    # Anoxic Transition window (1800s - 1836s, 36s)
    data_win2_ks4 = load_window_raw_spikes("win2_anox1", "kilosort4", 1800.0, 36.0, max_units=150)
    data_win2_lup = load_window_raw_spikes("win2_anox1", "lupin", 1800.0, 36.0, max_units=150)

    # 2. Evaluate Model Orders (K in 2..8)
    logger.info("\n>>> Evaluating Model Orders on Baseline Eupnea Window (win1_baseline)...")
    eval_win1_ks4 = evaluate_model_order(data_win1_ks4, k_range=range(2, 8))
    eval_win1_lup = evaluate_model_order(data_win1_lup, k_range=range(2, 8))

    logger.info("\n>>> Evaluating Model Orders on Anoxic Transition Window (win2_anox1)...")
    eval_win2_ks4 = evaluate_model_order(data_win2_ks4, k_range=range(2, 8))
    eval_win2_lup = evaluate_model_order(data_win2_lup, k_range=range(2, 8))

    # 3. Generate Comparative Figures
    logger.info("\n>>> Generating Figures...")
    generate_model_order_evaluation_figures(eval_win1_ks4, eval_win1_lup, eval_win2_ks4, eval_win2_lup, out_dir)
    generate_raw_spiking_state_profiles_and_rasters(eval_win2_ks4, data_win2_ks4, eval_win2_lup, data_win2_lup, out_dir)

    # 4. Save CSV Table
    df_summary = pd.concat([
        eval_win1_ks4["df"], eval_win1_lup["df"],
        eval_win2_ks4["df"], eval_win2_lup["df"]
    ], ignore_index=True)
    summary_csv = out_dir / "hmm_model_order_selection_results.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved evaluation metrics table to {summary_csv}")

    logger.info(f"Suite completed in {time.time() - t0_main:.2f}s.")


if __name__ == "__main__":
    main()
