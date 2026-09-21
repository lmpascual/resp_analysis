#!/usr/bin/env python3
"""
Comprehensive Per-Neuron Event Modulation & Alignment Analysis for e097.
Evaluates whether each of the 96 single units is best aligned to:
1. Diaphragm EMG Inspiration Onset (I_on)
2. Diaphragm EMG Inspiration Offset (I_off_EMG)
3. Flow Meter Expiration Onset (E_on_flow / flow trough)

Computes:
- Event-locked temporal jitter (sigma_on, sigma_off_emg, sigma_off_flow)
- Breath duration (T_I) linear regression slopes (beta_on, beta_off)
- PETH peak sharpness and FWHM kurtosis ratios
- Continuous phase circular vector strength gain (Delta R = R_flow - R_emg)
- Formal 4-class classification: I_on-Locked, E_on(Flow)-Locked, I_off(EMG)-Locked, Duration-Scaler

Outputs:
- e097_prelaser_respiratory_neurodynamics/06_exploratory_flow_expiration/neuron_modulation_classification_scorecard.csv
- e097_prelaser_respiratory_neurodynamics/06_exploratory_flow_expiration/fig_event_modulation_classification_summary.png / .svg
- e097_prelaser_respiratory_neurodynamics/06_exploratory_flow_expiration/neuron_modulation_interactive_dashboard.html
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "e097_prelaser_respiratory_neurodynamics" / "06_exploratory_flow_expiration"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("STARTING PER-NEURON EVENT MODULATION & ALIGNMENT ANALYSIS (e097)")
    print("=" * 80)

    # 1. Load Data
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_60s" / "lupin" / "consensus_master"
    alf_dir = cons_dir / "alf"
    spk_times_all = np.load(alf_dir / "spike_times.npy")
    spk_clusters_all = np.load(alf_dir / "spike_clusters.npy")
    df_info = pd.read_csv(cons_dir / "cluster_info.tsv", sep="\t")

    qc_col = "passed_qc" if "passed_qc" in df_info.columns else "passed_allen_qc"
    df_sua = df_info[df_info[qc_col] == True].copy().sort_values("depth_um", ascending=True).reset_index(drop=True)

    # Load Phase classifications
    phase_csv = PROJECT_ROOT / "e097_prelaser_respiratory_neurodynamics" / "unit_respiratory_phase_classification.csv"
    df_phase = pd.read_csv(phase_csv)
    df_sua = df_sua.merge(df_phase[["cluster_id", "vector_strength", "unit_class", "mean_phase"]], on="cluster_id", how="left")

    # Load Breath Metrics
    metrics_csv = OUT_DIR / "breath_by_breath_flow_vs_emg_metrics.csv"
    df_metrics = pd.read_csv(metrics_csv)
    eupnea_mask = ~df_metrics["is_sigh"]
    df_eupnea = df_metrics[eupnea_mask].copy().sort_values("t_on_emg", ascending=True).reset_index(drop=True)
    n_breaths = len(df_eupnea)
    duration_s = 60.0

    print(f"Analyzing {len(df_sua)} clean units across {n_breaths} eupneic cycles.")

    # 2. Extract Event Arrays
    t_on_arr = df_eupnea["t_on_emg"].values
    t_off_emg_arr = df_eupnea["t_off_emg"].values
    t_flow_slope_arr = df_eupnea["t_flow_slope"].values
    ti_flow_arr = df_eupnea["T_I_flow_slope_ms"].values / 1000.0  # seconds
    ti_emg_arr = df_eupnea["T_I_emg_ms"].values / 1000.0        # seconds

    # Precompute Continuous Phase Arrays for Vector Strength
    # Sample at 2 kHz
    dt_phase = 0.0005
    t_phase_eval = np.arange(0, duration_s, dt_phase)

    def get_phase_vector(onsets, offsets, t_eval):
        phi = np.full_like(t_eval, np.nan, dtype=np.float64)
        for k in range(len(offsets) - 1):
            t_on = onsets[k]
            t_off = offsets[k]
            t_next = onsets[k + 1]
            if t_off <= t_on or t_next <= t_off:
                continue
            insp_mask = (t_eval >= t_on) & (t_eval < t_off)
            if np.any(insp_mask):
                phi[insp_mask] = -np.pi + (t_eval[insp_mask] - t_on) / (t_off - t_on) * np.pi
            exp_mask = (t_eval >= t_off) & (t_eval < t_next)
            if np.any(exp_mask):
                phi[exp_mask] = (t_eval[exp_mask] - t_off) / (t_next - t_off) * np.pi
        v_idx = np.where(~np.isnan(phi))[0]
        if len(v_idx) > 0:
            phi[:v_idx[0]] = phi[v_idx[0]]
            phi[v_idx[-1]:] = phi[v_idx[-1]]
        return phi

    phi_emg_arr = get_phase_vector(t_on_arr, t_off_emg_arr, t_phase_eval)
    phi_flow_arr = get_phase_vector(t_on_arr, t_flow_slope_arr, t_phase_eval)

    # 3. Analyze Each Neuron
    peth_dt = 0.005  # 5 ms bins
    bins_on = np.arange(-0.150, 0.305, peth_dt)
    bins_off_flow = np.arange(-0.250, 0.255, peth_dt)
    bins_off_emg = np.arange(-0.250, 0.255, peth_dt)

    centers_on = (bins_on[:-1] + bins_on[1:]) / 2.0
    centers_off_flow = (bins_off_flow[:-1] + bins_off_flow[1:]) / 2.0
    centers_off_emg = (bins_off_emg[:-1] + bins_off_emg[1:]) / 2.0

    kernel = signal.windows.gaussian(11, std=2)
    kernel /= np.sum(kernel)

    scorecard = []

    for idx, row in df_sua.iterrows():
        uid = int(row["cluster_id"])
        depth = float(row["depth_um"]) if not np.isnan(row["depth_um"]) else 0.0
        unit_class = str(row["unit_class"]) if pd.notna(row["unit_class"]) else "Unknown"

        spks = spk_times_all[spk_clusters_all == uid]
        n_spk = len(spks)
        overall_fr = float(n_spk / duration_s)

        if n_spk < 20:
            # Low firing rate unit
            scorecard.append({
                "cluster_id": uid, "depth_um": depth, "overall_fr": round(overall_fr, 2),
                "unit_class": unit_class, "best_alignment": "Low-FR / Indeterminate",
                "sigma_on_ms": np.nan, "sigma_off_flow_ms": np.nan, "sigma_off_emg_ms": np.nan,
                "beta_ti_slope": np.nan, "r2_ti": np.nan,
                "R_emg": np.nan, "R_flow": np.nan, "delta_R": np.nan,
                "peth_kurt_on": np.nan, "peth_kurt_flow": np.nan, "peth_kurt_emg": np.nan,
                "primary_driver": "Indeterminate"
            })
            continue

        # A. Compute Continuous Phase Vector Strength R under EMG vs Flow
        spk_idx_phase = np.clip((spks / dt_phase).astype(int), 0, len(t_phase_eval) - 1)
        spk_phi_emg = phi_emg_arr[spk_idx_phase]
        spk_phi_flow = phi_flow_arr[spk_idx_phase]

        valid_emg = spk_phi_emg[~np.isnan(spk_phi_emg)]
        valid_flow = spk_phi_flow[~np.isnan(spk_phi_flow)]

        R_emg = float(np.abs(np.mean(np.exp(1j * valid_emg)))) if len(valid_emg) > 0 else 0.0
        R_flow = float(np.abs(np.mean(np.exp(1j * valid_flow)))) if len(valid_flow) > 0 else 0.0
        delta_R = R_flow - R_emg

        # B. Compute Per-Breath Burst Latencies
        # For each breath, find the burst peak / median spike in the window
        lat_on_list = []
        lat_off_flow_list = []
        lat_off_emg_list = []
        ti_list = []

        rates_on_trials = []
        rates_flow_trials = []
        rates_emg_trials = []

        for b in range(n_breaths - 1):
            t_on = t_on_arr[b]
            t_off_flow = t_flow_slope_arr[b]
            t_off_emg = t_off_emg_arr[b]
            t_next = t_on_arr[b + 1]

            # Spikes in breath cycle
            spk_b = spks[(spks >= t_on - 0.05) & (spks < t_next)]
            if len(spk_b) >= 2:
                # Burst peak / median latency
                t_median_spk = np.median(spk_b)
                lat_on = (t_median_spk - t_on) * 1000.0  # ms
                lat_flow = (t_median_spk - t_off_flow) * 1000.0  # ms
                lat_emg = (t_median_spk - t_off_emg) * 1000.0  # ms

                lat_on_list.append(lat_on)
                lat_off_flow_list.append(lat_flow)
                lat_off_emg_list.append(lat_emg)
                ti_list.append(ti_flow_arr[b] * 1000.0)  # ms

            # Histograms for PETH
            c_on, _ = np.histogram(spks - t_on, bins=bins_on)
            rates_on_trials.append(np.convolve(c_on / peth_dt, kernel, mode="same"))

            c_flow, _ = np.histogram(spks - t_off_flow, bins=bins_off_flow)
            rates_flow_trials.append(np.convolve(c_flow / peth_dt, kernel, mode="same"))

            c_emg, _ = np.histogram(spks - t_off_emg, bins=bins_off_emg)
            rates_emg_trials.append(np.convolve(c_emg / peth_dt, kernel, mode="same"))

        # Jitters
        sigma_on = float(np.std(lat_on_list)) if len(lat_on_list) > 10 else np.nan
        sigma_flow = float(np.std(lat_off_flow_list)) if len(lat_off_flow_list) > 10 else np.nan
        sigma_emg = float(np.std(lat_off_emg_list)) if len(lat_off_emg_list) > 10 else np.nan

        # Regression slope against T_I
        if len(lat_on_list) > 15:
            res = stats.linregress(ti_list, lat_on_list)
            beta_ti = float(res.slope)
            r2_ti = float(res.rvalue ** 2)
        else:
            beta_ti = np.nan
            r2_ti = np.nan

        # PETH Kurtosis / Sharpness
        def compute_sharpness(mean_curve, centers_ms):
            pk = np.max(mean_curve)
            half_pk = pk / 2.0
            above = np.where(mean_curve >= half_pk)[0]
            if len(above) >= 2:
                fwhm = (centers_ms[above[-1]] - centers_ms[above[0]])
                return float(pk / (fwhm + 1e-5))
            return float(pk / 100.0)

        mean_peth_on = np.mean(rates_on_trials, axis=0) if len(rates_on_trials) > 0 else np.zeros(len(centers_on))
        mean_peth_flow = np.mean(rates_flow_trials, axis=0) if len(rates_flow_trials) > 0 else np.zeros(len(centers_off_flow))
        mean_peth_emg = np.mean(rates_emg_trials, axis=0) if len(rates_emg_trials) > 0 else np.zeros(len(centers_off_emg))

        kurt_on = compute_sharpness(mean_peth_on, centers_on * 1000.0)
        kurt_flow = compute_sharpness(mean_peth_flow, centers_off_flow * 1000.0)
        kurt_emg = compute_sharpness(mean_peth_emg, centers_off_emg * 1000.0)

        # C. Classification Decision Engine
        # Classify into:
        # 1. 'I_on-Locked': sigma_on is lowest AND beta_ti < 0.4
        # 2. 'E_on(Flow)-Locked': sigma_flow < sigma_on AND (sigma_flow <= sigma_emg or delta_R > 0.01) AND beta_ti > 0.45
        # 3. 'I_off(EMG)-Locked': sigma_emg < sigma_flow AND sigma_emg < sigma_on AND beta_ti > 0.45
        # 4. 'Duration-Scaler / Tonic': R < 0.15 or intermediate slope without tight jitter

        min_sig = min(sigma_on, sigma_flow, sigma_emg) if not np.isnan(sigma_on) else 999

        if R_flow < 0.15 and R_emg < 0.15:
            cls = "Tonic / Unmodulated"
        elif not np.isnan(beta_ti) and beta_ti < 0.35 and sigma_on <= min(sigma_flow, sigma_emg) * 1.15:
            cls = "I_on-Locked"
        elif not np.isnan(sigma_flow) and sigma_flow < sigma_on:
            if sigma_flow <= sigma_emg * 1.05 or delta_R > 0.005:
                cls = "E_on(Flow)-Locked"
            else:
                cls = "I_off(EMG)-Locked"
        elif not np.isnan(beta_ti) and beta_ti > 0.65:
            # Offset tracking
            if delta_R >= 0.0 or sigma_flow <= sigma_emg:
                cls = "E_on(Flow)-Locked"
            else:
                cls = "I_off(EMG)-Locked"
        elif not np.isnan(beta_ti) and 0.35 <= beta_ti <= 0.65:
            cls = "Duration-Scaler"
        else:
            cls = "I_on-Locked" if sigma_on <= min_sig else ("E_on(Flow)-Locked" if sigma_flow <= min_sig else "I_off(EMG)-Locked")

        scorecard.append({
            "cluster_id": uid,
            "depth_um": depth,
            "overall_fr": round(overall_fr, 2),
            "unit_class": unit_class,
            "best_alignment": cls,
            "sigma_on_ms": round(sigma_on, 1) if not np.isnan(sigma_on) else np.nan,
            "sigma_off_flow_ms": round(sigma_flow, 1) if not np.isnan(sigma_flow) else np.nan,
            "sigma_off_emg_ms": round(sigma_emg, 1) if not np.isnan(sigma_emg) else np.nan,
            "beta_ti_slope": round(beta_ti, 3) if not np.isnan(beta_ti) else np.nan,
            "r2_ti": round(r2_ti, 3) if not np.isnan(r2_ti) else np.nan,
            "R_emg": round(R_emg, 3),
            "R_flow": round(R_flow, 3),
            "delta_R": round(delta_R, 3),
            "peth_kurt_on": round(kurt_on, 2),
            "peth_kurt_flow": round(kurt_flow, 2),
            "peth_kurt_emg": round(kurt_emg, 2),
            "primary_driver": cls.split("-")[0]
        })

    df_scores = pd.DataFrame(scorecard)
    csv_out = OUT_DIR / "neuron_modulation_classification_scorecard.csv"
    df_scores.to_csv(csv_out, index=False)
    print(f"Saved modulation scorecard to {csv_out}")

    # Summary table
    counts = df_scores["best_alignment"].value_counts()
    print("\n--- POPULATION CLASSIFICATION SUMMARY (N = 96 UNITS) ---")
    for k, v in counts.items():
        pct = (v / len(df_scores)) * 100.0
        print(f"  * {k:25s}: {v:2d} neurons ({pct:5.1f}%)")

    # 4. Generate Publication-Quality Summary Figure
    print("Generating Figure: Event Modulation Classification Summary...")
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = gridspec.GridSpec(2, 3, hspace=0.32, wspace=0.28)

    palette = {
        "I_on-Locked": "#22c55e",
        "E_on(Flow)-Locked": "#a855f7",
        "I_off(EMG)-Locked": "#ef4444",
        "Duration-Scaler": "#f59e0b",
        "Tonic / Unmodulated": "#64748b",
        "Low-FR / Indeterminate": "#94a3b8"
    }

    # Panel A: Classification Pie / Donut
    ax1 = fig.add_subplot(gs[0, 0])
    valid_counts = counts[counts.index != "Low-FR / Indeterminate"]
    pie_colors = [palette.get(k, "#64748b") for k in valid_counts.index]
    wedges, texts, autotexts = ax1.pie(
        valid_counts.values,
        labels=[f"{k}\n(N={v})" for k, v in valid_counts.items()],
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor='#1e293b', linewidth=1.5),
        textprops=dict(fontsize=9, color='#1e293b', fontweight='bold')
    )
    ax1.set_title("A. Population Driver Breakdown (N=96)", fontsize=11, fontweight='bold', loc='left')

    # Panel B: Jitter Scatter: sigma_flow vs sigma_emg
    ax2 = fig.add_subplot(gs[0, 1])
    valid_df = df_scores.dropna(subset=["sigma_off_flow_ms", "sigma_off_emg_ms"])
    for cls_name, grp in valid_df.groupby("best_alignment"):
        c = palette.get(cls_name, "#64748b")
        ax2.scatter(grp["sigma_off_emg_ms"], grp["sigma_off_flow_ms"], color=c, label=cls_name, s=45, alpha=0.8, edgecolors='none')
    lims = [10, 140]
    ax2.plot(lims, lims, color='#94a3b8', linestyle='--', lw=1.2, label='Equal Jitter (y=x)')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(r'EMG Offset Jitter $\sigma_{off}^{EMG}$ (ms)', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel(r'Flow Expiration Jitter $\sigma_{off}^{flow}$ (ms)', fontsize=10.5, fontweight='bold')
    ax2.set_title(r"B. Temporal Jitter: Flow vs EMG Offset ($\sigma$)", fontsize=11, fontweight='bold', loc='left')
    ax2.legend(loc='upper left', fontsize=8.0)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Phase Vector Strength Gain (Delta R)
    ax3 = fig.add_subplot(gs[0, 2])
    dR_vals = df_scores["delta_R"].dropna()
    ax3.hist(dR_vals, bins=25, color='#a855f7', alpha=0.75, edgecolor='#581c87')
    ax3.axvline(0, color='#334155', linestyle='-', lw=1.2)
    ax3.axvline(dR_vals.median(), color='#9333ea', linestyle='--', lw=1.8, label=f'Median $\Delta R = {dR_vals.median():+.3f}$')
    ax3.set_xlabel(r'$\Delta R = R_{flow} - R_{EMG}$ (Vector Strength Gain)', fontsize=10.5, fontweight='bold')
    ax3.set_ylabel('Number of Neurons', fontsize=10.5, fontweight='bold')
    ax3.set_title(r"C. Phase Coherence Gain under Flow ($\Delta R$)", fontsize=11, fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=9.0)
    ax3.grid(True, linestyle=':', alpha=0.5)

    # Panel D: Anatomical Depth Distribution along VRC Probe
    ax4 = fig.add_subplot(gs[1, :2])
    for cls_name, grp in df_scores.groupby("best_alignment"):
        c = palette.get(cls_name, "#64748b")
        ax4.scatter(grp["depth_um"], grp["beta_ti_slope"], color=c, label=cls_name, s=60, alpha=0.85, edgecolors='#1e293b', lw=0.6)
    ax4.axhline(0.0, color='#22c55e', linestyle=':', lw=1.2, label=r'Ideal $I_{on}$-Locked ($\beta = 0$)')
    ax4.axhline(1.0, color='#a855f7', linestyle=':', lw=1.2, label=r'Ideal $E_{on}$-Locked ($\beta = 1$)')
    # Mark Anatomical zones
    ax4.axvspan(60, 750, color='#e0f2fe', alpha=0.35, label='BötC (60-750 µm)')
    ax4.axvspan(750, 1750, color='#fef3c7', alpha=0.35, label='preBötC Core (750-1750 µm)')
    ax4.axvspan(1750, 2800, color='#f3e8ff', alpha=0.35, label='rVRG / cVRG (1750-2800 µm)')
    ax4.set_xlabel(r'Probe Shank Depth ($\mu$m)', fontsize=11, fontweight='bold')
    ax4.set_ylabel(r'$T_I$ Regression Slope $\beta$ (0 = Onset, 1 = Offset)', fontsize=10.5, fontweight='bold')
    ax4.set_title("D. Anatomical Segregation of Onset vs Flow-Offset Drivers across VRC Column", fontsize=11, fontweight='bold', loc='left')
    ax4.legend(loc='lower right', ncol=2, fontsize=8.5)
    ax4.grid(True, linestyle=':', alpha=0.5)

    # Panel E: Key Exemplar Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    exemplar_ids = [255, 94, 18, 70, 163]
    ex_df = df_scores[df_scores["cluster_id"].isin(exemplar_ids)]
    y_pos = np.arange(len(ex_df))
    bars = ax5.barh(y_pos, ex_df["beta_ti_slope"], color=[palette.get(c, '#64748b') for c in ex_df["best_alignment"]], edgecolor='#1e293b')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f"Unit #{r['cluster_id']} ({r['best_alignment'][:10]}..)" for _, r in ex_df.iterrows()], fontsize=9, fontweight='bold')
    ax5.set_xlabel(r'Slope $\beta$ against $T_I$', fontsize=10.5, fontweight='bold')
    ax5.set_title("E. Exemplar Archetypes", fontsize=11, fontweight='bold', loc='left')
    ax5.axvline(0.5, color='#94a3b8', linestyle='--', lw=1.0)
    ax5.grid(True, linestyle=':', alpha=0.5)

    fig_png = OUT_DIR / "fig_event_modulation_classification_summary.png"
    fig_svg = OUT_DIR / "fig_event_modulation_classification_summary.svg"
    fig.savefig(fig_png, dpi=300, bbox_inches='tight')
    fig.savefig(fig_svg, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary figure to {fig_png}")

    # 5. Build Interactive HTML Dashboard
    print("Building Interactive Dashboard HTML...")
    build_modulation_dashboard_html(df_scores, OUT_DIR)

    print("=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Output files in: {OUT_DIR}")
    print("=" * 80)


def build_modulation_dashboard_html(df_scores, out_dir):
    records = df_scores.to_dict(orient="records")
    json_str = json.dumps(records)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Neuron Event-Modulation & Driver Classifier — e097</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #111827;
      --border: rgba(255, 255, 255, 0.08);
      --text: #f1f5f9;
      --muted: #94a3b8;
      --cyan: #38bdf8;
      --green: #22c55e;
      --purple: #a855f7;
      --red: #ef4444;
      --amber: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    header { max-width: 1550px; margin: 0 auto 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
    h1 { font-size: 20px; font-weight: 700; color: #fff; }
    .subtitle { font-size: 13px; color: var(--muted); margin-top: 4px; }

    .kpi-row { max-width: 1550px; margin: 0 auto 24px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .kpi-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; }
    .kpi-title { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
    .kpi-val { font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }

    .main-grid { max-width: 1550px; margin: 0 auto; display: grid; grid-template-columns: 1fr 420px; gap: 24px; }

    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
    .card-title { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }

    /* Filters */
    .filter-bar { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
    .f-btn { background: #1e293b; color: #fff; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; }
    .f-btn.active { background: #0284c7; border-color: #0284c7; }

    /* Table */
    .table-wrap { max-height: 600px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
    th { background: #060910; color: var(--muted); font-weight: 600; padding: 10px 12px; position: sticky; top: 0; cursor: pointer; border-bottom: 1px solid var(--border); }
    th:hover { color: #fff; }
    td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); font-family: 'JetBrains Mono', monospace; }
    tr:hover { background: rgba(255,255,255,0.03); cursor: pointer; }
    tr.selected { background: rgba(56, 189, 248, 0.15) !important; }

    .badge { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; font-family: 'Inter', sans-serif; display: inline-block; }
    .badge-ion { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-flow { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .badge-emg { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-scaler { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Inspector Sidebar */
    .inspector-content { display: flex; flex-direction: column; gap: 12px; font-size: 13px; }
    .inspect-row { display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255,255,255,0.06); padding-bottom: 6px; }
    .inspect-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #fff; }
  </style>
</head>
<body>

  <header>
    <div>
      <h1>Neuron Event-Modulation & Driver Classifier</h1>
      <div class="subtitle">Statistical dissociation of I_on (EMG), I_off (EMG), and E_on (Flow Trough) across all 96 VRC Single Units</div>
    </div>
  </header>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-title">I_on-Locked Drivers</div>
      <div class="kpi-val" style="color: var(--green);" id="kpiIon">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">E_on(Flow)-Locked Drivers</div>
      <div class="kpi-val" style="color: var(--purple);" id="kpiFlow">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">I_off(EMG)-Locked Drivers</div>
      <div class="kpi-val" style="color: var(--red);" id="kpiEmg">--</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-title">Duration-Scalers / Tonic</div>
      <div class="kpi-val" style="color: var(--amber);" id="kpiScaler">--</div>
    </div>
  </div>

  <div class="main-grid">
    <div class="card">
      <div class="card-title">
        <span>Population Scorecard (96 Single Units)</span>
        <span style="font-size: 12px; color: var(--muted);">Click any row to inspect</span>
      </div>

      <div class="filter-bar">
        <button class="f-btn active" id="fAll">All (96)</button>
        <button class="f-btn" id="fIon">I_on-Locked</button>
        <button class="f-btn" id="fFlow">E_on(Flow)-Locked</button>
        <button class="f-btn" id="fEmg">I_off(EMG)-Locked</button>
        <button class="f-btn" id="fScaler">Duration-Scaler</button>
      </div>

      <div class="table-wrap">
        <table id="scoresTable">
          <thead>
            <tr>
              <th onclick="sortTable('cluster_id')">Unit</th>
              <th onclick="sortTable('depth_um')">Depth (µm)</th>
              <th onclick="sortTable('unit_class')">Archetype</th>
              <th onclick="sortTable('best_alignment')">Primary Driver</th>
              <th onclick="sortTable('sigma_on_ms')">σ_on (ms)</th>
              <th onclick="sortTable('sigma_off_flow_ms')">σ_flow (ms)</th>
              <th onclick="sortTable('sigma_off_emg_ms')">σ_emg (ms)</th>
              <th onclick="sortTable('beta_ti_slope')">Slope β</th>
              <th onclick="sortTable('delta_R')">ΔR (Flow-EMG)</th>
            </tr>
          </thead>
          <tbody id="tableBody"></tbody>
        </table>
      </div>
    </div>

    <!-- Inspector Panel -->
    <div class="card">
      <div class="card-title">Unit Modulation Inspector</div>
      <div class="inspector-content" id="inspectContent">
        <div style="color: var(--muted); text-align: center; padding: 40px 0;">Select a unit from the table to view its full analytical scorecard.</div>
      </div>
    </div>
  </div>

  <script>
    const DATA = __JSON_DATA__;
    let currentFilter = 'all';
    let sortCol = 'cluster_id';
    let sortAsc = true;
    let selectedUnitId = 255;

    // KPI Counts
    const counts = {};
    DATA.forEach(d => { counts[d.best_alignment] = (counts[d.best_alignment] || 0) + 1; });
    document.getElementById('kpiIon').innerText = counts['I_on-Locked'] || 0;
    document.getElementById('kpiFlow').innerText = counts['E_on(Flow)-Locked'] || 0;
    document.getElementById('kpiEmg').innerText = counts['I_off(EMG)-Locked'] || 0;
    document.getElementById('kpiScaler').innerText = (counts['Duration-Scaler'] || 0) + (counts['Tonic / Unmodulated'] || 0);

    function getBadge(cls) {
      if (cls.includes('I_on')) return `<span class="badge badge-ion">${cls}</span>`;
      if (cls.includes('Flow')) return `<span class="badge badge-flow">${cls}</span>`;
      if (cls.includes('EMG')) return `<span class="badge badge-emg">${cls}</span>`;
      return `<span class="badge badge-scaler">${cls}</span>`;
    }

    function renderTable() {
      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = '';

      let filtered = DATA.filter(d => {
        if (currentFilter === 'all') return true;
        if (currentFilter === 'ion') return d.best_alignment === 'I_on-Locked';
        if (currentFilter === 'flow') return d.best_alignment === 'E_on(Flow)-Locked';
        if (currentFilter === 'emg') return d.best_alignment === 'I_off(EMG)-Locked';
        if (currentFilter === 'scaler') return d.best_alignment.includes('Scaler') || d.best_alignment.includes('Tonic');
        return true;
      });

      filtered.sort((a, b) => {
        let vA = a[sortCol], vB = b[sortCol];
        if (vA === null || isNaN(vA)) vA = sortAsc ? 9999 : -9999;
        if (vB === null || isNaN(vB)) vB = sortAsc ? 9999 : -9999;
        if (typeof vA === 'string') return sortAsc ? vA.localeCompare(vB) : vB.localeCompare(vA);
        return sortAsc ? vA - vB : vB - vA;
      });

      filtered.forEach(d => {
        const tr = document.createElement('tr');
        if (d.cluster_id === selectedUnitId) tr.className = 'selected';
        tr.innerHTML = `
          <td><strong>#${d.cluster_id}</strong></td>
          <td>${d.depth_um.toFixed(0)}</td>
          <td>${d.unit_class}</td>
          <td>${getBadge(d.best_alignment)}</td>
          <td>${d.sigma_on_ms !== null ? d.sigma_on_ms.toFixed(1) : '--'}</td>
          <td style="color:var(--purple);">${d.sigma_off_flow_ms !== null ? d.sigma_off_flow_ms.toFixed(1) : '--'}</td>
          <td style="color:var(--red);">${d.sigma_off_emg_ms !== null ? d.sigma_off_emg_ms.toFixed(1) : '--'}</td>
          <td>${d.beta_ti_slope !== null ? d.beta_ti_slope.toFixed(2) : '--'}</td>
          <td style="color:${d.delta_R > 0 ? '#4ade80' : '#f87171'};">${d.delta_R !== null ? (d.delta_R > 0 ? '+' : '') + d.delta_R.toFixed(3) : '--'}</td>
        `;
        tr.onclick = () => { selectUnit(d.cluster_id); };
        tbody.appendChild(tr);
      });
    }

    function selectUnit(uid) {
      selectedUnitId = uid;
      renderTable();
      const u = DATA.find(d => d.cluster_id === uid);
      if (!u) return;

      const p = document.getElementById('inspectContent');
      p.innerHTML = `
        <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">Unit #${u.cluster_id} (${u.depth_um.toFixed(0)} µm)</div>
        <div style="margin-bottom: 12px;">${getBadge(u.best_alignment)}</div>

        <div class="inspect-row"><span>Archetype:</span><span class="inspect-val">${u.unit_class}</span></div>
        <div class="inspect-row"><span>Overall Rate:</span><span class="inspect-val">${u.overall_fr} Hz</span></div>
        <div class="inspect-row"><span>I_on Latency Jitter (σ):</span><span class="inspect-val" style="color:var(--green);">${u.sigma_on_ms !== null ? u.sigma_on_ms + ' ms' : '--'}</span></div>
        <div class="inspect-row"><span>E_on(Flow) Jitter (σ):</span><span class="inspect-val" style="color:var(--purple);">${u.sigma_off_flow_ms !== null ? u.sigma_off_flow_ms + ' ms' : '--'}</span></div>
        <div class="inspect-row"><span>I_off(EMG) Jitter (σ):</span><span class="inspect-val" style="color:var(--red);">${u.sigma_off_emg_ms !== null ? u.sigma_off_emg_ms + ' ms' : '--'}</span></div>
        <div class="inspect-row"><span>T_I Regression Slope (β):</span><span class="inspect-val">${u.beta_ti_slope !== null ? u.beta_ti_slope : '--'}</span></div>
        <div class="inspect-row"><span>Regression Fit (R²):</span><span class="inspect-val">${u.r2_ti !== null ? u.r2_ti : '--'}</span></div>
        <div class="inspect-row"><span>Rayleigh Vector Strength (Flow):</span><span class="inspect-val">${u.R_flow !== null ? u.R_flow : '--'}</span></div>
        <div class="inspect-row"><span>Vector Strength Gain (ΔR):</span><span class="inspect-val" style="color:${u.delta_R > 0 ? '#4ade80' : '#f87171'};">${u.delta_R !== null ? (u.delta_R > 0 ? '+' : '') + u.delta_R : '--'}</span></div>
        <div class="inspect-row"><span>PETH Sharpness (Flow):</span><span class="inspect-val">${u.peth_kurt_flow !== null ? u.peth_kurt_flow : '--'}</span></div>
        <div class="inspect-row"><span>PETH Sharpness (EMG):</span><span class="inspect-val">${u.peth_kurt_emg !== null ? u.peth_kurt_emg : '--'}</span></div>

        <div style="margin-top: 16px;">
          <a href="all_96_neurons_flow_aligned_peth_interactive.html" target="_blank" style="display:block; text-align:center; background:#0284c7; color:#fff; padding:10px; border-radius:8px; text-decoration:none; font-weight:600; font-size:12px;">
            Open in 96-Neuron Raster Explorer ↗
          </a>
        </div>
      `;
    }

    function sortTable(col) {
      if (sortCol === col) { sortAsc = !sortAsc; }
      else { sortCol = col; sortAsc = true; }
      renderTable();
    }

    document.getElementById('fAll').onclick = (e) => { setF(e, 'all'); };
    document.getElementById('fIon').onclick = (e) => { setF(e, 'ion'); };
    document.getElementById('fFlow').onclick = (e) => { setF(e, 'flow'); };
    document.getElementById('fEmg').onclick = (e) => { setF(e, 'emg'); };
    document.getElementById('fScaler').onclick = (e) => { setF(e, 'scaler'); };

    function setF(e, f) {
      document.querySelectorAll('.f-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentFilter = f;
      renderTable();
    }

    renderTable();
    selectUnit(255);
  </script>
</body>
</html>
"""
    final_html = html_template.replace("__JSON_DATA__", json_str)
    html_path = out_dir / "neuron_modulation_interactive_dashboard.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Saved interactive modulation dashboard to {html_path}")


if __name__ == "__main__":
    main()
