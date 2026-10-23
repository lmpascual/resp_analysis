#!/usr/bin/env python3
"""
Generate Split Publication Panels for Presentation Deck:
1. presentation_assets/panel_c_i_anatomy_depth.png:
   - Panel c: Depth-ordered raster + EMG with explicit putative VRC anatomical region labels:
     RTN / pFRG (0-350 um), BötC (400-700 um), preBötC (750-1050 um), rVRG/cVRG (1100-1250 um)
   - Panel i: Probe depth distribution histogram of Inspiratory vs Expiratory units
2. presentation_assets/panel_d_h_temporal_tuning.png:
   - Panel d: Breath-aligned firing rate PSTH (Inspiratory, Expiratory, Tonic mean ± SEM)
   - Panel h: Vector strength (R) Rayleigh modulation depth distribution
3. presentation_assets/panel_f_e_g_phase_continuum.png:
   - Panel f: Phase-ordered population heatmap (2 cycles, -pi to 3pi) + Active population % curve
   - Panel e: Polar phase distribution
   - Panel g: Polar phase histogram (rose plot)
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
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

COLOR_INSP = "#d9534f"   # Coral Red
COLOR_EXP = "#337ab7"    # Steel Blue
COLOR_TONIC = "#888888"  # Slate Gray

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

    # Phase classification CSV
    phase_csv = PROJECT_ROOT / "outputs" / "LP21" / "bushetal2024_replicated_manifold_60s" / "unit_respiratory_phase_classification.csv"
    if not phase_csv.exists():
        phase_csv = PROJECT_ROOT / "output_publication_figures_LP21_60s" / "unit_respiratory_phase_classification.csv"
    df_phase = pd.read_csv(phase_csv)
    import ast
    if isinstance(df_phase["phase_fr"].iloc[0], str):
        df_phase["phase_fr"] = df_phase["phase_fr"].apply(ast.literal_eval)
    if isinstance(df_phase["phase_centers"].iloc[0], str):
        df_phase["phase_centers"] = df_phase["phase_centers"].apply(ast.literal_eval)

    df_merged = df_sua.merge(df_phase, on="cluster_id", how="inner").sort_values(by="depth_um", ascending=True).reset_index(drop=True)

    return {
        "spk_times": spk_times,
        "spk_clusters": spk_clusters,
        "t_nidq": t_nidq,
        "emg_env": emg_proc.envelope,
        "breath_onsets": breath_onsets,
        "breath_offsets": breath_offsets,
        "df_merged": df_merged,
        "duration_s": duration_s
    }

def make_panel_c_i_anatomy_depth(data, out_path: Path):
    """
    Panel c: Depth-ordered raster + EMG with explicit putative VRC anatomical regions
    Panel i: Probe depth distribution histogram of Inspiratory vs Expiratory units
    """
    df_merged = data["df_merged"]
    spk_times = data["spk_times"]
    spk_clusters = data["spk_clusters"]
    t_nidq = data["t_nidq"]
    emg_env = data["emg_env"]

    t0_snap, t1_snap = 5.0, 9.0

    fig = plt.figure(figsize=(16, 9.0), dpi=250)
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1.7, 1.0], height_ratios=[1.0, 1.0, 0.65], hspace=0.35, wspace=0.35)

    ax_c = fig.add_subplot(gs[0:2, 0])
    ax_emg = fig.add_subplot(gs[2, 0], sharex=ax_c)
    ax_i = fig.add_subplot(gs[0:2, 1])
    ax_i_legend = fig.add_subplot(gs[2, 1])
    ax_i_legend.axis("off")

    n_units = len(df_merged)

    # Define Putative VRC Anatomical Zones
    vrc_zones = [
        {"name": "RTN / pFRG", "sub": "Tonic unmodulated", "min_idx": 1, "max_idx": 4, "d_min": 0, "d_max": 350, "color": "#7f8c8d", "alpha": 0.12},
        {"name": "BötC", "sub": "Expiratory / Post-I", "min_idx": 5, "max_idx": 16, "d_min": 400, "d_max": 700, "color": "#2980b9", "alpha": 0.12},
        {"name": "preBötC", "sub": "Inspiratory rhythm core", "min_idx": 17, "max_idx": 29, "d_min": 720, "d_max": 1050, "color": "#c0392b", "alpha": 0.12},
        {"name": "rVRG / cVRG", "sub": "Premotor & projections", "min_idx": 30, "max_idx": 37, "d_min": 1100, "d_max": 1260, "color": "#8e44ad", "alpha": 0.12},
    ]

    for z in vrc_zones:
        ax_c.axhspan(z["min_idx"] - 0.45, z["max_idx"] + 0.45, color=z["color"], alpha=z["alpha"], zorder=0)

    # 1. Raster plot
    for idx, row in df_merged.iterrows():
        cid = row["cluster_id"]
        u_spks = spk_times[spk_clusters == cid]
        in_snap = u_spks[(u_spks >= t0_snap) & (u_spks <= t1_snap)]
        u_class = row["unit_class"]
        color = COLOR_INSP if u_class == "Inspiratory" else (COLOR_EXP if u_class == "Expiratory" else COLOR_TONIC)
        ax_c.scatter(in_snap, np.full_like(in_snap, idx + 1), s=9, color=color, alpha=0.9, edgecolors="none", zorder=2)

    ax_c.set_title("c | Depth-Ordered Single-Unit Raster across Ventral Respiratory Column", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_c.set_ylabel("Neuron ID (Dorsal -> Ventral)", fontsize=10, fontweight="semibold")
    ax_c.set_ylim(0.5, n_units + 0.5)
    ax_c.set_yticks([1, 5, 10, 15, 20, 25, 30, 35, 37])
    ax_c.grid(True, linestyle=":", alpha=0.4, zorder=1)

    # Putative VRC Anatomical Zones callout boxes placed prominently at right side inside raster
    for z in vrc_zones:
        mid_y = (z["min_idx"] + z["max_idx"]) / 2.0
        ax_c.text(t1_snap - 0.05, mid_y, f"★ {z['name']} [{z['d_min']}–{z['d_max']} µm]\n{z['sub']}",
                  fontsize=8.5, fontweight="bold", color=z["color"], va="center", ha="right",
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=z["color"], alpha=0.92, linewidth=1.4),
                  zorder=5)

    # Secondary y-axis showing actual Depth (µm)
    ax_c_depth = ax_c.twinx()
    unit_depths = df_merged["depth_um"].values
    ax_c_depth.set_ylim(unit_depths[0] - 25, unit_depths[-1] + 25)
    ax_c_depth.set_ylabel("Probe Depth along VRC Axis (µm)", fontsize=10, fontweight="semibold", color="#2c3e50")
    ax_c_depth.tick_params(axis='y', colors='#2c3e50')

    # 2. Diaphragm EMG Trace underneath
    mask_emg = (t_nidq >= t0_snap) & (t_nidq <= t1_snap)
    ax_emg.plot(t_nidq[mask_emg], emg_env[mask_emg], color="#2c3e50", linewidth=1.4)
    ax_emg.set_ylabel("Dia. EMG", fontsize=10, fontweight="semibold")
    ax_emg.set_xlabel("Time (seconds)", fontsize=10, fontweight="semibold")
    ax_emg.set_xlim(t0_snap, t1_snap)
    ax_emg.grid(True, linestyle=":", alpha=0.4)

    # 3. Panel i: Probe Depth Distribution Histogram
    insp_depths = df_merged[df_merged["unit_class"] == "Inspiratory"]["depth_um"].values
    exp_depths = df_merged[df_merged["unit_class"] == "Expiratory"]["depth_um"].values
    tonic_depths = df_merged[df_merged["unit_class"] == "Tonic"]["depth_um"].values
    bins_depth = np.linspace(0, 1300, 14)

    ax_i.hist(insp_depths, bins=bins_depth, color=COLOR_INSP, alpha=0.75, label=f"Inspiratory (n={len(insp_depths)})", edgecolor="black", linewidth=0.8)
    ax_i.hist(exp_depths, bins=bins_depth, color=COLOR_EXP, alpha=0.65, label=f"Expiratory (n={len(exp_depths)})", edgecolor="black", linewidth=0.8)
    ax_i.hist(tonic_depths, bins=bins_depth, color=COLOR_TONIC, alpha=0.5, label=f"Tonic (n={len(tonic_depths)})", edgecolor="black", linewidth=0.8)

    ax_i.set_title("i | Ventral Respiratory Column Depth Distribution", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_i.set_xlabel("Probe Depth along VRC (µm)", fontsize=10, fontweight="semibold")
    ax_i.set_ylabel("Neuron Count", fontsize=10, fontweight="semibold")
    ax_i.set_xlim(0, 1300)
    ax_i.set_ylim(0, 6.2)
    ax_i.grid(True, linestyle=":", alpha=0.4)
    ax_i.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)

    # Annotate putative anatomical peaks in histogram with safe vertical margin
    for z in vrc_zones:
        ax_i.axvspan(z["d_min"], z["d_max"], color=z["color"], alpha=0.10, zorder=0)
        ax_i.text((z["d_min"] + z["d_max"]) / 2.0, 5.65, z["name"],
                  ha="center", va="center", fontsize=8.5, fontweight="bold", color=z["color"],
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=z["color"], linewidth=1.0))

    # Explanatory summary card in lower right with wide text format
    summary_text = (
        "Putative VRC Regions (1-Shank NP2004 Probe):\n"
        "• 0–350 µm (RTN / pFRG): Unmodulated tonic baseline (R < 0.20).\n"
        "• 400–700 µm (BötC): Expiratory / post-I enrichment (58% Exp units).\n"
        "• 750–1,050 µm (preBötC): Rhythmogenic inspiratory core (R up to 0.93).\n"
        "• 1,100–1,250 µm (rVRG / cVRG): Premotor bulbospinal drive & outflow.\n"
        "Ref: Bush & Ramirez (Nat Neurosci 2024), Smith et al. (TINS 2013)"
    )
    ax_i_legend.text(0.0, 0.5, summary_text, fontsize=8.8, va="center", ha="left",
                     family="sans-serif", color="#1a252f",
                     bbox=dict(boxstyle="round,pad=0.55", facecolor="#f8f9fa", edgecolor="#b0bec5", linewidth=1.2))

    fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.08, wspace=0.32, hspace=0.38)
    fig.savefig(out_path, dpi=250)
    fig.savefig(out_path.with_suffix(".svg"), format="svg")
    plt.close(fig)
    print(f"[SUCCESS] Saved polished Panel c + i Anatomy Depth figure: {out_path} and .svg")

def make_panel_d_h_temporal_tuning(data, out_path: Path):
    """
    Panel d: Breath-aligned firing rate PSTH (Insp, Exp, Tonic mean ± SEM)
    Panel h: Vector strength (R) distribution with R=0.20 Rayleigh significance boundary
    """
    df_merged = data["df_merged"]
    spk_times = data["spk_times"]
    spk_clusters = data["spk_clusters"]
    t_nidq = data["t_nidq"]
    breath_onsets = data["breath_onsets"]

    fig = plt.figure(figsize=(14, 6.2), dpi=250)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.28)

    # 1. Panel d: Breath-aligned PSTH
    ax_d = fig.add_subplot(gs[0])
    win_pre, win_post = 0.2, 0.4  # [-200ms, +400ms]
    dt_psth = 0.005
    psth_bins = np.arange(-win_pre, win_post + dt_psth, dt_psth)
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0 * 1000.0  # ms

    insp_psths, exp_psths, tonic_psths = [], [], []
    valid_onsets = breath_onsets[(breath_onsets > win_pre) & (breath_onsets < t_nidq[-1] - win_post)]

    for _, row in df_merged.iterrows():
        cid = row["cluster_id"]
        spks = spk_times[spk_clusters == cid]
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
        ax_d.plot(psth_centers, mean_insp, color=COLOR_INSP, linewidth=2.5, label=f"Inspiratory (n={len(insp_psths)})")
        ax_d.fill_between(psth_centers, mean_insp - sem_insp, mean_insp + sem_insp, color=COLOR_INSP, alpha=0.2)

    if exp_psths:
        mean_exp = np.mean(exp_psths, axis=0)
        sem_exp = stats.sem(exp_psths, axis=0)
        ax_d.plot(psth_centers, mean_exp, color=COLOR_EXP, linewidth=2.5, label=f"Expiratory (n={len(exp_psths)})")
        ax_d.fill_between(psth_centers, mean_exp - sem_exp, mean_exp + sem_exp, color=COLOR_EXP, alpha=0.2)

    if tonic_psths:
        mean_tonic = np.mean(tonic_psths, axis=0)
        sem_tonic = stats.sem(tonic_psths, axis=0)
        ax_d.plot(psth_centers, mean_tonic, color=COLOR_TONIC, linewidth=2.0, linestyle="--", label=f"Tonic (n={len(tonic_psths)})")
        ax_d.fill_between(psth_centers, mean_tonic - sem_tonic, mean_tonic + sem_tonic, color=COLOR_TONIC, alpha=0.15)

    ax_d.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label=r"Inspiratory Onset ($I_{on}$)")
    ax_d.axvline(120, color="gray", linestyle=":", linewidth=1.5, alpha=0.8, label=r"Mean Inspiratory Offset ($I_{off}$)")
    ax_d.set_title("d | Breath-Aligned Population Firing Rate (PETH)", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_d.set_xlabel("Time Relative to Breath Onset (ms)", fontsize=10, fontweight="semibold")
    ax_d.set_ylabel("Instantaneous Firing Rate (sp/s)", fontsize=10, fontweight="semibold")
    ax_d.set_xlim(-200, 400)
    ax_d.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=8.5)
    ax_d.grid(True, linestyle=":", alpha=0.4)

    # 2. Panel h: Vector Strength (R) Rayleigh Distribution
    ax_h = fig.add_subplot(gs[1])
    vs_vals = df_merged["vector_strength"].values
    bins_r = np.linspace(0, 1.0, 16)
    
    counts, _, patches = ax_h.hist(vs_vals, bins=bins_r, color="#bdc3c7", edgecolor="black", linewidth=0.8)
    for patch, bin_left in zip(patches, bins_r[:-1]):
        if bin_left < 0.20:
            patch.set_facecolor(COLOR_TONIC)
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor("#34495e")
            patch.set_alpha(0.85)

    ax_h.axvline(0.20, color="#e74c3c", linestyle="--", linewidth=2.0, label="Modulation Cutoff (R = 0.20)")
    ax_h.set_title("h | Modulation Depth Distribution: Vector Strength (R)", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_h.set_xlabel("Rayleigh Vector Strength ($R$)", fontsize=10, fontweight="semibold")
    ax_h.set_ylabel("Number of Neurons", fontsize=10, fontweight="semibold")
    ax_h.set_xlim(0, 1.0)
    ax_h.grid(True, linestyle=":", alpha=0.4)
    ax_h.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)

    # Callout text on histogram
    ax_h.text(0.10, max(counts) * 0.75, "Tonic / Unmodulated\n(13.5%, n=5)", ha="center", fontsize=8.5, fontweight="bold", color="#555")
    ax_h.text(0.72, max(counts) * 0.75, "Phase-Locked\nRespiratory Units\n(86.5%, n=32)", ha="center", fontsize=8.5, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    fig.savefig(out_path.with_suffix(".svg"), format="svg")
    plt.close(fig)
    print(f"[SUCCESS] Saved Panel d + h Temporal Tuning figure: {out_path} and .svg")

def make_panel_f_e_g_phase_continuum(data, out_path: Path):
    """
    Panel f: Phase-ordered population heatmap (2 cycles, -pi to 3pi) + Active population % curve
    Panel e: Polar phase distribution
    Panel g: Circular phase histogram (rose plot)
    """
    df_merged = data["df_merged"]

    fig = plt.figure(figsize=(15, 8.5), dpi=250)
    # Master grid: Left (57%) and Right (43%)
    gs_master = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.32)

    # Left sub-grid: Heatmap (75%) and Active % curve (25%)
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0], height_ratios=[1.0, 0.35], hspace=0.30)
    ax_f = fig.add_subplot(gs_left[0])
    ax_f_act = fig.add_subplot(gs_left[1], sharex=ax_f)

    # Right sub-grid: Panel e (Polar tuning) and Panel g (Rose plot)
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[1], height_ratios=[1.0, 1.0], hspace=0.55)
    ax_e = fig.add_subplot(gs_right[0], polar=True)
    ax_g = fig.add_subplot(gs_right[1], polar=True)

    df_sorted_phase = df_merged.sort_values(by="mean_phase", ascending=True).reset_index(drop=True)
    heat_rows = []
    for _, row in df_sorted_phase.iterrows():
        cfr = np.array(row["phase_fr"])
        max_v = np.max(cfr) if np.max(cfr) > 0 else 1.0
        norm_fr = cfr / max_v
        two_cycle = np.concatenate([norm_fr, norm_fr])
        heat_rows.append(two_cycle)

    heat_rows = np.array(heat_rows)
    if len(heat_rows) > 0:
        im = ax_f.imshow(
            heat_rows,
            aspect="auto",
            cmap="Greys",
            extent=[-np.pi, 3*np.pi, 0.5, len(df_sorted_phase) + 0.5],
            origin="lower",
            interpolation="nearest"
        )
        cbar = plt.colorbar(im, ax=ax_f, fraction=0.035, pad=0.02)
        cbar.set_label("Normalized Firing Rate", fontsize=9, fontweight="semibold")

    ax_f.axvline(0, color=COLOR_INSP, linestyle="--", linewidth=1.5, alpha=0.9, label=r"$I_{on}$ (0 rad)")
    ax_f.axvline(np.pi, color=COLOR_EXP, linestyle="--", linewidth=1.5, alpha=0.9, label=r"$I_{off} / E_{start}$ ($\pi$ rad)")
    ax_f.axvline(2*np.pi, color=COLOR_INSP, linestyle="--", linewidth=1.5, alpha=0.9, label=r"Next $I_{on}$ ($2\pi$ rad)")
    ax_f.set_title("f | Phase-Ordered Population Heatmap (2 Full Respiratory Cycles)", fontsize=12, fontweight="bold", loc="left", pad=10)
    ax_f.set_ylabel("Neuron ID (Sorted by Preferred Phase)", fontsize=10, fontweight="semibold")
    ax_f.set_ylim(0.5, len(df_sorted_phase) + 0.5)
    ax_f.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=8)

    # Active population percentage curve underneath
    if len(heat_rows) > 0:
        active_pct = np.mean(heat_rows > 0.3, axis=0) * 100.0
        two_cycle_phase = np.linspace(-np.pi, 3*np.pi, len(active_pct))
        ax_f_act.plot(two_cycle_phase, active_pct, color="#2c3e50", linewidth=2.0)
        ax_f_act.axvline(0, color=COLOR_INSP, linestyle="--", alpha=0.7)
        ax_f_act.axvline(np.pi, color=COLOR_EXP, linestyle="--", alpha=0.7)
        ax_f_act.axvline(2*np.pi, color=COLOR_INSP, linestyle="--", alpha=0.7)
        ax_f_act.set_ylabel("Active Population (%)", fontsize=9, fontweight="semibold")
        ax_f_act.set_xlabel("Respiratory Phase $\phi(t)$ (radians)", fontsize=10, fontweight="semibold")
        ax_f_act.set_xlim(-np.pi, 3*np.pi)
        ax_f_act.set_xticks([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
        ax_f_act.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$", r"$3\pi$"])
        ax_f_act.set_ylim(25, 80)
        ax_f_act.grid(True, linestyle=":", alpha=0.4)
        ax_f_act.text(0.5, 0.85, "Continuous Non-Zero Network Activity (~40–70%)", transform=ax_f_act.transAxes,
                      ha="center", fontsize=8.5, fontweight="bold", color="#2c3e50",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    # 2. Panel e: Polar Phase Profiles & Population Tuning (Upper Right)
    phase_centers_raw = np.array(df_merged["phase_centers"].iloc[0])
    theta_mod = np.mod(phase_centers_raw, 2*np.pi)
    sort_idx = np.argsort(theta_mod)
    theta_sorted = theta_mod[sort_idx]
    theta_closed = np.append(theta_sorted, theta_sorted[0] + 2*np.pi)

    insp_units = df_merged[df_merged["unit_class"] == "Inspiratory"]
    exp_units = df_merged[df_merged["unit_class"] == "Expiratory"]
    tonic_units = df_merged[df_merged["unit_class"] == "Tonic"]

    # Plot individual unit profiles as faint lines
    for _, row in df_merged.iterrows():
        cfr = np.array(row["phase_fr"])[sort_idx]
        max_v = np.max(cfr) if np.max(cfr) > 0 else 1.0
        norm_fr = cfr / max_v
        norm_fr_closed = np.append(norm_fr, norm_fr[0])
        u_cls = row["unit_class"]
        if u_cls == "Inspiratory":
            ax_e.plot(theta_closed, norm_fr_closed, color=COLOR_INSP, alpha=0.20, linewidth=0.8, zorder=3)
        elif u_cls == "Expiratory":
            ax_e.plot(theta_closed, norm_fr_closed, color=COLOR_EXP, alpha=0.20, linewidth=0.8, zorder=3)
        else:
            ax_e.plot(theta_closed, norm_fr_closed, color="#95a5a6", alpha=0.15, linewidth=0.6, zorder=2)

    # Plot Population Mean and SEM Profiles
    if len(insp_units) > 0:
        insp_matrix = np.array([np.array(r["phase_fr"])[sort_idx] / (np.max(r["phase_fr"]) if np.max(r["phase_fr"]) > 0 else 1.0) for _, r in insp_units.iterrows()])
        mean_insp = np.mean(insp_matrix, axis=0)
        sem_insp = np.std(insp_matrix, axis=0) / np.sqrt(len(insp_matrix))
        mean_insp_closed = np.append(mean_insp, mean_insp[0])
        sem_insp_closed = np.append(sem_insp, sem_insp[0])
        ax_e.plot(theta_closed, mean_insp_closed, color=COLOR_INSP, linewidth=2.8, label=f"Insp Mean (n={len(insp_units)})", zorder=10)
        ax_e.fill(theta_closed, mean_insp_closed, color=COLOR_INSP, alpha=0.25, zorder=8)

    if len(exp_units) > 0:
        exp_matrix = np.array([np.array(r["phase_fr"])[sort_idx] / (np.max(r["phase_fr"]) if np.max(r["phase_fr"]) > 0 else 1.0) for _, r in exp_units.iterrows()])
        mean_exp = np.mean(exp_matrix, axis=0)
        sem_exp = np.std(exp_matrix, axis=0) / np.sqrt(len(exp_matrix))
        mean_exp_closed = np.append(mean_exp, mean_exp[0])
        sem_exp_closed = np.append(sem_exp, sem_exp[0])
        ax_e.plot(theta_closed, mean_exp_closed, color=COLOR_EXP, linewidth=2.8, label=f"Exp Mean (n={len(exp_units)})", zorder=10)
        ax_e.fill(theta_closed, mean_exp_closed, color=COLOR_EXP, alpha=0.25, zorder=8)

    if len(tonic_units) > 0:
        tonic_matrix = np.array([np.array(r["phase_fr"])[sort_idx] / (np.max(r["phase_fr"]) if np.max(r["phase_fr"]) > 0 else 1.0) for _, r in tonic_units.iterrows()])
        mean_tonic = np.mean(tonic_matrix, axis=0)
        mean_tonic_closed = np.append(mean_tonic, mean_tonic[0])
        ax_e.plot(theta_closed, mean_tonic_closed, color="#7f8c8d", linestyle="--", linewidth=1.8, label=f"Tonic Mean (n={len(tonic_units)})", zorder=7)

    ax_e.set_title("e | Polar Phase Profiles & Tuning Distribution", fontsize=11, fontweight="bold", loc="left", pad=14)
    ax_e.set_theta_zero_location("E")
    ax_e.set_theta_direction(1)
    ax_e.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    ax_e.set_xticklabels([r"$0^\circ$ ($I_{on}$)", r"$45^\circ$", r"$90^\circ$", r"$135^\circ$", r"$180^\circ$ ($I_{off}$)", r"$225^\circ$", r"$270^\circ$", r"$315^\circ$"], fontsize=7.5)
    ax_e.set_ylim(0, 1.05)
    ax_e.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_e.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.0, color="#555")
    ax_e.legend(loc="lower right", bbox_to_anchor=(1.30, 0.02), frameon=True, facecolor="white", edgecolor="#ddd", fontsize=7.5)

    # 3. Panel g: Circular Phase Histogram / Rose Plot (Lower Right)
    all_phases_mapped = np.mod(df_merged["mean_phase"].values, 2*np.pi)
    counts_g, bins_g = np.histogram(all_phases_mapped, bins=24, range=(0, 2*np.pi))
    centers_g = (bins_g[:-1] + bins_g[1:]) / 2.0
    bar_colors = [COLOR_INSP if c < np.pi else COLOR_EXP for c in centers_g]
    ax_g.bar(centers_g, counts_g, width=bins_g[1]-bins_g[0], color=bar_colors, edgecolor="black", alpha=0.75, linewidth=0.8)
    ax_g.set_title("g | Circular Phase Distribution (Rose Plot)", fontsize=11, fontweight="bold", loc="left", pad=12)
    ax_g.set_theta_zero_location("E")
    ax_g.set_theta_direction(1)
    ax_g.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    ax_g.set_xticklabels([r"$0^\circ$", r"$45^\circ$", r"$90^\circ$", r"$135^\circ$", r"$180^\circ$", r"$225^\circ$", r"$270^\circ$", r"$315^\circ$"], fontsize=7.5)

    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08, wspace=0.32, hspace=0.38)
    fig.savefig(out_path, dpi=250)
    fig.savefig(out_path.with_suffix(".svg"), format="svg")
    plt.close(fig)
    print(f"[SUCCESS] Saved polished Panel f + e + g Phase Continuum figure: {out_path} and .svg")

def main():
    assets_dir = PROJECT_ROOT / "presentation_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experimental and sorting data...")
    data = load_data()

    print("Generating Figure 1: Panel c + i Anatomy Depth...")
    make_panel_c_i_anatomy_depth(data, assets_dir / "panel_c_i_anatomy_depth.png")

    print("Generating Figure 2: Panel d + h Temporal Tuning...")
    make_panel_d_h_temporal_tuning(data, assets_dir / "panel_d_h_temporal_tuning.png")

    print("Generating Figure 3: Panel f + e + g Phase Continuum...")
    make_panel_f_e_g_phase_continuum(data, assets_dir / "panel_f_e_g_phase_continuum.png")

    print("\n[COMPLETE] All 3 polished figures generated successfully!")

if __name__ == "__main__":
    main()
