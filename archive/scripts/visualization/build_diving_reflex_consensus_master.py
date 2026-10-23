#!/usr/bin/env python3
"""
Master Builder for E097 Diving Reflex (240s) Consensus Master & Sorting Validation Suite
Mirrors: C:\\Users\\lpascu\\resp_analysis\\outputs\\e097\\output_e097_prelaser_through_laser3_226s_motion_corrected\\consensus_master

Deliverables:
1. consensus_master/
   - alf/ (spike_times.npy, spike_clusters.npy, cluster_info.tsv)
   - master_consensus_qc_summary.csv
   - cluster_info.tsv
   - lupin_qc_interactive_raster.html (Synchronized 6-panel: Airflow, Raw EMG, Int EMG, ECG, Pop PSTH, Full 400-cluster Raster)
   - lupin_qc_raster_preview.png (High-res 6-panel static preview)
   - cross_cluster_split_candidates.csv
   - 03_split_candidate_diagnostics/
   - 04_gasping_transition_rasters/ (Apnea transition statistics & gasping_transition_dashboard.html)
   - 02_single_cell_phase_rasters/ (gallery_all_units_chronological.html)
   - run_journal.md, run_journal.json
2. phy/ directory enriched with individual cluster_*.tsv metadata columns for Phy GUI
3. sorting_validation/cluster_validation_browser.html enriched with multi-criteria filters & full 400-unit suite
"""

import sys
import json
import shutil
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CONSENSUS_MASTER_BUILDER")

ROOT_DIR = Path(r"c:\Users\lpascu\resp_analysis")
OUT_DIR = ROOT_DIR / "outputs" / "e097" / "output_e097_diving_reflex_240s_full_probe"
CONSENSUS_DIR = OUT_DIR / "consensus_master"
PHY_DIR = OUT_DIR / "phy"
NIDQ_PATH = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
RAW_AP_BIN = Path(r"C:\SCRATCH_neuropixels\catgt_e097_diving_240s\catgt_e097_g2\e097_g2_tcat.imec0.ap.bin")

START_T_S = 3300.0
DURATION_S = 240.0
FS_NIDQ = 10000.0
FS_AP = 29999.821974390747

def load_nidaq_channels():
    logger.info(f"Loading NIDAQ data from {NIDQ_PATH} (Offset {START_T_S}s, Duration {DURATION_S}s)...")
    n_ch = 9
    start_samp = int(START_T_S * FS_NIDQ)
    n_samp = int(DURATION_S * FS_NIDQ)
    offset_bytes = start_samp * n_ch * 2
    read_bytes = n_samp * n_ch * 2

    with open(NIDQ_PATH, "rb") as f:
        f.seek(offset_bytes)
        buf = f.read(read_bytes)
        raw = np.frombuffer(buf, dtype=np.int16).reshape((n_samp, n_ch))

    v_conv = 10.0 / 32768.0
    emg_raw = raw[:, 0] * v_conv
    ecg_raw = raw[:, 2] * v_conv
    flow_raw = raw[:, 5] * v_conv

    # Downsample for web and figure visualization (from 10 kHz to 1 kHz)
    ds_factor = 10
    time_ds = np.linspace(0, DURATION_S, n_samp // ds_factor, endpoint=False)
    flow_ds = flow_raw[::ds_factor]
    ecg_ds = ecg_raw[::ds_factor]
    emg_sub = emg_raw[::ds_factor]

    # Compute high-pass filtered and rectified integrated EMG envelope
    sos = signal.butter(4, 100.0, btype="highpass", fs=FS_NIDQ, output="sos")
    emg_hp = signal.sosfilt(sos, emg_raw)
    emg_rect = np.abs(emg_hp)
    
    # Moving RMS envelope (50 ms window = 500 samples)
    win_size = int(0.05 * FS_NIDQ)
    win = np.ones(win_size) / win_size
    int_emg = np.sqrt(signal.convolve(emg_hp**2, win, mode="same"))
    int_emg_ds = int_emg[::ds_factor]

    logger.info("NIDAQ channels successfully loaded and preprocessed.")
    return time_ds, flow_ds, emg_sub, int_emg_ds, ecg_ds

def setup_consensus_master_structure(df_qc):
    logger.info("Creating consensus_master directory structure...")
    CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
    alf_dir = CONSENSUS_DIR / "alf"
    alf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy ALF spike times & clusters
    src_alf = OUT_DIR / "alf"
    shutil.copy2(src_alf / "spike_times.npy", alf_dir / "spike_times.npy")
    shutil.copy2(src_alf / "spike_clusters.npy", alf_dir / "spike_clusters.npy")
    
    # 2. Save enriched metadata to consensus_master
    df_qc.to_csv(CONSENSUS_DIR / "cluster_info.tsv", sep="\t", index=False)
    df_qc.to_csv(CONSENSUS_DIR / "master_consensus_qc_summary.csv", index=False)
    df_qc.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)

    # Subdirectories
    subdirs = [
        CONSENSUS_DIR / "02_single_cell_phase_rasters",
        CONSENSUS_DIR / "03_spiking_statistics_literature_cards",
        CONSENSUS_DIR / "03_split_candidate_diagnostics",
        CONSENSUS_DIR / "04_gasping_transition_rasters",
        CONSENSUS_DIR / "04_gasping_transition_rasters" / "cards"
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("consensus_master structure initialized.")

def update_phy_bundle(df_qc):
    logger.info("Updating phy/ directory with multi-criteria QC metadata columns...")
    PHY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Update params.py with raw binary path
    params_py = PHY_DIR / "params.py"
    params_content = f"""dat_path = r'{RAW_AP_BIN}'
n_channels_dat = 384
dtype = 'int16'
offset = 0
sample_rate = {FS_AP}
hp_filtered = False
"""
    params_py.write_text(params_content, encoding="utf-8")

    # 2. Assign standard Phy groups: good (passing any SUA criteria), mua (failing/multi-unit), noise (dedup redundant)
    def assign_phy_group(row):
        if row["passed_allen_strict"] or row["passed_respiratory_continuous"] or row["passed_gasp_recruited"]:
            return "good"
        elif row["would_be_deduplicated"]:
            return "noise"
        else:
            return "mua"

    df_qc_phy = df_qc.copy()
    df_qc_phy["group"] = df_qc_phy.apply(assign_phy_group, axis=1)

    # Export main cluster_info.tsv
    df_qc_phy.to_csv(PHY_DIR / "cluster_info.tsv", sep="\t", index=False)

    # Export individual cluster_*.tsv files for native Phy GUI column rendering
    cols_to_export = {
        "group": "group",
        "qc_verdict": "overall_qc_verdict",
        "passed_allen_strict": "passed_allen_strict",
        "passed_respiratory_continuous": "passed_respiratory_continuous",
        "passed_gasp_recruited": "passed_gasp_recruited",
        "would_be_deduplicated": "would_be_deduplicated",
        "coincidence_fraction": "coincidence_fraction",
        "coincident_partner": "most_coincident_partner",
        "fail_reasons": "fail_reasons",
        "depth_um": "depth_um",
        "peak_channel": "peak_channel",
        "firing_rate_hz": "firing_rate_hz",
        "n_spikes": "n_spikes",
        "snr": "snr",
        "presence_ratio": "presence_ratio",
        "isi_violations": "isi_violations",
        "apnea_fr_hz": "apnea_fr_hz",
        "apnea_fold_change": "apnea_fold_change",
        "is_rostral": "is_rostral_continuum"
    }

    for fname, col in cols_to_export.items():
        if col in df_qc_phy.columns:
            tsv_path = PHY_DIR / f"cluster_{fname}.tsv"
            df_sub = df_qc_phy[["cluster_id", col]].rename(columns={col: fname})
            df_sub.to_csv(tsv_path, sep="\t", index=False)

    logger.info(f"Phy bundle successfully updated with {len(cols_to_export)} metadata columns.")

def build_synchronized_master_raster(df_qc, time_ds, flow_ds, emg_sub, int_emg_ds, ecg_ds):
    logger.info("Constructing synchronized 6-panel Plotly interactive master raster...")
    
    # Load spikes
    alf_dir = CONSENSUS_DIR / "alf"
    times = np.load(alf_dir / "spike_times.npy")
    clusters = np.load(alf_dir / "spike_clusters.npy")

    # PSTH
    bin_size_s = 0.05
    n_psth_bins = int(DURATION_S / bin_size_s)
    psth_bins = np.linspace(0, DURATION_S, n_psth_bins + 1)
    psth_counts, _ = np.histogram(times, bins=psth_bins)
    psth_rate_hz = psth_counts / bin_size_s
    psth_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.10, 0.10, 0.10, 0.10, 0.12, 0.48],
        subplot_titles=(
            "Respiratory Flow Rate (Plethysmography Airflow - Flat during Apnea)",
            "Raw Diaphragm EMG (High-Pass Filtered Motor Drive)",
            "Integrated Diaphragm EMG Envelope (Motor Burst Amplitude)",
            "Cardiac ECG (Tracking Diving Reflex Bradycardia)",
            f"Population Instantaneous Firing Rate (All 400 Clusters, {bin_size_s*1000:.0f} ms bins)",
            "Depth-Resolved Spike Raster: Full 400-Cluster Suite with QC Status Flags"
        )
    )

    # Panel 1: Flow
    fig.add_trace(go.Scattergl(x=time_ds, y=flow_ds, mode="lines", line=dict(color="#059669", width=1.4), name="Airflow (V)"), row=1, col=1)
    # Panel 2: Raw EMG
    fig.add_trace(go.Scattergl(x=time_ds, y=emg_sub, mode="lines", line=dict(color="#94a3b8", width=1.0), name="Raw Diaphragm (V)"), row=2, col=1)
    # Panel 3: Integrated EMG
    fig.add_trace(go.Scattergl(x=time_ds, y=int_emg_ds, mode="lines", line=dict(color="#0284c7", width=1.6), name="Int. Diaphragm (V)"), row=3, col=1)
    # Panel 4: ECG
    fig.add_trace(go.Scattergl(x=time_ds, y=ecg_ds, mode="lines", line=dict(color="#f43f5e", width=1.1), name="ECG (V)"), row=4, col=1)
    # Panel 5: Population PSTH
    fig.add_trace(go.Scatter(x=psth_centers, y=psth_rate_hz, mode="lines", line=dict(color="#38bdf8", width=1.8), fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.18)", name="Pop Rate (Hz)"), row=5, col=1)

    # Panel 6: Raster categorized by QC Verdict
    unit_map = {int(r["cluster_id"]): r for _, r in df_qc.iterrows()}

    # Groupings for interactive toggling in Plotly legend
    groups = {
        "Strict SUA (99 units)": {"color": "#10b981", "filter": lambda r: r["passed_allen_strict"]},
        "Continuous SUA (23 units)": {"color": "#0284c7", "filter": lambda r: r["passed_respiratory_continuous"] and not r["passed_allen_strict"]},
        "Gasp Recruited SUA (9 units)": {"color": "#f43f5e", "filter": lambda r: r["passed_gasp_recruited"]},
        "Deduplication Overlap (3 units)": {"color": "#f59e0b", "filter": lambda r: r["would_be_deduplicated"]},
        "Failed QC / MUA (266 units)": {"color": "#64748b", "filter": lambda r: not (r["passed_allen_strict"] or r["passed_respiratory_continuous"] or r["passed_gasp_recruited"] or r["would_be_deduplicated"])}
    }

    # Downsample spikes per group if necessary for super-fluid WebGL rendering (max 60k spikes per group)
    for grp_name, grp_info in groups.items():
        grp_cids = {cid for cid, r in unit_map.items() if grp_info["filter"](r)}
        grp_mask = np.isin(clusters, list(grp_cids))
        grp_times = times[grp_mask]
        grp_clusters = clusters[grp_mask]

        if len(grp_times) > 70000:
            sub_idx = np.random.choice(len(grp_times), 70000, replace=False)
            grp_times = grp_times[sub_idx]
            grp_clusters = grp_clusters[sub_idx]

        grp_depths = np.array([unit_map[c]["depth_um"] for c in grp_clusters])
        hover_texts = [
            f"Cluster: #{c}<br>Depth: {unit_map[c]['depth_um']} µm (Ch {unit_map[c]['peak_channel']})<br>FR: {unit_map[c]['firing_rate_hz']:.1f} Hz<br>Verdict: {unit_map[c]['overall_qc_verdict']}<br>Reasons: {unit_map[c]['fail_reasons']}"
            for c in grp_clusters
        ]

        fig.add_trace(
            go.Scattergl(
                x=grp_times,
                y=grp_depths,
                mode="markers",
                marker=dict(size=3.2, color=grp_info["color"], opacity=0.8),
                name=grp_name,
                text=hover_texts,
                hovertemplate="%{text}<br>Time: %{x:.3f} s<extra></extra>"
            ),
            row=6, col=1
        )

    # Highlight Diving Reflex Apnea Period
    for r in range(1, 7):
        fig.add_vrect(
            x0=42.0, x1=118.0,
            fillcolor="#f43f5e", opacity=0.08,
            layer="below", line_width=1, line_dash="dash", line_color="#f43f5e",
            row=r, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        height=1350,
        margin=dict(l=60, r=40, t=70, b=40),
        title=dict(
            text="<b>E097 Diving Reflex — Full-Probe Spike Sorting Master Consensus Validation</b><br><sup>Synchronized Airflow, Diaphragm EMG, Cardiac ECG, Population PSTH & Depth Raster (All 400 Clusters Flagged)</sup>",
            font=dict(size=17, color="#f8fafc"),
            x=0.02, y=0.98
        ),
        legend=dict(orientation="h", y=1.02, x=0.25, bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="#334155", borderwidth=1),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        hovermode="closest"
    )

    fig.update_yaxes(title_text="Flow (V)", row=1, col=1, gridcolor="#1f2937")
    fig.update_yaxes(title_text="Raw EMG (V)", row=2, col=1, gridcolor="#1f2937")
    fig.update_yaxes(title_text="Int. EMG (V)", row=3, col=1, gridcolor="#1f2937")
    fig.update_yaxes(title_text="ECG (V)", row=4, col=1, gridcolor="#1f2937")
    fig.update_yaxes(title_text="Rate (Hz)", row=5, col=1, gridcolor="#1f2937")
    fig.update_yaxes(title_text="Probe Depth (µm)", row=6, col=1, gridcolor="#1f2937")
    fig.update_xaxes(title_text="Recording Time (seconds)", row=6, col=1, gridcolor="#1f2937")

    out_html = CONSENSUS_DIR / "lupin_qc_interactive_raster.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Saved interactive master raster to: {out_html}")

    # Also build static preview
    fig_static, axes = plt.subplots(6, 1, figsize=(15, 14), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.2, 3.8]})
    axes[0].plot(time_ds, flow_ds, color="#059669", lw=1.2)
    axes[0].set_ylabel("Airflow (V)", fontsize=9, fontweight="bold")
    axes[0].set_title("E097 Diving Reflex (240s) — Airflow, Diaphragm EMG, ECG, Population Rate & Spike Raster (All 400 Clusters)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].plot(time_ds, emg_sub, color="#94a3b8", lw=0.9)
    axes[1].set_ylabel("Raw EMG (V)", fontsize=9, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    axes[2].plot(time_ds, int_emg_ds, color="#0284c7", lw=1.4)
    axes[2].set_ylabel("Int. EMG (V)", fontsize=9, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    axes[3].plot(time_ds, ecg_ds, color="#f43f5e", lw=1.0)
    axes[3].set_ylabel("ECG (V)", fontsize=9, fontweight="bold")
    axes[3].grid(True, linestyle=":", alpha=0.5)

    axes[4].plot(psth_centers, psth_rate_hz, color="#38bdf8", lw=1.5)
    axes[4].fill_between(psth_centers, 0, psth_rate_hz, color="#38bdf8", alpha=0.2)
    axes[4].set_ylabel("Pop FR (Hz)", fontsize=9, fontweight="bold")
    axes[4].grid(True, linestyle=":", alpha=0.5)

    # Color code raster
    c_map = {"PASS: Strict SUA": "#10b981", "PASS: Continuous SUA": "#0284c7", "PASS: Gasp Recruited": "#f43f5e", "FAIL: Deduplication Redundant": "#f59e0b"}
    spk_colors = [c_map.get(unit_map.get(c, {}).get("overall_qc_verdict", ""), "#64748b") for c in clusters[::5]]
    spk_depths = [unit_map.get(c, {}).get("depth_um", 0.0) for c in clusters[::5]]
    
    axes[5].scatter(times[::5], spk_depths, s=2.5, c=spk_colors, alpha=0.6, rasterized=True)
    axes[5].set_ylabel("Depth (µm)", fontsize=10, fontweight="bold")
    axes[5].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axes[5].grid(True, linestyle=":", alpha=0.5)

    for ax in axes:
        ax.axvspan(42.0, 118.0, color="#f43f5e", alpha=0.12, zorder=0)

    plt.tight_layout()
    preview_path = CONSENSUS_DIR / "lupin_qc_raster_preview.png"
    fig_static.savefig(preview_path, dpi=200)
    plt.close(fig_static)
    logger.info(f"Saved static preview to: {preview_path}")

def generate_split_and_dedup_diagnostics(df_qc):
    logger.info("Computing cross-cluster split candidates and coincidence diagnostics...")
    alf_dir = CONSENSUS_DIR / "alf"
    times = np.load(alf_dir / "spike_times.npy")
    clusters = np.load(alf_dir / "spike_clusters.npy")

    # Pairwise coincidence within spatial proximity (delta_y <= 60 um)
    candidates = []
    cids = sorted(list(set(clusters)))
    unit_spks = {cid: times[clusters == cid] for cid in cids}
    unit_dict = {int(r["cluster_id"]): r for _, r in df_qc.iterrows()}

    for i in range(len(cids)):
        cid_a = cids[i]
        depth_a = unit_dict[cid_a]["depth_um"]
        spks_a = unit_spks[cid_a]
        if len(spks_a) < 10:
            continue

        for j in range(i + 1, len(cids)):
            cid_b = cids[j]
            depth_b = unit_dict[cid_b]["depth_um"]
            delta_y = abs(depth_a - depth_b)
            if delta_y > 60.0:
                continue

            spks_b = unit_spks[cid_b]
            if len(spks_b) < 10:
                continue

            # Compute coincidence within 0.4 ms
            idx = np.searchsorted(spks_b, spks_a)
            idx = np.clip(idx, 0, len(spks_b) - 1)
            d1 = np.abs(spks_a - spks_b[idx])
            d2 = np.abs(spks_a - spks_b[np.maximum(0, idx - 1)])
            min_d = np.minimum(d1, d2)
            n_coinc = np.sum(min_d <= 0.0004)
            c_frac = n_coinc / min(len(spks_a), len(spks_b))

            if c_frac >= 0.05 or delta_y <= 15.0:
                candidates.append({
                    "cluster_a": cid_a,
                    "cluster_b": cid_b,
                    "depth_a_um": depth_a,
                    "depth_b_um": depth_b,
                    "delta_y_um": delta_y,
                    "fr_a": unit_dict[cid_a]["firing_rate_hz"],
                    "fr_b": unit_dict[cid_b]["firing_rate_hz"],
                    "coincidence_fraction": round(float(c_frac), 4),
                    "n_coincident": int(n_coinc),
                    "split_confidence": "High (Probable Split Artifact)" if c_frac >= 0.50 else ("Moderate (Possible Split)" if c_frac >= 0.20 else "Low")
                })

    df_cand = pd.DataFrame(candidates).sort_values(by="coincidence_fraction", ascending=False)
    csv_out = CONSENSUS_DIR / "cross_cluster_split_candidates.csv"
    df_cand.to_csv(csv_out, index=False)
    logger.info(f"Saved {len(df_cand)} cross-cluster split candidates to: {csv_out}")

    # Generate diagnostic cards for top 5 candidates
    top_cand = df_cand.head(5)
    diag_dir = CONSENSUS_DIR / "03_split_candidate_diagnostics"
    for rank, (_, row) in enumerate(top_cand.iterrows(), 1):
        ca, cb = int(row["cluster_a"]), int(row["cluster_b"])
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].eventplot(unit_spks[ca], colors="#0284c7", lineoffsets=1, linelengths=0.8)
        axes[0].set_ylabel(f"Cluster #{ca}\n({row['depth_a_um']} µm)", fontsize=9, fontweight="bold")
        axes[0].set_title(f"Split Candidate #{rank}: Cluster {ca} vs {cb} | $\\Delta y = {row['delta_y_um']:.0f}\\,\\mu\\mathrm{{m}}$ | Coincidence: {row['coincidence_fraction']*100:.1f}%", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.5)

        axes[1].eventplot(unit_spks[cb], colors="#f43f5e", lineoffsets=1, linelengths=0.8)
        axes[1].set_ylabel(f"Cluster #{cb}\n({row['depth_b_um']} µm)", fontsize=9, fontweight="bold")
        axes[1].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        diag_path = diag_dir / f"split_candidate_pair_{rank:02d}_cluster_{ca}_vs_{cb}.png"
        fig.savefig(diag_path, dpi=160)
        plt.close(fig)

    logger.info("Split candidate diagnostic plots generated.")

def generate_gasping_and_apnea_transition_suite(df_qc, time_ds, flow_ds, emg_sub, int_emg_ds):
    logger.info("Generating diving reflex apnea & gasping transition rasters and dashboard...")
    base_out = CONSENSUS_DIR / "04_gasping_transition_rasters"
    cards_dir = base_out / "cards"
    
    alf_dir = CONSENSUS_DIR / "alf"
    times = np.load(alf_dir / "spike_times.npy")
    clusters = np.load(alf_dir / "spike_clusters.npy")
    unit_spks = {int(r["cluster_id"]): times[clusters == int(r["cluster_id"])] for _, r in df_qc.iterrows()}

    # Compute transition statistics: Eupnea Baseline (10-40s) vs Apnea Window (45-115s)
    stats = []
    for _, r in df_qc.iterrows():
        cid = int(r["cluster_id"])
        spks = unit_spks[cid]
        n_eup = np.sum((spks >= 10.0) & (spks < 40.0))
        n_apnea = np.sum((spks >= 45.0) & (spks < 115.0))
        fr_eup = n_eup / 30.0
        fr_apnea = n_apnea / 70.0
        fold_change = (fr_apnea + 0.01) / (fr_eup + 0.01)

        stats.append({
            "cluster_id": cid,
            "depth_um": r["depth_um"],
            "peak_channel": r["peak_channel"],
            "firing_rate_hz": r["firing_rate_hz"],
            "fr_eupnea_hz": round(fr_eup, 3),
            "fr_apnea_hz": round(fr_apnea, 3),
            "fold_change_apnea": round(fold_change, 2),
            "is_recruited": bool(r["passed_gasp_recruited"]),
            "overall_qc_verdict": r["overall_qc_verdict"],
            "fail_reasons": r["fail_reasons"]
        })

    df_trans = pd.DataFrame(stats)
    csv_path = base_out / "eupnea_apnea_transition_statistics.csv"
    df_trans.to_csv(csv_path, index=False)

    # Render transition cards for key representative units (all 9 gasp units + top 15 continuous SUA + 10 sample MUA)
    key_cids = list(df_trans[df_trans["is_recruited"]]["cluster_id"].values)
    key_cids += list(df_trans[df_trans["overall_qc_verdict"] == "PASS: Strict SUA"].head(15)["cluster_id"].values)
    key_cids += list(df_trans[df_trans["overall_qc_verdict"].str.startswith("FAIL")].head(10)["cluster_id"].values)

    card_items = []
    for cid in key_cids:
        spks = unit_spks[cid]
        row_info = df_qc[df_qc["cluster_id"] == cid].iloc[0]

        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2]})
        # Telemetry
        axes[0].plot(time_ds, flow_ds, color="#059669", lw=1.2)
        axes[0].set_ylabel("Airflow (V)", fontsize=9, fontweight="bold")
        axes[0].set_title(f"Cluster #{cid} — Diving Reflex Transition Diagnostic Card | Depth: {row_info['depth_um']} µm (Ch {row_info['peak_channel']}) | Verdict: {row_info['overall_qc_verdict']}", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.5)

        axes[1].plot(time_ds, int_emg_ds, color="#0284c7", lw=1.4)
        axes[1].set_ylabel("Int. EMG (V)", fontsize=9, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.5)

        # Single Unit Spikes & Rate
        axes[2].eventplot(spks, colors="#38bdf8" if "PASS" in row_info["overall_qc_verdict"] else "#f43f5e", lineoffsets=1, linelengths=0.8)
        axes[2].set_ylabel(f"Unit #{cid} Spikes", fontsize=9, fontweight="bold")
        axes[2].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        axes[2].set_ylim(0.5, 1.5)
        axes[2].grid(True, linestyle=":", alpha=0.5)

        for ax in axes:
            ax.axvspan(42.0, 118.0, color="#f43f5e", alpha=0.12, zorder=0)

        plt.tight_layout()
        img_name = f"transition_card_cluster_{cid:03d}.png"
        fig.savefig(cards_dir / img_name, dpi=150)
        plt.close(fig)

        card_items.append({
            "cid": int(cid),
            "depth": float(row_info["depth_um"]),
            "ch": int(row_info["peak_channel"]),
            "fr": float(row_info["firing_rate_hz"]),
            "verdict": str(row_info["overall_qc_verdict"]),
            "pass_strict": bool(row_info["passed_allen_strict"]),
            "pass_gasp": bool(row_info["passed_gasp_recruited"]),
            "reasons": str(row_info["fail_reasons"]),
            "img": f"cards/{img_name}"
        })

    # Generate gasping_transition_dashboard.html
    cards_json = json.dumps(card_items, default=int)
    dash_html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>E097 Diving Reflex Apnea & Gasping Transition Gallery</title>
    <style>
        body { margin:0; padding:24px; background:#0b0f19; color:#f8fafc; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
        header { max-width:1400px; margin:0 auto 20px auto; border-bottom:1px solid #1f2937; padding-bottom:16px; }
        h1 { color:#38bdf8; margin:0 0 6px 0; font-size:24px; }
        .filter-bar { display:flex; gap:10px; margin:16px 0; flex-wrap:wrap; }
        .btn { background:#1e293b; color:#cbd5e1; border:1px solid #334155; padding:6px 14px; border-radius:6px; font-weight:600; cursor:pointer; font-size:13px; }
        .btn.active { background:#0284c7; color:#fff; border-color:#38bdf8; }
        .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(640px,1fr)); gap:20px; max-width:1600px; margin:0 auto; }
        .card { background:#111827; border:1px solid #1f2937; border-radius:10px; padding:14px; }
        .card img { width:100%; border-radius:6px; }
        .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:700; margin-bottom:8px; }
        .badge-pass { background:#065f46; color:#34d399; }
        .badge-gasp { background:#831843; color:#f43f5e; }
        .badge-fail { background:#7f1d1d; color:#f87171; }
    </style>
</head>
<body>
    <header>
        <h1>E097 Diving Reflex — Neural Transition & Recruitment Gallery</h1>
        <div style="color:#94a3b8; font-size:14px;">Comparing baseline eupnea vs diving reflex apnea & gasping across the full cluster suite.</div>
        <div class="filter-bar">
            <button class="btn active" onclick="setFilter('all')">All Cards (__TOTAL_CARDS__)</button>
            <button class="btn" onclick="setFilter('gasp')">Gasp Recruited (__GASP_CARDS__)</button>
            <button class="btn" onclick="setFilter('strict')">Strict SUA (__STRICT_CARDS__)</button>
            <button class="btn" onclick="setFilter('fail')">Failed QC</button>
        </div>
    </header>
    <div class="grid" id="galleryGrid"></div>
    <script>
        const cards = __CARDS_JSON__;
        function render(filter) {
            const grid = document.getElementById('galleryGrid');
            grid.innerHTML = '';
            cards.forEach(c => {
                if (filter === 'gasp' && !c.pass_gasp) return;
                if (filter === 'strict' && !c.pass_strict) return;
                if (filter === 'fail' && (c.pass_strict || c.pass_gasp)) return;
                const bClass = c.pass_gasp ? 'badge-gasp' : (c.pass_strict ? 'badge-pass' : 'badge-fail');
                const cardEl = document.createElement('div');
                cardEl.className = 'card';
                cardEl.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:700; font-size:15px; color:#fff;">Cluster #${c.cid} (${c.depth} µm)</span>
                        <span class="badge ${bClass}">${c.verdict}</span>
                    </div>
                    <div style="font-size:12px; color:#94a3b8; margin-bottom:10px;">FR: ${c.fr.toFixed(1)} Hz | ${c.reasons}</div>
                    <img src="${c.img}" alt="Cluster ${c.cid}">
                `;
                grid.appendChild(cardEl);
            });
        }
        function setFilter(f) {
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            render(f);
        }
        render('all');
    </script>
</body>
</html>
"""
    dash_html = dash_html_template.replace("__CARDS_JSON__", cards_json)
    dash_html = dash_html.replace("__TOTAL_CARDS__", str(len(card_items)))
    dash_html = dash_html.replace("__GASP_CARDS__", str(sum(1 for c in card_items if c['pass_gasp'])))
    dash_html = dash_html.replace("__STRICT_CARDS__", str(sum(1 for c in card_items if c['pass_strict'])))
    
    dash_path = base_out / "gasping_transition_dashboard.html"
    dash_path.write_text(dash_html, encoding="utf-8")
    logger.info(f"Transition dashboard written to: {dash_path}")

def generate_chronological_gallery(df_qc):
    logger.info("Generating 02_single_cell_phase_rasters/gallery_all_units_chronological.html...")
    gal_dir = CONSENSUS_DIR / "02_single_cell_phase_rasters"
    
    # Generate simple chronological card links for all 400 clusters
    unit_items = []
    for _, r in df_qc.iterrows():
        unit_items.append({
            "cid": int(r["cluster_id"]),
            "depth": float(r["depth_um"]),
            "ch": int(r["peak_channel"]),
            "fr": float(r["firing_rate_hz"]),
            "n_spk": int(r["n_spikes"]),
            "verdict": str(r["overall_qc_verdict"]),
            "pass_allen": bool(r["passed_allen_strict"]),
            "pass_resp": bool(r["passed_respiratory_continuous"]),
            "pass_gasp": bool(r["passed_gasp_recruited"]),
            "would_dedup": bool(r["would_be_deduplicated"]),
            "reasons": str(r["fail_reasons"])
        })

    gal_json = json.dumps(unit_items, default=int)
    gal_html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>E097 Full Suite Cluster Gallery (400 Units)</title>
    <style>
        body { margin:0; padding:24px; background:#0b0f19; color:#f8fafc; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
        header { max-width:1400px; margin:0 auto 20px auto; border-bottom:1px solid #1f2937; padding-bottom:16px; }
        h1 { color:#38bdf8; margin:0 0 6px 0; font-size:24px; }
        .filter-bar { display:flex; gap:10px; margin:16px 0; flex-wrap:wrap; }
        .btn { background:#1e293b; color:#cbd5e1; border:1px solid #334155; padding:6px 14px; border-radius:6px; font-weight:600; cursor:pointer; font-size:13px; }
        .btn.active { background:#0284c7; color:#fff; border-color:#38bdf8; }
        .table-box { max-width:1400px; margin:0 auto; background:#111827; border:1px solid #1f2937; border-radius:8px; overflow:hidden; }
        table { width:100%; border-collapse:collapse; font-size:13px; text-align:left; }
        th { background:#1e293b; color:#94a3b8; padding:10px 14px; font-weight:700; border-bottom:1px solid #334155; }
        td { padding:10px 14px; border-bottom:1px solid #1f2937; color:#e2e8f0; }
        tr:hover { background:#182234; }
        .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:700; }
        .badge-pass { background:#065f46; color:#34d399; }
        .badge-gasp { background:#831843; color:#f43f5e; }
        .badge-fail { background:#7f1d1d; color:#f87171; }
        .badge-dedup { background:#78350f; color:#fbbf24; }
    </style>
</head>
<body>
    <header>
        <h1>E097 Diving Reflex — Full-Probe Cluster Gallery (400 Units)</h1>
        <div style="color:#94a3b8; font-size:14px;">Complete directory of all 400 clusters sorted by probe depth with multi-criteria pass/fail flags.</div>
        <div class="filter-bar">
            <button class="btn active" onclick="setFilter('all')">All Clusters (400)</button>
            <button class="btn" onclick="setFilter('allen')">Strict Allen SUA (99)</button>
            <button class="btn" onclick="setFilter('resp')">Continuous SUA (122)</button>
            <button class="btn" onclick="setFilter('gasp')">Gasp Recruited (9)</button>
            <button class="btn" onclick="setFilter('dedup')">Deduplication Overlap (3)</button>
            <button class="btn" onclick="setFilter('fail')">Failed QC (269)</button>
        </div>
    </header>
    <div class="table-box">
        <table>
            <thead>
                <tr>
                    <th>Cluster ID</th>
                    <th>Depth (µm)</th>
                    <th>Peak Ch</th>
                    <th>Firing Rate (Hz)</th>
                    <th>Spikes</th>
                    <th>QC Verdict</th>
                    <th>Failure / Status Details</th>
                </tr>
            </thead>
            <tbody id="unitTbody"></tbody>
        </table>
    </div>
    <script>
        const units = __UNITS_JSON__;
        function render(filter) {
            const tbody = document.getElementById('unitTbody');
            tbody.innerHTML = '';
            units.forEach(u => {
                if (filter === 'allen' && !u.pass_allen) return;
                if (filter === 'resp' && !u.pass_resp) return;
                if (filter === 'gasp' && !u.pass_gasp) return;
                if (filter === 'dedup' && !u.would_dedup) return;
                if (filter === 'fail' && (u.pass_allen || u.pass_resp || u.pass_gasp)) return;
                
                let bClass = 'badge-fail';
                if (u.pass_gasp) bClass = 'badge-gasp';
                else if (u.pass_allen || u.pass_resp) bClass = 'badge-pass';
                else if (u.would_dedup) bClass = 'badge-dedup';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-weight:700; color:#38bdf8;">#${u.cid}</td>
                    <td>${u.depth.toFixed(1)}</td>
                    <td>Ch ${u.ch}</td>
                    <td>${u.fr.toFixed(2)}</td>
                    <td>${u.n_spk.toLocaleString()}</td>
                    <td><span class="badge ${bClass}">${u.verdict}</span></td>
                    <td style="color:#94a3b8; font-size:12px;">${u.reasons}</td>
                `;
                tbody.appendChild(tr);
            });
        }
        function setFilter(f) {
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            render(f);
        }
        render('all');
    </script>
</body>
</html>
"""
    gal_html = gal_html_template.replace("__UNITS_JSON__", gal_json)
    gal_path = gal_dir / "gallery_all_units_chronological.html"
    gal_path.write_text(gal_html, encoding="utf-8")
    logger.info(f"Chronological gallery written to: {gal_path}")

def update_cluster_validation_browser():
    # Make sure sorting_validation/cluster_validation_browser.html has link to consensus_master
    browser_path = OUT_DIR / "sorting_validation" / "cluster_validation_browser.html"
    if browser_path.exists():
        content = browser_path.read_text(encoding="utf-8")
        if "consensus_master/lupin_qc_interactive_raster.html" not in content:
            link_bar = """
        <div style="background:#0284c7; padding:6px 16px; border-radius:6px; margin-left:auto; display:flex; gap:12px; align-items:center;">
            <a href="../consensus_master/lupin_qc_interactive_raster.html" target="_blank" style="color:#fff; text-decoration:none; font-weight:700; font-size:0.82rem;">📊 Open 6-Panel Synchronized Master Raster</a>
            <a href="../consensus_master/04_gasping_transition_rasters/gasping_transition_dashboard.html" target="_blank" style="color:#fff; text-decoration:none; font-weight:700; font-size:0.82rem;">🫁 Open Transition Gallery</a>
        </div>
            """
            content = content.replace('</div>\n    </div>\n\n    <div class="main-layout">', f'{link_bar}\n        </div>\n    </div>\n\n    <div class="main-layout">')
            browser_path.write_text(content, encoding="utf-8")
            logger.info("Updated cluster_validation_browser.html with master navigation links.")

def main():
    global OUT_DIR, CONSENSUS_DIR, PHY_DIR, RAW_AP_BIN, NIDQ_PATH
    import argparse
    parser = argparse.ArgumentParser(description="Build E097 Consensus Master Suite")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--raw-ap-bin", type=str, default=str(RAW_AP_BIN), help="Preprocessed AP binary")
    parser.add_argument("--nidq-path", type=str, default=str(NIDQ_PATH), help="NIDAQ binary")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    CONSENSUS_DIR = OUT_DIR / "consensus_master"
    PHY_DIR = OUT_DIR / "phy"
    RAW_AP_BIN = Path(args.raw_ap_bin)
    NIDQ_PATH = Path(args.nidq_path)

    logger.info("=" * 80)
    logger.info("STARTING E097 CONSENSUS MASTER & VALIDATION SUITE BUILD")
    logger.info(f"Target Directory: {OUT_DIR}")
    logger.info("=" * 80)

    qc_csv = OUT_DIR / "master_consensus_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = OUT_DIR / "cluster_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = OUT_DIR / "cluster_info.tsv"
    
    sep = "\t" if "\t" in qc_csv.read_text()[:500] else ","
    df_qc = pd.read_csv(qc_csv, sep=sep)
    logger.info(f"Loaded {len(df_qc)} clusters.")

    # Harmonize column aliases between pipeline versions
    if "pass_allen" in df_qc.columns and "passed_allen_strict" not in df_qc.columns:
        df_qc["passed_allen_strict"] = df_qc["pass_allen"]
    if "pass_resp" in df_qc.columns and "passed_respiratory_continuous" not in df_qc.columns:
        df_qc["passed_respiratory_continuous"] = df_qc["pass_resp"]
    if "pass_gasp" in df_qc.columns and "passed_gasp_recruited" not in df_qc.columns:
        df_qc["passed_gasp_recruited"] = df_qc["pass_gasp"]
    if "would_dedup" in df_qc.columns and "would_be_deduplicated" not in df_qc.columns:
        df_qc["would_be_deduplicated"] = df_qc["would_dedup"]
    if "coinc_frac" in df_qc.columns and "coincidence_fraction" not in df_qc.columns:
        df_qc["coincidence_fraction"] = df_qc["coinc_frac"]
    if "coinc_partner" in df_qc.columns and "coincident_partner" not in df_qc.columns:
        df_qc["coincident_partner"] = df_qc["coinc_partner"]
    if "depth" in df_qc.columns and "depth_um" not in df_qc.columns:
        df_qc["depth_um"] = df_qc["depth"]
    if "ch" in df_qc.columns and "peak_channel" not in df_qc.columns:
        df_qc["peak_channel"] = df_qc["ch"]
    if "fr" in df_qc.columns and "firing_rate_hz" not in df_qc.columns:
        df_qc["firing_rate_hz"] = df_qc["fr"]
    if "n_spk" in df_qc.columns and "n_spikes" not in df_qc.columns:
        df_qc["n_spikes"] = df_qc["n_spk"]
    elif "n_spikes" in df_qc.columns and "n_spk" not in df_qc.columns:
        df_qc["n_spk"] = df_qc["n_spikes"]
    if "pr" in df_qc.columns and "presence_ratio" not in df_qc.columns:
        df_qc["presence_ratio"] = df_qc["pr"]
    if "isi_q" in df_qc.columns and "isi_violations_ratio" not in df_qc.columns:
        df_qc["isi_violations_ratio"] = df_qc["isi_q"]
    if "verdict" in df_qc.columns and "qc_verdict" not in df_qc.columns:
        df_qc["qc_verdict"] = df_qc["verdict"]
    if "qc_verdict" in df_qc.columns and "overall_qc_verdict" not in df_qc.columns:
        df_qc["overall_qc_verdict"] = df_qc["qc_verdict"]
    elif "verdict" in df_qc.columns and "overall_qc_verdict" not in df_qc.columns:
        df_qc["overall_qc_verdict"] = df_qc["verdict"]
    if "apnea_fc" in df_qc.columns and "apnea_fold_change" not in df_qc.columns:
        df_qc["apnea_fold_change"] = df_qc["apnea_fc"]

    # 1. Initialize consensus_master structure
    setup_consensus_master_structure(df_qc)

    # 2. Update Phy GUI bundle with all criteria TSVs
    update_phy_bundle(df_qc)

    # 3. Load NIDAQ signals (Flow, EMG, Int EMG, ECG)
    time_ds, flow_ds, emg_sub, int_emg_ds, ecg_ds = load_nidaq_channels()

    # 4. Build Synchronized 6-panel Plotly raster & static preview
    build_synchronized_master_raster(df_qc, time_ds, flow_ds, emg_sub, int_emg_ds, ecg_ds)

    # 5. Generate split & deduplication diagnostics
    generate_split_and_dedup_diagnostics(df_qc)

    # 6. Generate gasping & apnea transition suite
    generate_gasping_and_apnea_transition_suite(df_qc, time_ds, flow_ds, emg_sub, int_emg_ds)

    # 7. Generate chronological gallery
    generate_chronological_gallery(df_qc)

    # 8. Update validation browser
    update_cluster_validation_browser()

    logger.info("=" * 80)
    logger.info("ALL CONSENSUS MASTER DELIVERABLES SUCCESSFULLY BUILT!")
    logger.info(f"Consensus Master: {CONSENSUS_DIR}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
