#!/usr/bin/env python3
"""
Cardiorespiratory Phase Alignment Visualization
Plots 6 key breath trials:
1. Breath #105 (0 spikes, phi_C = 47.6% - Absolute Co-Silencing)
2. Breath #154 (3 spikes, phi_C = 52.8% - Secondary Co-Silencing)
3. Breath #98 (8 spikes, phi_C = 73.3%)
4. Breath #89 (9 spikes, phi_C = 74.1%)
5. Breath #90 (12 spikes, phi_C = 60.3%)
6. Breath #31 (31 spikes - Representative Baseline Mean)

Draws explicit vertical alignment markers:
- Red Dashed Line: Peak Inspiration Transition (t = 0 ms)
- Gold Solid/Dotted Lines: Cardiac QRS R-peaks
- Shaded Band: Inter-beat cardiac cycle (R-R)
"""
import sys
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

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PHASE_ALIGNMENT_VIZ")

def generate_phase_alignment_figures():
    nidq_path = Path(r"Z:\ramirezlab\pascual\rec_data\LP21_fent_EXPT_g1\LP21_fent_EXPT_g1_t0.nidq.bin")
    fs = 10000.0
    raw = np.memmap(nidq_path, dtype="int16", mode="r")
    n_samp = min(len(raw)//9, int(60.0 * fs))
    nidq = raw[:n_samp*9].reshape(-1, 9)
    t_nidq = np.arange(n_samp) / fs
    flow = nidq[:, 5] * (10.0 / 32768.0)
    ecg = nidq[:, 1] * (10.0 / 32768.0)

    # 1. Filter Flow (lowpass 10 Hz)
    b_f, a_f = signal.butter(2, 10.0, btype="lowpass", fs=fs)
    flow_filt = signal.filtfilt(b_f, a_f, flow)
    insp_peaks, _ = signal.find_peaks(-flow_filt, distance=int(0.25*fs), prominence=1.0)
    insp_times = t_nidq[insp_peaks]

    # 2. Pan-Tompkins Energy Derivative QRS Detector for Rodent ECG
    b_e, a_e = signal.butter(3, [15.0, 120.0], btype="bandpass", fs=fs)
    ecg_bp = signal.filtfilt(b_e, a_e, ecg)
    ecg_sq = (np.diff(ecg_bp, prepend=ecg_bp[0])) ** 2
    win_size = int(0.015 * fs)
    ecg_integ = np.convolve(ecg_sq, np.ones(win_size)/win_size, mode='same')
    thresh = 0.15 * np.percentile(ecg_integ, 98)
    peaks_energy, _ = signal.find_peaks(ecg_integ, distance=int(0.065 * fs), height=thresh)

    r_times_list = []
    search_win = int(0.02 * fs)
    for p in peaks_energy:
        w_start = max(0, p - search_win)
        w_end = min(len(ecg), p + search_win)
        r_times_list.append(t_nidq[w_start + np.argmax(ecg[w_start:w_end])])
    r_times = np.array(sorted(list(set(r_times_list))))
    ecg_filt = ecg_bp
    logger.info(f"Detected {len(r_times)} QRS peaks across recording (~{60.0/(np.median(np.diff(r_times))):.0f} bpm).")

    # 3. Load Spikes
    times = np.load(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\lupin\alf\spike_times.npy")
    clus = np.load(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\lupin\alf\spike_clusters.npy")
    qc_df = pd.read_csv(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\lupin\lupin_qc_summary.csv")
    qc_ids = set(qc_df[qc_df["passed_allen_qc"]]["cluster_id"])
    u_map = {r["cluster_id"]: r["depth_um"] for _, r in qc_df.iterrows()}

    # Target Breaths
    trials = [
        {"b_id": 105, "title": "A. Breath #105 (0 Spikes)", "status": "⭐ Absolute Co-Silencing (Rank 1)", "color": "#e74c3c"},
        {"b_id": 154, "title": "B. Breath #154 (3 Spikes)", "status": "⭐ Secondary Co-Silencing (Rank 2)", "color": "#e67e22"},
        {"b_id": 98,  "title": "C. Breath #98 (8 Spikes)",  "status": "Rank 3 Suppression", "color": "#f1c40f"},
        {"b_id": 89,  "title": "D. Breath #89 (9 Spikes)",  "status": "Rank 4 Suppression", "color": "#2ecc71"},
        {"b_id": 90,  "title": "E. Breath #90 (12 Spikes)", "status": "Rank 5 Suppression", "color": "#3498db"},
        {"b_id": 31,  "title": "F. Breath #31 (31 Spikes)", "status": "Baseline Mean (Asynchronous Spiking)", "color": "#9b59b6"}
    ]

    t_pre = 0.18   # 180 ms pre-inspiration
    t_post = 0.22  # 220 ms post-inspiration

    # -------------------------------------------------------------
    # Matplotlib High-Resolution 6-Trial Grid Figure (3 Rows x 6 Columns)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 6, figsize=(24, 11), sharex=True)

    for col, tr in enumerate(trials):
        b_id = tr["b_id"]
        t_insp = insp_times[b_id - 1]

        # Time window
        t_start = t_insp - t_pre
        t_end = t_insp + t_post
        idx_win = (t_nidq >= t_start) & (t_nidq <= t_end)

        rel_t_ms = (t_nidq[idx_win] - t_insp) * 1000.0
        win_flow = flow[idx_win]
        win_ecg = ecg_filt[idx_win]

        # Find R-peaks in this window
        win_r_times = r_times[(r_times >= t_start - 0.05) & (r_times <= t_end + 0.05)]
        rel_r_ms = (win_r_times - t_insp) * 1000.0

        # Find immediate surrounding R-peaks for cardiac phase
        r_prev = r_times[r_times <= t_insp][-1]
        r_next = r_times[r_times > t_insp][0]
        phi_c = (t_insp - r_prev) / (r_next - r_prev)
        rr_ms = (r_next - r_prev) * 1000.0

        rel_r_prev_ms = (r_prev - t_insp) * 1000.0
        rel_r_next_ms = (r_next - t_insp) * 1000.0

        # ---------------------------------------------------------
        # Row 1: Respiratory Flow Rate Trace
        # ---------------------------------------------------------
        ax_f = axes[0, col]
        ax_f.plot(rel_t_ms, win_flow, color="#e17055", lw=1.8)
        ax_f.axhline(0, color="gray", linestyle=":", lw=1.0)
        ax_f.axvline(0, color="#e74c3c", linestyle="--", lw=2.0, label="Peak Insp (t=0)")
        
        # Draw R-peak lines on flow
        for r_ms in rel_r_ms:
            ax_f.axvline(r_ms, color="#f39c12", linestyle=":", lw=1.3)

        ax_f.set_title(f"{tr['title']}\n{tr['status']}", fontsize=10, fontweight="bold", pad=8)
        if col == 0:
            ax_f.set_ylabel("Flow (V)\n[Airflow]", fontsize=9, fontweight="bold")
        ax_f.grid(True, linestyle=":", alpha=0.5)

        # Badge
        ax_f.text(
            0.04, 0.88, f"Cardiac Phase φ_C = {phi_c*100:.1f}%\nRR = {rr_ms:.1f} ms",
            transform=ax_f.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", alpha=0.85, edgecolor="#95a5a6")
        )

        # ---------------------------------------------------------
        # Row 2: ECG Trace with R-Peak Alignment
        # ---------------------------------------------------------
        ax_e = axes[1, col]
        ax_e.plot(rel_t_ms, win_ecg, color="#f39c12", lw=1.4)
        ax_e.axhline(0, color="gray", linestyle=":", lw=0.8)
        ax_e.axvline(0, color="#e74c3c", linestyle="--", lw=2.0)

        # Highlight cardiac R-R cycle straddling t=0
        ax_e.axvspan(rel_r_prev_ms, rel_r_next_ms, color="#f1c40f", alpha=0.15, label="Straddling R-R Cycle")

        for r_ms in rel_r_ms:
            ax_e.axvline(r_ms, color="#d35400", linestyle="-", lw=1.5)
            if -t_pre*1000 <= r_ms <= t_post*1000:
                ax_e.text(r_ms, np.max(win_ecg)*0.85, "QRS", color="#d35400", fontsize=7.5, fontweight="bold", ha="center")

        if col == 0:
            ax_e.set_ylabel("ECG (V)\n[Cardiac]", fontsize=9, fontweight="bold")
        ax_e.grid(True, linestyle=":", alpha=0.5)

        # ---------------------------------------------------------
        # Row 3: Single-Unit Spike Raster Across Probe Depth
        # ---------------------------------------------------------
        ax_s = axes[2, col]
        spk_mask = (times >= t_start) & (times <= t_end) & np.isin(clus, list(qc_ids))
        spk_times_win = times[spk_mask]
        spk_clus_win = clus[spk_mask]

        rel_spk_ms = (spk_times_win - t_insp) * 1000.0
        spk_depths = np.array([u_map[c] for c in spk_clus_win])

        # Separate superficial (< 150 um) vs deeper units (> 150 um)
        sh_mask = spk_depths < 150.0
        deep_mask = ~sh_mask

        # Plot deeper units in gray
        ax_s.scatter(rel_spk_ms[deep_mask], spk_depths[deep_mask], s=8, color="#7f8c8d", alpha=0.6, label="Deeper Units (>150 µm)")
        # Plot superficial units in vibrant color
        ax_s.scatter(rel_spk_ms[sh_mask], spk_depths[sh_mask], s=16, color="#00cec9", alpha=0.95, edgecolors="#0984e3", lw=0.5, label="Superficial (<150 µm)")

        # Vertical alignment lines
        ax_s.axvline(0, color="#e74c3c", linestyle="--", lw=2.0)
        for r_ms in rel_r_ms:
            ax_s.axvline(r_ms, color="#f39c12", linestyle=":", lw=1.3)

        # Highlight 50ms test window around inspiration [-25ms, +25ms]
        ax_s.axvspan(-25, 25, color="#e74c3c", alpha=0.08)

        if col == 0:
            ax_s.set_ylabel("Probe Depth (µm)\n[0 = Dorsal]", fontsize=9, fontweight="bold")
        ax_s.set_xlabel("Time relative to Insp (ms)", fontsize=9, fontweight="bold")
        ax_s.set_xlim([-t_pre * 1000, t_post * 1000])
        ax_s.set_ylim([0, 2000])
        ax_s.grid(True, linestyle=":", alpha=0.5)

        # Highlight superficial silencing zone
        ax_s.axhspan(0, 150, color="#00cec9", alpha=0.10)

        # Count spikes in 50ms window
        cnt_50ms = np.sum(sh_mask & (rel_spk_ms >= -25) & (rel_spk_ms <= 25))
        ax_s.text(
            0.04, 0.88, f"Superficial: {cnt_50ms} spikes",
            transform=ax_s.transAxes, fontsize=8.5, fontweight="bold",
            color="#d63031" if cnt_50ms < 5 else "#2d3436",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", alpha=0.85, edgecolor="#95a5a6")
        )

    # Global legend and titles
    plt.suptitle(
        "Cardiorespiratory Phase Concurrence & Network Co-Silencing: 6-Trial Comparative Analysis\n"
        "Red Dashed Line = Peak Inspiration (t = 0 ms) | Gold Lines = Cardiac QRS Complexes | Shaded Band = R-R Cycle",
        fontsize=14, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_png = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\cardiorespiratory_phase_alignment_6trials.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    logger.info(f"Saved 6-trial alignment figure to: {out_png}")

    # -------------------------------------------------------------
    # Plotly Interactive 6-Column Dashboard
    # -------------------------------------------------------------
    fig_plotly = make_subplots(
        rows=3, cols=6,
        subplot_titles=[
            f"{tr['title']}" for tr in trials
        ] + [
            f"ECG: φ_C = {( (insp_times[tr['b_id']-1] - r_times[r_times <= insp_times[tr['b_id']-1]][-1]) / (r_times[r_times > insp_times[tr['b_id']-1]][0] - r_times[r_times <= insp_times[tr['b_id']-1]][-1]) )*100:.1f}%" for tr in trials
        ] + [
            f"Raster (Depth 0-2000 µm)" for tr in trials
        ],
        vertical_spacing=0.06, horizontal_spacing=0.035,
        shared_xaxes=True
    )

    for col_idx, tr in enumerate(trials):
        col = col_idx + 1
        b_id = tr["b_id"]
        t_insp = insp_times[b_id - 1]
        t_start = t_insp - t_pre
        t_end = t_insp + t_post
        idx_win = (t_nidq >= t_start) & (t_nidq <= t_end)
        rel_t_ms = (t_nidq[idx_win] - t_insp) * 1000.0

        # Row 1: Flow
        fig_plotly.add_trace(
            go.Scatter(x=rel_t_ms, y=flow[idx_win], mode="lines", line=dict(color="#ff7675", width=2), name="Flow Rate", showlegend=(col==1)),
            row=1, col=col
        )

        # Row 2: ECG
        fig_plotly.add_trace(
            go.Scatter(x=rel_t_ms, y=ecg_filt[idx_win], mode="lines", line=dict(color="#fdcb6e", width=1.8), name="ECG", showlegend=(col==1)),
            row=2, col=col
        )

        # Row 3: Raster
        spk_mask = (times >= t_start) & (times <= t_end) & np.isin(clus, list(qc_ids))
        spk_times_win = times[spk_mask]
        spk_clus_win = clus[spk_mask]
        rel_spk_ms = (spk_times_win - t_insp) * 1000.0
        spk_depths = np.array([u_map[c] for c in spk_clus_win])

        fig_plotly.add_trace(
            go.Scatter(
                x=rel_spk_ms, y=spk_depths,
                mode="markers",
                marker=dict(size=4, color=spk_depths, colorscale="Viridis", opacity=0.85),
                name="Spikes", showlegend=(col==1)
            ),
            row=3, col=col
        )

        # Add vertical line at t=0 on all rows
        for r_i in [1, 2, 3]:
            fig_plotly.add_vline(x=0, line_dash="dash", line_color="#e74c3c", line_width=1.5, row=r_i, col=col)

    fig_plotly.update_layout(
        template="plotly_dark",
        title="<b>Cardiorespiratory Phase Concurrence & Network Co-Silencing: 6-Trial Interactive Dashboard</b><br><sup>Red Dashed Line = Peak Inspiration (t = 0 ms) | Synchronized Across Respiratory Flow, Cardiac ECG, and Spike Raster</sup>",
        height=950,
        margin=dict(l=60, r=40, t=90, b=50)
    )

    out_html = Path(r"c:\Users\lpascu\analysis_src\output_LP21_fent_EXPT_g1\cardiorespiratory_phase_alignment_dashboard.html")
    fig_plotly.write_html(str(out_html))
    logger.info(f"Saved interactive dashboard to: {out_html}")
    return out_png, out_html

def main():
    out_png, out_html = generate_phase_alignment_figures()
    print("\n" + "=" * 85)
    print(" [SUCCESS] Cardiorespiratory Phase Alignment Visualizations Generated!")
    print(f" PNG High-Res Figure : {out_png}")
    print(f" Interactive HTML    : {out_html}")
    print("=" * 85)

if __name__ == "__main__":
    main()
