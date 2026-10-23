#!/usr/bin/env python3
r"""
Respiratory Phase Preference Classification Pipeline (Bush et al. 2025 Criteria)

Classifies single units recorded in the VRC by their respiratory phase modulation:
1. Respiratory Modulation (Mazurek Vector Strength R in [0, 1]):
   - R < 0.20  --> TONIC (Unmodulated / Continuous background)
   - R >= 0.20 --> PHASIC
2. Preferred Phase (phi_pref in [-pi, pi] radians):
   - phi = 0   : Inspiration Onset (Diaphragm activation)
   - phi = pi  : Inspiration Offset (Diaphragm peak/inactivation)
   - [0, pi]   : INSPIRATORY (Insp)
     - [0, pi/2]    --> Early-Inspiratory / Pre-Inspiratory (Pre-I)
     - [pi/2, pi]   --> Late-Inspiratory
   - [-pi, 0)  : EXPIRATORY (Exp)
     - [-pi, -pi/2] --> Post-Inspiratory (Post-I / Early-Exp)
     - [-pi/2, 0)   --> Late-Expiratory (E2 / Pre-I transition)

Outputs:
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\respiratory_phase_classification_summary.csv
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\respiratory_phase_classification_summary.png
- c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation\respiratory_phase_classification_summary.html
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
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

def classify_unit_phase(r_val, phi_val):
    if r_val < 0.20:
        return "Tonic", "Tonic"
    
    if phi_val >= 0:
        main_class = "Inspiratory"
        if phi_val <= (np.pi / 2.0):
            sub_class = "Early-Inspiratory"
        else:
            sub_class = "Late-Inspiratory"
    else:
        main_class = "Expiratory"
        if phi_val <= -(np.pi / 2.0):
            sub_class = "Post-Inspiratory (Post-I)"
        else:
            sub_class = "Late-Expiratory (E2/Pre-I)"
            
    return main_class, sub_class

def main():
    out_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1_pacemaker_isolation")
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(r"c:\Users\lpascu\analysis_src\output_TP_Tacr1_1\alf_session")
    alf_dir = session_dir / "alf" / "probe00"
    raw_dir = Path(r"Z:\ramirezlab\pascual\rec_data\TP_Tacr1_1_npx2_g4")
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\0e7be222-f160-4bc2-921c-f3fdf8b7825e")

    print("=" * 90)
    print(" Running Respiratory Phase Preference Classification (Bush et al. 2025 Criteria)")
    print(f" ALF Directory: {alf_dir}")
    print(f" Output Directory: {out_dir}")
    print("=" * 90)

    # 1. Load Spikes & Allen Quality Metrics CSV
    qc_csv = out_dir / "allen_neuropixels_quality_metrics_baseline_60s.csv"
    if not qc_csv.exists():
        print(f" Error: {qc_csv} not found.")
        sys.exit(1)

    metrics_df = pd.read_csv(qc_csv)
    passed_df = metrics_df[metrics_df['passed_allen_qc']].copy().reset_index(drop=True)
    passed_uids = set(passed_df['cluster_id'])

    spikes_times = np.load(alf_dir / "spike_times.npy")
    spikes_clusters = np.load(alf_dir / "spike_clusters.npy")
    info_df = pd.read_csv(alf_dir / "cluster_info.tsv", sep="\t")

    fs_ap = 30000.0
    if spikes_times.dtype in [np.int64, np.int32] and spikes_times.max() > 100000:
        spikes_times = spikes_times / fs_ap

    MAX_TIME_S = 550.0
    spk_mask = (spikes_times >= 0.0) & (spikes_times <= MAX_TIME_S) & (np.isin(spikes_clusters, list(passed_uids)))
    spikes_times = spikes_times[spk_mask]
    spikes_clusters = spikes_clusters[spk_mask]

    # 2. Load NIDQ Diaphragm EMG & Compute Continuous Phase phi(t)
    nidq_meta = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.meta"
    nidq_bin = raw_dir / "TP_Tacr1_1_npx2_g4_t0.nidq.bin"
    nidq_data = read_nidq_bin(nidq_bin, nidq_meta)

    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    nidq_mask = (t_nidq >= 0.0) & (t_nidq <= MAX_TIME_S)
    t_nidq = t_nidq[nidq_mask]
    emg_raw = nidq_data.all_channels[0][nidq_mask]

    emg_cfg = EMGConfig(fs=fs_nidq, highpass_cutoff=300.0, lowpass_cutoff=3000.0, notch_freq=60.0, smoothing_window_ms=20.0)
    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = emg_proc.breath_onsets
    breath_offsets = emg_proc.breath_offsets

    # Continuous Respiratory Phase phi(t)
    dt_phase = 0.002 # 2ms resolution
    t_bins = np.arange(0.0, MAX_TIME_S, dt_phase)
    n_bins = len(t_bins)
    phi_vector = np.zeros(n_bins)

    n_cycles = min(len(breath_onsets) - 1, len(breath_offsets))
    for i in range(n_cycles):
        t_on = breath_onsets[i]
        t_off = breath_offsets[i] if breath_offsets[i] > t_on else t_on + 0.15
        t_next = breath_onsets[i + 1]

        idx_insp = np.where((t_bins >= t_on) & (t_bins < t_off))[0]
        if len(idx_insp) > 0: phi_vector[idx_insp] = np.linspace(0, np.pi, len(idx_insp))

        idx_exp = np.where((t_bins >= t_off) & (t_bins < t_next))[0]
        if len(idx_exp) > 0: phi_vector[idx_exp] = np.linspace(-np.pi, 0.0, len(idx_exp))

    # 3. Mazurek Vector Mean & Phase Preference Classification per Unit
    r_list, phi_list, main_classes, sub_classes = [], [], [], []

    for idx, row in passed_df.iterrows():
        uid = int(row['cluster_id'])
        u_spks = np.sort(spikes_times[spikes_clusters == uid])

        spk_bin_indices = np.searchsorted(t_bins, u_spks) - 1
        spk_bin_indices = spk_bin_indices[(spk_bin_indices >= 0) & (spk_bin_indices < n_bins)]
        spk_phases = phi_vector[spk_bin_indices]

        if len(spk_phases) < 15:
            r_val = 0.0
            phi_val = 0.0
        else:
            cos_m = np.mean(np.cos(spk_phases))
            sin_m = np.mean(np.sin(spk_phases))
            r_val = float(np.sqrt(cos_m**2 + sin_m**2))
            phi_val = float(np.arctan2(sin_m, cos_m))

        m_cls, s_cls = classify_unit_phase(r_val, phi_val)
        
        r_list.append(round(r_val, 4))
        phi_list.append(round(phi_val, 4))
        main_classes.append(m_cls)
        sub_classes.append(s_cls)

    passed_df['respiratory_modulation_R'] = r_list
    passed_df['preferred_phase_rad'] = phi_list
    passed_df['phase_class_main'] = main_classes
    passed_df['phase_class_sub'] = sub_classes

    # Save Classification CSV
    csv_out = out_dir / "respiratory_phase_classification_summary.csv"
    passed_df.to_csv(csv_out, index=False)
    print(f" Saved Phase Classification CSV: {csv_out}")

    # Print Summary Breakdown
    print("\n" + "=" * 90)
    print(" Respiratory Phase Classification Summary Breakdown (Bush et al. 2025 Criteria)")
    print("-" * 90)
    counts_main = passed_df['phase_class_main'].value_counts()
    for k, v in counts_main.items():
        print(f" -> {k:<15}: {v:3d} units ({v/len(passed_df)*100:5.1f}%)")
    print("-" * 90)
    counts_sub = passed_df['phase_class_sub'].value_counts()
    for k, v in counts_sub.items():
        print(f"   * {k:<30}: {v:3d} units ({v/len(passed_df)*100:5.1f}%)")
    print("=" * 90 + "\n")

    # 4. Generate Diagnostic Matplotlib Figure (3 Panels)
    fig_mpl = plt.figure(figsize=(20, 7))

    # Panel A: Pie Chart of Main Categories
    ax1 = fig_mpl.add_subplot(1, 3, 1)
    colors_pie = {'Expiratory': '#1f77b4', 'Inspiratory': '#d62728', 'Tonic': '#7f7f7f'}
    counts_dict = dict(passed_df['phase_class_main'].value_counts())
    labels = list(counts_dict.keys())
    sizes = list(counts_dict.values())
    c_list = [colors_pie.get(l, 'gray') for l in labels]

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=c_list, startangle=140, textprops={'fontweight':'bold', 'fontsize':11})
    ax1.set_title(f"A. VRC Population Classification\n(N = {len(passed_df)} Allen QC Passed Units)", fontweight='bold', fontsize=12)

    # Panel B: Polar Plot of Preferred Phase & Modulation Length R
    ax2 = fig_mpl.add_subplot(1, 3, 2, projection='polar')
    for idx, r in passed_df.iterrows():
        c_p = colors_pie.get(r['phase_class_main'], 'gray')
        ax2.scatter(r['preferred_phase_rad'], r['respiratory_modulation_R'], color=c_p, alpha=0.75, s=40)

    # Add circle for R = 0.20 cutoff
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(theta, [0.20]*100, color='black', linestyle='--', linewidth=1.5, label='Tonic Cutoff (R=0.20)')
    ax2.set_title("B. Preferred Respiratory Phase (rad) vs Vector Strength (R)", fontweight='bold', fontsize=12, pad=15)
    ax2.legend(loc='lower right', fontsize=8)

    # Panel C: Depth Distribution of Classes across Probe
    ax3 = fig_mpl.add_subplot(1, 3, 3)
    for m_cls, color in colors_pie.items():
        sub_df = passed_df[passed_df['phase_class_main'] == m_cls]
        ax3.scatter(sub_df['shank'], sub_df['depth'], color=color, label=f"{m_cls} (n={len(sub_df)})", alpha=0.75, s=50)

    ax3.set_xlabel("Probe Shank (0 to 3)", fontweight='bold', fontsize=10)
    ax3.set_ylabel("Probe Depth (µm)", fontweight='bold', fontsize=10)
    ax3.set_title("C. Anatomical Depth Distribution across Shanks", fontweight='bold', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.4)

    plt.suptitle("Subject TP_Tacr1_1: Respiratory Phase Preference Classification (Bush et al. 2025)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    png_out = out_dir / "respiratory_phase_classification_summary.png"
    plt.savefig(png_out, dpi=300)
    plt.close()

    # Copy PNG to artifact directory
    artifact_png = artifact_dir / "respiratory_phase_classification_summary.png"
    shutil.copy2(png_out, artifact_png)
    print(f" Saved Diagnostic Classification Figure: {png_out}")

    # 5. Interactive Plotly Table & Scatter Dashboard
    fig_plotly = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        specs=[[{"type": "polar"}, {"type": "table"}]],
        subplot_titles=["A. Polar Vector Strength (R) vs Preferred Phase", "B. Single-Unit Classification Table"]
    )

    for m_cls, color in colors_pie.items():
        sub_df = passed_df[passed_df['phase_class_main'] == m_cls]
        fig_plotly.add_trace(go.Scatterpolar(
            r=sub_df['respiratory_modulation_R'],
            theta=sub_df['preferred_phase_rad'] * (180.0 / np.pi),
            mode='markers',
            marker=dict(color=color, size=8),
            text=[f"Unit {u} (S{s}, {d}µm)<br>R: {r} | Subtype: {st}" for u, s, d, r, st in zip(sub_df['cluster_id'], sub_df['shank'], sub_df['depth'], sub_df['respiratory_modulation_R'], sub_df['phase_class_sub'])],
            name=m_cls
        ), row=1, col=1)

    fig_plotly.add_trace(go.Table(
        header=dict(
            values=["<b>Unit ID</b>", "<b>Shank</b>", "<b>Depth (µm)</b>", "<b>FR (Hz)</b>", "<b>Vector Strength (R)</b>", "<b>Main Class</b>", "<b>Subtype</b>"],
            fill_color='#1f77b4',
            font=dict(color='white', size=11),
            align='center'
        ),
        cells=dict(
            values=[
                passed_df['cluster_id'],
                passed_df['shank'],
                passed_df['depth'],
                passed_df['firing_rate_60s'],
                passed_df['respiratory_modulation_R'],
                passed_df['phase_class_main'],
                passed_df['phase_class_sub']
            ],
            fill_color=[['#ffdddd' if m == 'Inspiratory' else '#ddeeff' if m == 'Expiratory' else '#f0f0f0' for m in passed_df['phase_class_main']]],
            align='center',
            font=dict(size=10)
        )
    ), row=1, col=2)

    fig_plotly.update_layout(
        title="<b>Subject TP_Tacr1_1: Respiratory Phase Preference Classification Dashboard</b>",
        template="plotly_white",
        height=750
    )

    html_out = out_dir / "respiratory_phase_classification_summary.html"
    fig_plotly.write_html(str(html_out))
    print(f" Saved Interactive HTML Dashboard: {html_out}")

    print("\n==========================================================================================")
    print(" [SUCCESS] Respiratory Phase Preference Classification Complete!")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
