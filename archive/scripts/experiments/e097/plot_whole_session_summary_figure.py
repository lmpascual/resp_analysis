#!/usr/bin/env python3
"""
Publication-Quality Whole-Session Summary Figure for e097_g2:
- Panel A: Diaphragm EMG breathing motor patterns & integrated envelope
- Panel B: Cycle-by-cycle Inter-Breath Intervals (IBI) with apnea threshold (2.0s)
- Panel C: Instantaneous Heart Rate (ECG) tachogram
- Panel D: Laser stimulation pulses, timeline epochs & surgical milestones from e097.docx
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_JSON = Path("outputs/e097/e097_whole_session_data.json")
OUT_PNG = Path("outputs/e097/e097_whole_session_cardiorespiratory_summary.png")

def plot_summary():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    tl = data["timeline"]
    sc = data["ibi_scatter"]
    trains = data["laser_trains"]
    milestones = data["milestones"]
    meta = data["metadata"]
    
    t_min = np.array(tl["time_min"])
    hr_bpm = np.array(tl["hr_bpm"])
    rr_bpm = np.array(tl["resp_rate_bpm"])
    emg_env = np.array(tl["emg_envelope"])
    laser = np.array(tl["laser_state"])
    
    sc_t_min = np.array(sc["time_min"])
    sc_ibi = np.array(sc["ibi_s"])
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.2, 1.2, 0.6]})
    fig.patch.set_facecolor('#07090e')
    
    for ax in axes:
        ax.set_facecolor('#0a0e17')
        ax.grid(True, color='#ffffff', alpha=0.06, linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            
    # Track 1: Respiration Rate & Diaphragm EMG Envelope
    ax1 = axes[0]
    ax1.plot(t_min, rr_bpm, color='#00f5d4', linewidth=1.2, label='Resp Rate (BPM)')
    ax1.set_ylabel('Resp Rate\n(BPM)', color='#00f5d4', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 320)
    ax1.tick_params(colors='#94a3b8')
    
    # Secondary axis for Integrated EMG Envelope
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_min, emg_env, color='#2dd4bf', linewidth=0.8, alpha=0.5, linestyle=':', label='Int EMG (V)')
    ax1_twin.set_ylabel('Int EMG (V)', color='#2dd4bf', fontsize=10)
    ax1_twin.tick_params(colors='#94a3b8')
    ax1_twin.spines['right'].set_color('#1e293b')
    ax1_twin.set_ylim(0, max(emg_env)*1.2 if len(emg_env)>0 else 1.0)
    
    # Track 2: Inter-Breath Intervals (IBI)
    ax2 = axes[1]
    # Color-code scatter: <1s cyan, 1-2s amber, >=2s red
    colors = np.where(sc_ibi >= 2.0, '#ef4444', np.where(sc_ibi >= 1.0, '#f59e0b', '#38bdf8'))
    ax2.scatter(sc_t_min, sc_ibi, c=colors, s=3, alpha=0.6, rasterized=True, label='Single Breath IBI')
    ax2.plot(t_min, tl["ibi_s"], color='#f59e0b', linewidth=1.2, label='Mean IBI (s)')
    ax2.axhline(2.0, color='#ef4444', linestyle='--', linewidth=1.2, alpha=0.8, label='Apnea Threshold (2.0s)')
    ax2.set_ylabel('Inter-Breath\nInterval (s)', color='#f59e0b', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 6.0)
    ax2.tick_params(colors='#94a3b8')
    
    # Track 3: Instantaneous Heart Rate (ECG)
    ax3 = axes[2]
    ax3.plot(t_min, hr_bpm, color='#ff0055', linewidth=1.2, label='Heart Rate (BPM)')
    ax3.set_ylabel('Heart Rate\n(BPM)', color='#ff0055', fontsize=11, fontweight='bold')
    ax3.set_ylim(50, 750)
    ax3.tick_params(colors='#94a3b8')
    
    # Track 4: Laser Stimulation & Milestone Shading
    ax4 = axes[3]
    ax4.plot(t_min, laser, color='#a855f7', linewidth=1.2)
    ax4.fill_between(t_min, 0, laser, color='#a855f7', alpha=0.3)
    ax4.set_ylabel('Laser (V)', color='#a855f7', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 1.2)
    ax4.set_xlabel('Recording Time (Minutes)', color='#e2e8f0', fontsize=12, fontweight='bold')
    ax4.tick_params(colors='#94a3b8')
    ax4.set_xlim(0, max(t_min))
    
    # Add vertical shading for Laser Trains across all axes
    for tr in trains:
        t0_m = tr["t_start"] / 60.0
        t1_m = (tr["t_start"] + tr["dur"]) / 60.0
        for ax in axes:
            ax.axvspan(t0_m, t1_m, color='#a855f7', alpha=0.25, zorder=0)
            
    # Add milestone annotations at the top of Track 1
    key_milestones = [
        (856.8 / 60.0, "Laser 1 (14:16)", "#c084fc"),
        (1053.2 / 60.0, "Laser 4 (17:33)", "#c084fc"),
        (1800.0 / 60.0, "Anoxia 1 (30:00)", "#38bdf8"),
        (2310.0 / 60.0, "Anoxia 2 (38:30)", "#38bdf8"),
        (2760.0 / 60.0, "DR 1 (46:00)", "#34d399"),
        (3330.0 / 60.0, "DR 2 (55:30)", "#34d399"),
        (3720.5 / 60.0, "Laser 9 (62:00)", "#c084fc"),
        (3990.0 / 60.0, "KA Inj (66:30)", "#f87171"),
        (4603.0 / 60.0, "Tonicity (76:43)", "#ef4444"),
        (5400.0 / 60.0, "Seizure (90:00)", "#ef4444"),
        (6480.0 / 60.0, "Terminal (108:00)", "#94a3b8"),
    ]
    
    for t_m, label, col in key_milestones:
        ax1.axvline(t_m, color=col, linestyle=':', alpha=0.6, linewidth=1.0)
        ax1.text(t_m, 305, label, rotation=90, va='top', ha='right', color=col, fontsize=8, fontweight='bold', alpha=0.9)

    plt.suptitle(
        f"e097_g2 Whole-Session Cardiorespiratory Profile (110.1 min) | Animal: {meta['animal_id']} ({meta['strain']}, {meta['sex']}, {meta['weight_g']}g)\n"
        f"Baseline HR: {meta['baseline_hr_bpm']} BPM | Baseline Resp: {meta['baseline_rr_bpm']} BPM ({meta['baseline_ibi_s']}s IBI) | 9 Laser Opto Trains | Surgeon: {meta['surgeon']} ({meta['date']})",
        fontsize=13, fontweight='bold', color='#f8fafc', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Summary figure saved: {OUT_PNG}")

if __name__ == "__main__":
    plot_summary()
