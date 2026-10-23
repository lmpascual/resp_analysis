"""
Batch Render Laser 1 and Laser 2 Eupnea-to-Gasping Transition Cards and Compute 3-Train Statistics Table
Animal: e097 (226s Continuous Recording, Motion-Corrected Consensus Master)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch

from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.processors.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=== Starting Laser 1 & Laser 2 Transition Suite Generation ===", flush=True)

# Output Directories
base_out = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "04_gasping_transition_rasters"
cards_dir = base_out / "cards"
l1_dir = cards_dir / "laser1"
l2_dir = cards_dir / "laser2"
l3_dir = cards_dir / "laser3"
l1_dir.mkdir(exist_ok=True, parents=True)
l2_dir.mkdir(exist_ok=True, parents=True)
l3_dir.mkdir(exist_ok=True, parents=True)

# 1. Load NIDAQ Telemetry
print("[1/5] Loading NIDAQ telemetry...", flush=True)
nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
fs = 10000.0; n_ch = 9; start_t = 796.827; dur = 226.243
raw = np.memmap(nidq_file, dtype="int16", mode="r")
slice_data = raw[int(start_t*fs)*n_ch : int((start_t+dur)*fs)*n_ch].reshape(-1, n_ch)
emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
t_vec = np.arange(len(emg_v)) / fs

# 2. Extract Breaths
print("[2/5] Detecting breath onsets, offsets, and flow zero-crossings...", flush=True)
emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
emg_proc = process_emg(emg_v, t_vec, emg_cfg)
valid_pairs = [(on, off, off - on) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.2 and on < t_vec[-1] - 0.3 and off > on]
valid_onsets = np.array([p[0] for p in valid_pairs])
valid_offsets = np.array([p[1] for p in valid_pairs])
flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
flow_zcs = flow_res.flow_zero_crossings

# 3. Load Spikes and Summary
cons_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
df_summary = pd.read_csv(cons_dir / "03_spiking_statistics_literature_cards_neg50_pos100" / "spiking_statistics_lit_summary_neg50_pos100.csv")
unique_cids = sorted(list(np.unique(spk_clusters)))
print(f"Loaded {len(unique_cids)} units to process across Laser 1 and Laser 2.", flush=True)

# Train Definitions
train_configs = {
    "laser1": {
        "name": "Laser 1",
        "t_laser_on": 60.0,
        "laser_dur": 2.0,
        "tr_start": 185,
        "tr_end": 225,
        "laser_tr_range": (198, 200),
        "gasp_threshold": 2.8,
        "apnea_threshold": 0.25,
        "out_dir": l1_dir
    },
    "laser2": {
        "name": "Laser 2",
        "t_laser_on": 120.0,
        "laser_dur": 2.0,
        "tr_start": 405,
        "tr_end": 445,
        "laser_tr_range": (418, 420),
        "gasp_threshold": 3.0,
        "apnea_threshold": 0.25,
        "out_dir": l2_dir
    }
}

t_pre_ms = -100.0
t_post_ms = 200.0
win_dur_s = (t_post_ms - t_pre_ms) / 1000.0  # 0.300s
dt_peth = 0.002
peth_bins = np.arange(t_pre_ms / 1000.0, t_post_ms / 1000.0 + dt_peth, dt_peth)
peth_centers = (peth_bins[:-1] + peth_bins[1:]) / 2.0 * 1000.0
eps = 1e-5

# Precompute telemetry and trial info for each train
train_data = {}
for train_key, cfg in train_configs.items():
    sub_trs = list(range(cfg["tr_start"], cfg["tr_end"] + 1))
    t_win_start = valid_onsets[cfg["tr_start"]] - 1.0
    t_win_end = valid_onsets[cfg["tr_end"]] + 1.0
    idx_t0 = int(max(0, t_win_start * fs))
    idx_t1 = int(min(len(t_vec), t_win_end * fs))
    sub_t = t_vec[idx_t0:idx_t1]
    sub_emg = emg_v[idx_t0:idx_t1]
    sub_flow = flow_v[idx_t0:idx_t1]

    w_rms = int(0.04 * fs)
    sub_emg_sq = sub_emg ** 2
    sub_emg_rms = np.sqrt(np.convolve(sub_emg_sq, np.ones(w_rms)/w_rms, mode='same'))

    trial_info = []
    l_min, l_max = cfg["laser_tr_range"]
    for i_row, tr in enumerate(sub_trs):
        t_on = valid_onsets[tr]
        t_zc = flow_zcs[tr]
        t_off = valid_offsets[tr]
        dur_i = (t_off - t_on) * 1000.0
        flow_chunk = flow_v[int((t_on - 0.05)*fs):int((t_on + 0.25)*fs)]
        flow_max = float(np.max(np.abs(flow_chunk))) if len(flow_chunk) > 0 else 0.0

        if tr >= l_min and tr <= l_max:
            state = "LASER"
            state_color = "#ffedd5"
            badge_color = "#ea580c"
        elif flow_max > cfg["gasp_threshold"]:
            state = "GASP"
            state_color = "#ffe4e6"
            badge_color = "#e11d48"
        elif flow_max < cfg["apnea_threshold"]:
            state = "APNEA_PAUSE"
            state_color = "#f1f5f9"
            badge_color = "#64748b"
        elif tr < l_min:
            state = "EUPNEA_PRE"
            state_color = "#f0f9ff"
            badge_color = "#0284c7"
        else:
            state = "EUPNEA_POST"
            state_color = "#f0fdf4"
            badge_color = "#0d9488"

        trial_info.append({
            "row": i_row,
            "tr": tr,
            "t_on": t_on,
            "t_zc": t_zc,
            "t_off": t_off,
            "dur_i": dur_i,
            "flow_max": flow_max,
            "state": state,
            "state_color": state_color,
            "badge_color": badge_color
        })

    train_data[train_key] = {
        "sub_trs": sub_trs,
        "n_sub": len(sub_trs),
        "t_win_start": t_win_start,
        "t_win_end": t_win_end,
        "sub_t": sub_t,
        "sub_emg_rms": sub_emg_rms,
        "sub_flow": sub_flow,
        "trial_info": trial_info
    }

# Render Loop
for train_key, cfg in train_configs.items():
    print(f"\n[3/5] Rendering 497 cards for {cfg['name']}...", flush=True)
    td = train_data[train_key]
    n_sub = td["n_sub"]
    sub_t = td["sub_t"]
    sub_emg_rms = td["sub_emg_rms"]
    sub_flow = td["sub_flow"]
    trial_info = td["trial_info"]
    out_target_dir = cfg["out_dir"]

    for idx_u, cid in enumerate(unique_cids):
        c_spks = spk_times[spk_clusters == cid]
        match = df_summary[df_summary["cluster_id"] == cid]
        if len(match) > 0:
            row_m = match.iloc[0]
            depth = float(row_m["depth_um"])
            cell_type = str(row_m["cell_type"])
            is_qc = bool(row_m["is_qc_passed"])
        else:
            depth = 0.0
            cell_type = "Unclassified"
            is_qc = False

        spikes_per_breath = []
        flow_max_per_breath = []
        peth_eupnea, peth_laser, peth_gasp, peth_rec = [], [], [], []

        for ti in trial_info:
            t_zc = ti["t_zc"]
            spks_win = c_spks[(c_spks >= t_zc + t_pre_ms / 1000.0) & (c_spks <= t_zc + t_post_ms / 1000.0)]
            spikes_per_breath.append(len(spks_win))
            flow_max_per_breath.append(ti["flow_max"])

            counts, _ = np.histogram(spks_win - t_zc, bins=peth_bins)
            rate_curve = counts / dt_peth
            if ti["state"] == "EUPNEA_PRE":
                peth_eupnea.append(rate_curve)
            elif ti["state"] == "LASER":
                peth_laser.append(rate_curve)
            elif ti["state"] == "GASP":
                peth_gasp.append(rate_curve)
            elif ti["state"] == "EUPNEA_POST":
                peth_rec.append(rate_curve)

        spk_pre = [spikes_per_breath[ti["row"]] for ti in trial_info if ti["state"] == "EUPNEA_PRE"]
        spk_las = [spikes_per_breath[ti["row"]] for ti in trial_info if ti["state"] == "LASER"]
        spk_gasp = [spikes_per_breath[ti["row"]] for ti in trial_info if ti["state"] == "GASP"]
        spk_post = [spikes_per_breath[ti["row"]] for ti in trial_info if ti["state"] == "EUPNEA_POST"]

        fr_pre = (np.mean(spk_pre) / win_dur_s) if len(spk_pre) > 0 else 0.0
        fr_las = (np.mean(spk_las) / win_dur_s) if len(spk_las) > 0 else 0.0
        fr_gasp = (np.mean(spk_gasp) / win_dur_s) if len(spk_gasp) > 0 else 0.0
        fr_post = (np.mean(spk_post) / win_dur_s) if len(spk_post) > 0 else 0.0

        gmi = float((fr_gasp - fr_pre) / (fr_gasp + fr_pre + eps))
        lmi = float((fr_las - fr_pre) / (fr_las + fr_pre + eps))
        rec_idx = float((fr_post - fr_pre) / (fr_post + fr_pre + eps))

        # Render Figure
        fig = plt.figure(figsize=(18, 13), dpi=180)
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 2.2, 1.2], hspace=0.35, wspace=0.28)

        # Row 0: Telemetry
        ax_emg = fig.add_subplot(gs[0, :])
        ax_flow = ax_emg.twinx()
        t_rel = sub_t - cfg["t_laser_on"]
        ax_emg.plot(t_rel, sub_emg_rms * 1000.0, color="#2563eb", linewidth=1.2, label="Diaphragm EMG RMS (mV)", zorder=3)
        ax_flow.plot(t_rel, sub_flow, color="#0f766e", linewidth=1.2, label="Flow Meter (V)", zorder=4)
        ax_emg.fill_between([0.0, cfg["laser_dur"]], -50.0, 5000.0, color="#f97316", alpha=0.30, label=f"{cfg['name']} Stimulation ({cfg['laser_dur']}s)", zorder=1)

        gasp_y_elevations = [5.3, 4.7, 5.3, 4.7]
        gasp_cnt = 0
        for ti in trial_info:
            if ti["state"] == "GASP":
                t_gasp_rel = ti["t_on"] - cfg["t_laser_on"]
                ax_flow.axvline(t_gasp_rel, color="#e11d48", linestyle="--", linewidth=1.5, alpha=0.85, zorder=5)
                y_pos = gasp_y_elevations[gasp_cnt % len(gasp_y_elevations)]
                ax_flow.text(t_gasp_rel, y_pos, f"Gasp ({ti['flow_max']:.1f}V)", color="#e11d48", fontsize=8, fontweight="bold", ha="center")
                gasp_cnt += 1

        ax_emg.set_xlim(t_rel[0], t_rel[-1])
        ax_emg.set_xlabel(f"Time from {cfg['name']} Onset (s) [t = {cfg['t_laser_on']:.1f}s]", fontsize=9, fontweight="bold")
        ax_emg.set_ylabel("Diaphragm EMG (mV)", color="#2563eb", fontsize=9, fontweight="bold")
        ax_flow.set_ylabel("Flow Meter (V)", color="#0f766e", fontsize=9, fontweight="bold")
        ax_flow.set_ylim(-1.5, 6.0)
        ax_emg.set_title(f"Continuous Physiological Telemetry & Laser Arrest — Animal e097 [{cfg['name']}] (Window: {td['t_win_start']:.1f}s to {td['t_win_end']:.1f}s)", fontsize=10, fontweight="bold")
        lines1, labels1 = ax_emg.get_legend_handles_labels()
        lines2, labels2 = ax_flow.get_legend_handles_labels()
        ax_emg.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, framealpha=0.9)
        ax_emg.grid(True, linestyle=":", alpha=0.5)

        # Row 1: Raster & Flow Bars
        ax_ras = fig.add_subplot(gs[1, :2])
        ax_flow_bar = fig.add_subplot(gs[1, 2], sharey=ax_ras)

        for ti in trial_info:
            row_y = ti["row"] + 1
            t_zc = ti["t_zc"]
            t_on = ti["t_on"]
            ax_ras.fill_between([t_pre_ms, t_post_ms], [row_y - 0.5, row_y - 0.5], [row_y + 0.5, row_y + 0.5], color=ti["state_color"], zorder=1)
            t_ion_rel_ms = (t_on - t_zc) * 1000.0
            ax_ras.plot(t_ion_rel_ms, row_y, marker='<', color="#0284c7", markersize=4, zorder=4)

            spks_win = c_spks[(c_spks >= t_zc + t_pre_ms / 1000.0) & (c_spks <= t_zc + t_post_ms / 1000.0)]
            spks_rel_ms = (spks_win - t_zc) * 1000.0
            if len(spks_rel_ms) > 0:
                ax_ras.scatter(spks_rel_ms, [row_y]*len(spks_rel_ms), marker='|', color="#0f172a", s=28, linewidth=1.2, zorder=5)

            bar_col = ti["badge_color"]
            ax_flow_bar.barh(row_y, ti["flow_max"], height=0.75, color=bar_col, alpha=0.85, zorder=3)
            if ti["state"] == "GASP":
                ax_flow_bar.text(ti["flow_max"] + 0.1, row_y, f"Gasp ({ti['flow_max']:.1f}V)", color="#e11d48", fontsize=7.5, fontweight="bold", va="center")

        ax_ras.axvline(0, color="#475569", linestyle="--", linewidth=1.4, zorder=6)
        ax_ras.set_xlim(t_pre_ms, t_post_ms)
        ax_ras.set_ylim(0.5, n_sub + 0.5)
        ax_ras.set_yticks(np.arange(1, n_sub + 1, 3))
        ax_ras.set_yticklabels([f"#{trial_info[i]['tr']} ({trial_info[i]['t_on']:.1f}s)" for i in range(0, n_sub, 3)], fontsize=8)
        ax_ras.set_xlabel("Time from Flow Inspiration Offset (Ioff / Zero-Crossing) [ms]", fontsize=9, fontweight="bold")
        ax_ras.set_ylabel("Breath Trial (Chronological Order)", fontsize=9, fontweight="bold")
        ax_ras.set_title(f"Zoomed-In Chronological Breath Raster across {cfg['name']} Transition", fontsize=10, fontweight="bold")
        ax_ras.grid(True, linestyle=":", alpha=0.3)

        ax_flow_bar.set_xlim(0, 6.0)
        ax_flow_bar.set_xlabel("Peak Tidal Flow (V)", fontsize=9, fontweight="bold")
        ax_flow_bar.set_title("Tidal Flow Amplitude", fontsize=10, fontweight="bold")
        ax_flow_bar.grid(True, linestyle=":", alpha=0.3)
        plt.setp(ax_flow_bar.get_yticklabels(), visible=False)

        # Row 2: Analysis Panels
        ax_peth = fig.add_subplot(gs[2, 0])
        k_smooth = np.ones(5) / 5.0
        if len(peth_eupnea) > 0:
            mean_eup = np.convolve(np.mean(peth_eupnea, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_eup, color="#0284c7", linewidth=2.0, label=f"Pre Eupnea (n={len(peth_eupnea)})")
        if len(peth_gasp) > 0:
            mean_gasp = np.convolve(np.mean(peth_gasp, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_gasp, color="#e11d48", linewidth=2.2, label=f"Ballistic Gasp (n={len(peth_gasp)})")
        if len(peth_rec) > 0:
            mean_rec = np.convolve(np.mean(peth_rec, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_rec, color="#0d9488", linewidth=1.8, linestyle="--", label=f"Post Eupnea (n={len(peth_rec)})")

        ax_peth.axvline(0, color="#475569", linestyle="--", linewidth=1.0)
        ax_peth.set_xlabel("Time from Flow Ioff (ms)", fontsize=8.5, fontweight="bold")
        ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=8.5, fontweight="bold")
        ax_peth.set_title(f"PETH: Eupnea vs. Gasp [{cfg['name']}]", fontsize=9.5, fontweight="bold")
        ax_peth.legend(loc="upper right", fontsize=7.5)
        ax_peth.grid(True, linestyle=":", alpha=0.4)

        ax_dyn = fig.add_subplot(gs[2, 1])
        ax_dyn_twin = ax_dyn.twinx()
        ax_dyn.plot(np.arange(1, n_sub + 1), spikes_per_breath, color="#0f172a", marker='o', markersize=4, linewidth=1.5, label="Spikes / Breath")
        ax_dyn_twin.plot(np.arange(1, n_sub + 1), flow_max_per_breath, color="#0f766e", marker='^', markersize=3.5, linestyle=":", label="Peak Flow (V)")
        ax_dyn.set_xlabel("Transition Trial Index", fontsize=8.5, fontweight="bold")
        ax_dyn.set_ylabel("Spike Count", color="#0f172a", fontsize=8.5, fontweight="bold")
        ax_dyn_twin.set_ylabel("Peak Flow (V)", color="#0f766e", fontsize=8.5, fontweight="bold")
        ax_dyn.set_title("Spikes per Breath vs. Flow Amplitude", fontsize=9.5, fontweight="bold")
        ax_dyn.grid(True, linestyle=":", alpha=0.4)

        ax_cross = fig.add_subplot(gs[2, 2])
        gasp_spk_l1 = np.mean([np.sum((c_spks >= flow_zcs[tr] - 0.05) & (c_spks <= flow_zcs[tr] + 0.10)) for tr in [204]])
        gasp_spk_l2 = np.mean([np.sum((c_spks >= flow_zcs[tr] - 0.05) & (c_spks <= flow_zcs[tr] + 0.10)) for tr in [421, 422, 423]])
        gasp_spk_l3 = np.mean([np.sum((c_spks >= flow_zcs[tr] - 0.05) & (c_spks <= flow_zcs[tr] + 0.10)) for tr in [589, 591, 592]])

        eup_spk_l1 = np.mean([np.sum((c_spks >= flow_zcs[tr] - 0.05) & (c_spks <= flow_zcs[tr] + 0.10)) for tr in range(180, 195)])
        eup_spk_l2 = np.mean([np.sum((c_spks >= flow_zcs[tr] - 0.05) & (c_spks <= flow_zcs[tr] + 0.10)) for tr in range(405, 417)])
        eup_spk_l3 = np.mean([np.sum((c_spks >= flow_zcs[tr] - 0.05) & (c_spks <= flow_zcs[tr] + 0.10)) for tr in range(575, 586)])

        x_bars = np.arange(3)
        w_bar = 0.35
        ax_cross.bar(x_bars - w_bar/2, [eup_spk_l1, eup_spk_l2, eup_spk_l3], width=w_bar, color="#0284c7", alpha=0.85, label="Pre Eupnea")
        ax_cross.bar(x_bars + w_bar/2, [gasp_spk_l1, gasp_spk_l2, gasp_spk_l3], width=w_bar, color="#e11d48", alpha=0.85, label="Ballistic Gasp")
        ax_cross.set_xticks(x_bars)
        ax_cross.set_xticklabels(["Laser 1", "Laser 2", "Laser 3"], fontsize=8.5, fontweight="bold")
        ax_cross.set_ylabel("Spikes / Breath Window", fontsize=8.5, fontweight="bold")
        ax_cross.set_title("Gasp Spiking: Laser 1 vs 2 vs 3", fontsize=9.5, fontweight="bold")
        ax_cross.legend(loc="upper right", fontsize=7.5)
        ax_cross.grid(True, linestyle=":", alpha=0.4)

        qc_badge = "QC PASSED" if is_qc else "Non-QC"
        plt.suptitle(
            f"Cluster #{cid:03d} (Depth: {depth:.0f} µm | {cell_type} [{qc_badge}]) — {cfg['name']} Gasping Transition Suite\n"
            f"[GMI: {gmi:+.2f} | Laser MI: {lmi:+.2f} | Recovery Index: {rec_idx:+.2f} | Pre: {fr_pre:.1f}Hz, Gasp: {fr_gasp:.1f}Hz, Post: {fr_post:.1f}Hz]",
            fontsize=12, fontweight="bold", y=0.98
        )

        out_f = out_target_dir / f"gasp_transition_cluster_{cid:03d}.png"
        fig.savefig(out_f)
        plt.close(fig)

        if (idx_u + 1) % 50 == 0 or idx_u == len(unique_cids) - 1:
            print(f"      Rendered {idx_u + 1}/{len(unique_cids)} units for {cfg['name']}...", flush=True)

print("\n=== Laser 1 and Laser 2 Transition Cards Complete! ===", flush=True)
