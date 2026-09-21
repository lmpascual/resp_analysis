#!/usr/bin/env python3
"""
Generate 3-Train Zoomed-In Eupnea-to-Gasping Transition Literature Diagnostic Cards
[-300ms to +300ms Aligned to Diaphragm Inspiration Offset (I_off)]
Captures in-laser breakthrough breaths and oral autoresuscitation gasps across Laser 1, Laser 2, and Laser 3.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(r"C:\Users\lpascu\analysis_src")
sys.path.insert(0, str(PROJECT_ROOT))
import src.pipeline.compat_patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.pipeline.processors.proc_diaphragm_transition import (
    extract_diaphragm_envelope,
    segment_diaphragm_breaths,
    build_train_transition_data
)

# Configuration
t_pre_ms = -300.0
t_post_ms = 300.0
win_dur_s = (t_post_ms - t_pre_ms) / 1000.0  # 0.600s
dt_peth = 0.002
peth_bins = np.arange(t_pre_ms / 1000.0, t_post_ms / 1000.0 + dt_peth, dt_peth)
peth_centers = (peth_bins[:-1] + peth_bins[1:]) / 2.0 * 1000.0
eps = 1e-5

# Directories
base_out = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master" / "04_gasping_transition_rasters"
cards_dir = base_out / "cards"
l1_dir = cards_dir / "laser1"
l2_dir = cards_dir / "laser2"
l3_dir = cards_dir / "laser3"
l1_dir.mkdir(exist_ok=True, parents=True)
l2_dir.mkdir(exist_ok=True, parents=True)
l3_dir.mkdir(exist_ok=True, parents=True)


def render_single_unit(args):
    """
    Worker function to render the 3-train transition diagnostic card for a single unit.
    """
    cid, spks, shared_train_data, out_paths = args

    for train_key in ["laser1", "laser2", "laser3"]:
        td = shared_train_data[train_key]
        out_path = out_paths[train_key]

        fig = plt.figure(figsize=(15, 11), dpi=130)
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 2.5, 1.4], hspace=0.36, wspace=0.28)

        # Row 0: Telemetry
        ax_emg = fig.add_subplot(gs[0, :])
        ax_flow = ax_emg.twinx()

        t_rel = td["sub_t_ds"] - td["t_laser_on"]
        ax_emg.plot(t_rel, td["sub_emg_ds"], color="#93c5fd", alpha=0.45, linewidth=0.6, label="Raw Diaphragm EMG")
        ax_emg.plot(t_rel, td["sub_emg_rms_ds"] * 2.5, color="#1d4ed8", linewidth=1.3, label="EMG Envelope (RMS)")
        ax_flow.plot(t_rel, td["sub_flow_ds"], color="#0f766e", linewidth=1.4, label="Flow Meter (Tidal Vol / Airflow)")

        ax_emg.axvspan(0.0, td["laser_dur"], color="#fef08a", alpha=0.45, label=f"Laser Train ({td['laser_dur']:.1f}s Arrest)")
        ax_emg.axvline(0.0, color="#ca8a04", linestyle="--", linewidth=1.5)

        gasp_count = 0
        for tr in td["trials"]:
            rel_t = tr["t_peak"] - td["t_laser_on"]
            if tr["state"] == "DIAPHRAGM_GASP":
                gasp_count += 1
                y_txt_offset = 0.25 if (gasp_count % 2 == 1) else 0.65
                ax_emg.axvline(rel_t, color="#e11d48", linestyle=":", linewidth=1.3)
                ax_emg.plot(rel_t, tr["emg_peak_v"] * 2.5, marker='*', color="#e11d48", markersize=11, zorder=5)
                lbl = f"Gasp #{tr['tr_id']}" + (" (Oral)" if tr["is_oral"] else "")
                ax_emg.text(rel_t, tr["emg_peak_v"] * 2.5 + y_txt_offset, lbl,
                            color="#be123c", fontsize=7.8, fontweight="bold", ha="center")
            elif tr["state"] == "LASER_BREAKTHROUGH":
                ax_emg.plot(rel_t, tr["emg_peak_v"] * 2.5, marker='^', color="#ea580c", markersize=6, zorder=5)

        ax_emg.set_xlim(t_rel[0], t_rel[-1])
        ax_emg.set_xlabel(f"Time from {td['train_name']} Onset (s) [t = {td['t_laser_on']:.1f}s]", fontsize=9, fontweight="bold")
        ax_emg.set_ylabel("Diaphragm EMG (mV)", color="#2563eb", fontsize=9, fontweight="bold")
        ax_flow.set_ylabel("Flow Meter (V)", color="#0f766e", fontsize=9, fontweight="bold")
        ax_flow.set_ylim(-2.0, 6.0)
        ax_emg.set_ylim(-3.5, 3.8)
        ax_emg.set_title(f"Continuous Physiological Telemetry & Diaphragm Gasp Events — Animal e097 [{td['train_name']}] (Window: {td['t_win_start']:.1f}s to {td['t_win_end']:.1f}s)", fontsize=10, fontweight="bold")
        lines1, labels1 = ax_emg.get_legend_handles_labels()
        lines2, labels2 = ax_flow.get_legend_handles_labels()
        ax_emg.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, framealpha=0.9)
        ax_emg.grid(True, linestyle=":", alpha=0.5)

        # Row 1: Raster & Amplitude Bars
        ax_ras = fig.add_subplot(gs[1, :2])
        ax_bar = fig.add_subplot(gs[1, 2], sharey=ax_ras)

        peth_eupnea = []
        peth_breakthrough = []
        peth_gasp = []
        peth_rec = []
        spikes_per_breath = []
        emg_peak_per_breath = []
        n_sub = len(td["trials"])

        ax_ras.axvspan(-300.0, -150.0, color="#f3e8ff", alpha=0.45, label="Pre-Inspiratory Zone (-300 to -150 ms)", zorder=0)

        for tr in td["trials"]:
            row_y = tr["row_idx"] + 1
            t_off = tr["t_off"]
            t_on = tr["t_on"]
            ax_ras.fill_between([t_pre_ms, t_post_ms], [row_y - 0.5, row_y - 0.5], [row_y + 0.5, row_y + 0.5], color=tr["state_color"], zorder=1)

            # Diaphragm inspiratory onset marker relative to I_off
            t_ion_rel_ms = (t_on - t_off) * 1000.0
            ax_ras.plot(t_ion_rel_ms, row_y, marker='<', color="#0284c7", markersize=4.5, zorder=4)

            spks_win = spks[(spks >= t_off + t_pre_ms / 1000.0) & (spks <= t_off + t_post_ms / 1000.0)]
            spks_rel_ms = (spks_win - t_off) * 1000.0
            if len(spks_rel_ms) > 0:
                ax_ras.scatter(spks_rel_ms, [row_y]*len(spks_rel_ms), marker='|', color="#0f172a", s=28, linewidth=1.2, zorder=5)

            bar_col = tr["badge_color"]
            bar_val = tr["emg_peak_v"] * 10.0
            ax_bar.barh(row_y, bar_val, height=0.75, color=bar_col, alpha=0.85, zorder=3)
            if tr["state"] == "DIAPHRAGM_GASP":
                lbl = f"Oral Gasp ({tr['emg_peak_v']:.2f}V)" if tr["is_oral"] else f"Gasp ({tr['emg_peak_v']:.2f}V)"
                ax_bar.text(bar_val + 0.15, row_y, lbl, color="#e11d48", fontsize=7.0, fontweight="bold", va="center")
            elif tr["state"] == "LASER_BREAKTHROUGH":
                ax_bar.text(bar_val + 0.15, row_y, "Breakthrough", color="#f97316", fontsize=6.8, va="center")

            counts, _ = np.histogram(spks_win - t_off, bins=peth_bins)
            rate_hz = counts / dt_peth

            if tr["state"] == "PRE_EUPNEA":
                peth_eupnea.append(rate_hz)
            elif tr["state"] == "LASER_BREAKTHROUGH":
                peth_breakthrough.append(rate_hz)
            elif tr["state"] == "DIAPHRAGM_GASP":
                peth_gasp.append(rate_hz)
            elif tr["state"] == "POST_EUPNEA":
                peth_rec.append(rate_hz)

            spikes_per_breath.append(len(spks_win))
            emg_peak_per_breath.append(tr["emg_peak_v"])

        ax_ras.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5, zorder=6, label="Diaphragm Ioff (Burst Termination)")
        ax_ras.set_xlim(t_pre_ms, t_post_ms)
        ax_ras.set_ylim(0.5, n_sub + 0.5)
        ax_ras.set_yticks(np.arange(1, n_sub + 1, 3))
        ax_ras.set_yticklabels([f"#{td['trials'][i]['tr_id']} ({td['trials'][i]['t_on']:.1f}s)" for i in range(0, n_sub, 3)], fontsize=8)
        ax_ras.set_xlabel("Time from Diaphragm Inspiration Offset (Ioff) [ms] — Expanded [-300ms, +300ms]", fontsize=9, fontweight="bold")
        ax_ras.set_ylabel("Breath Trial (Chronological Order)", fontsize=9, fontweight="bold")
        ax_ras.set_title(f"Zoomed-In Chronological Breath Raster across {td['train_name']} Transition [-300ms to +300ms]", fontsize=10, fontweight="bold")
        ax_ras.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
        ax_ras.grid(True, linestyle=":", alpha=0.3)

        ax_bar.set_xlim(0, 8.0)
        ax_bar.set_xlabel("Diaphragm Burst Amplitude (x10 V)", fontsize=9, fontweight="bold")
        ax_bar.set_title("Inspiratory Motor Drive", fontsize=10, fontweight="bold")
        ax_bar.grid(True, linestyle=":", alpha=0.3)
        plt.setp(ax_bar.get_yticklabels(), visible=False)

        # Row 2: Analysis Panels
        # Panel 0: PETH
        ax_peth = fig.add_subplot(gs[2, 0])
        k_smooth = np.ones(5) / 5.0
        if len(peth_eupnea) > 0:
            mean_eup = np.convolve(np.mean(peth_eupnea, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_eup, color="#0284c7", linewidth=2.0, label=f"Pre Eupnea (n={len(peth_eupnea)})")
        if len(peth_breakthrough) > 0:
            mean_bt = np.convolve(np.mean(peth_breakthrough, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_bt, color="#f97316", linewidth=1.8, linestyle=":", label=f"Laser Breakthrough (n={len(peth_breakthrough)})")
        if len(peth_gasp) > 0:
            mean_gasp = np.convolve(np.mean(peth_gasp, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_gasp, color="#e11d48", linewidth=2.2, label=f"Diaphragm Gasp (n={len(peth_gasp)})")
        if len(peth_rec) > 0:
            mean_rec = np.convolve(np.mean(peth_rec, axis=0), k_smooth, mode='same')
            ax_peth.plot(peth_centers, mean_rec, color="#0d9488", linewidth=1.8, linestyle="--", label=f"Post Eupnea (n={len(peth_rec)})")

        ax_peth.axvspan(-300.0, -150.0, color="#f3e8ff", alpha=0.45, label="Pre-I Zone")
        ax_peth.axvline(0, color="#dc2626", linestyle="--", linewidth=1.2)
        ax_peth.set_xlabel("Time from Diaphragm Ioff (ms)", fontsize=8.5, fontweight="bold")
        ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=8.5, fontweight="bold")
        ax_peth.set_title(f"PETH: Diaphragm Burst Dynamic [{td['train_name']}]", fontsize=9.5, fontweight="bold")
        ax_peth.legend(loc="upper right", fontsize=7.0)
        ax_peth.grid(True, linestyle=":", alpha=0.4)

        # Panel 1: Dynamics
        ax_dyn = fig.add_subplot(gs[2, 1])
        ax_dyn_twin = ax_dyn.twinx()
        ax_dyn.plot(np.arange(1, n_sub + 1), spikes_per_breath, color="#0f172a", marker='o', markersize=4, linewidth=1.5, label="Spikes / Breath")
        ax_dyn_twin.plot(np.arange(1, n_sub + 1), emg_peak_per_breath, color="#e11d48", marker='s', markersize=3.5, linestyle="--", label="Diaphragm RMS (V)")
        ax_dyn.set_xlabel("Transition Trial Index", fontsize=8.5, fontweight="bold")
        ax_dyn.set_ylabel("Spike Count (600ms Window)", color="#0f172a", fontsize=8.5, fontweight="bold")
        ax_dyn_twin.set_ylabel("Diaphragm Peak RMS (V)", color="#e11d48", fontsize=8.5, fontweight="bold")
        ax_dyn.set_title("Spikes per Breath vs. Diaphragm Drive", fontsize=9.5, fontweight="bold")
        ax_dyn.grid(True, linestyle=":", alpha=0.4)

        # Panel 2: Cross-Train Gasp Spiking
        ax_cross = fig.add_subplot(gs[2, 2])
        def get_train_spks(td_target, state_target):
            trs = [tr for tr in td_target["trials"] if tr["state"] == state_target]
            if len(trs) == 0:
                return 0.0
            return np.mean([np.sum((spks >= tr["t_off"] - 0.30) & (spks <= tr["t_off"] + 0.30)) for tr in trs])

        eup_l1 = get_train_spks(shared_train_data["laser1"], "PRE_EUPNEA")
        eup_l2 = get_train_spks(shared_train_data["laser2"], "PRE_EUPNEA")
        eup_l3 = get_train_spks(shared_train_data["laser3"], "PRE_EUPNEA")

        gasp_l1 = get_train_spks(shared_train_data["laser1"], "DIAPHRAGM_GASP")
        gasp_l2 = get_train_spks(shared_train_data["laser2"], "DIAPHRAGM_GASP")
        gasp_l3 = get_train_spks(shared_train_data["laser3"], "DIAPHRAGM_GASP")

        x_bars = np.arange(3)
        w_bar = 0.35
        ax_cross.bar(x_bars - w_bar/2, [eup_l1, eup_l2, eup_l3], width=w_bar, color="#0284c7", alpha=0.85, label="Pre Eupnea")
        ax_cross.bar(x_bars + w_bar/2, [gasp_l1, gasp_l2, gasp_l3], width=w_bar, color="#e11d48", alpha=0.85, label="Diaphragm Gasp")
        ax_cross.set_xticks(x_bars)
        ax_cross.set_xticklabels(["Laser 1", "Laser 2", "Laser 3"], fontsize=8.5, fontweight="bold")
        ax_cross.set_ylabel("Spikes / Breath (600ms Window)", fontsize=8.5, fontweight="bold")
        ax_cross.set_title("Gasp Spiking: Laser 1 vs 2 vs 3", fontsize=9.5, fontweight="bold")
        ax_cross.legend(loc="upper right", fontsize=7.5)
        ax_cross.grid(True, linestyle=":", alpha=0.4)

        fig.savefig(out_path, dpi=130)
        plt.close(fig)

        # For laser 3, also save to root cards dir for compatibility
        if train_key == "laser3":
            root_card_path = cards_dir / f"gasp_transition_cluster_{cid:03d}.png"
            import shutil
            shutil.copyfile(out_path, root_card_path)

    return cid


def main():
    print("=== Multi-Train Zoomed-In Transition Raster Generator (Diaphragm I_off Aligned) ===", flush=True)

    # 1. Load NIDAQ Telemetry
    print("[1/5] Loading NIDAQ analog streams (Diaphragm EMG Ch 0, Flow Meter Ch 5, Laser Ch 6)...", flush=True)
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0; n_ch = 9; start_t = 796.827; dur = 226.243
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[int(start_t*fs)*n_ch : int((start_t+dur)*fs)*n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    # 2. Extract Diaphragm Landmarking
    print("[2/5] Extracting Diaphragm EMG burst landmarks (I_on, I_off, peak envelope)...", flush=True)
    env = extract_diaphragm_envelope(emg_v, fs=fs)
    t_on, t_off, t_pk, pks = segment_diaphragm_breaths(env, t_vec, fs=fs)
    print(f"Segmented {len(t_on)} diaphragm breath events across the 226s recording.", flush=True)

    td_l1 = build_train_transition_data("laser1", "Laser 1", 60.0, 2.0, 4.0, 12.0, 67.2, t_vec, emg_v, flow_v, env, t_on, t_off, t_pk, fs=fs)
    td_l2 = build_train_transition_data("laser2", "Laser 2", 120.0, 2.0, 4.0, 10.0, 125.5, t_vec, emg_v, flow_v, env, t_on, t_off, t_pk, fs=fs)
    td_l3 = build_train_transition_data("laser3", "Laser 3", 164.0, 2.5, 4.0, 11.0, 170.2, t_vec, emg_v, flow_v, env, t_on, t_off, t_pk, fs=fs)

    def serialize_train_data(td_obj):
        return {
            "train_key": td_obj.train_key,
            "train_name": td_obj.train_name,
            "t_laser_on": td_obj.t_laser_on,
            "t_laser_off": td_obj.t_laser_off,
            "laser_dur": td_obj.laser_dur,
            "t_win_start": td_obj.t_win_start,
            "t_win_end": td_obj.t_win_end,
            "sub_t_ds": td_obj.sub_t_ds,
            "sub_emg_ds": td_obj.sub_emg_ds,
            "sub_flow_ds": td_obj.sub_flow_ds,
            "sub_emg_rms_ds": td_obj.sub_emg_rms_ds,
            "trials": [
                {
                    "tr_id": tr.tr_id,
                    "row_idx": tr.row_idx,
                    "t_on": tr.t_on,
                    "t_off": tr.t_off,
                    "t_peak": tr.t_peak,
                    "t_dur_ms": tr.t_dur_ms,
                    "emg_peak_v": tr.emg_peak_v,
                    "flow_min_v": tr.flow_min_v,
                    "flow_max_v": tr.flow_max_v,
                    "flow_amp_v": tr.flow_amp_v,
                    "is_oral": tr.is_oral,
                    "state": tr.state,
                    "state_color": tr.state_color,
                    "badge_color": tr.badge_color
                }
                for tr in td_obj.trials
            ]
        }

    shared_train_data = {
        "laser1": serialize_train_data(td_l1),
        "laser2": serialize_train_data(td_l2),
        "laser3": serialize_train_data(td_l3)
    }

    # 3. Load Spikes
    print("[3/5] Loading spike data...", flush=True)
    cons_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
    spk_times = np.load(cons_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(cons_dir / "alf" / "spike_clusters.npy")
    unique_cids = sorted(list(np.unique(spk_clusters)))
    print(f"Loaded {len(unique_cids)} units to process across all 3 trains.", flush=True)

    # 4. Prepare worker arguments
    print("[4/5] Preparing parallel tasks...", flush=True)
    tasks = []
    for cid in unique_cids:
        c_spks = spk_times[spk_clusters == cid]
        out_paths = {
            "laser1": l1_dir / f"gasp_transition_cluster_{cid:03d}.png",
            "laser2": l2_dir / f"gasp_transition_cluster_{cid:03d}.png",
            "laser3": l3_dir / f"gasp_transition_cluster_{cid:03d}.png"
        }
        tasks.append((cid, c_spks, shared_train_data, out_paths))

    print(f"[5/5] Launching parallel rendering across 12 worker processes ({len(tasks)} units)...", flush=True)
    done_cnt = 0
    with ProcessPoolExecutor(max_workers=12) as executor:
        for cid in executor.map(render_single_unit, tasks):
            done_cnt += 1
            if done_cnt % 50 == 0 or done_cnt == len(tasks):
                print(f"      Rendered 3-train Diaphragm I_off cards for {done_cnt}/{len(tasks)} units...", flush=True)

    print("=== All 1,491 Diaphragm I_off Transition Cards Successfully Rendered! ===", flush=True)


if __name__ == "__main__":
    main()
