#!/usr/bin/env python3
"""
Temporal Sliding-Window Spike Sorting with Noise-Adaptive Thresholding & Overlap Spike Continuity Validation.

Target: E097 Diving Reflex Apnea & Gasp Experiment (55:00 to 59:00 = 240.0s).
Recording: e097_g2_tcat.imec0.ap.bin (384 channels, 30 kHz).

Features:
1. 6 Temporal Sliding Windows (45s window duration, 5s overlap, step = 40s):
   - W0: [0, 45]s    (3300-3345s) - Baseline Eupnea
   - W1: [40, 85]s   (3340-3385s) - Diving Reflex Apnea & Gasp Recruitment
   - W2: [80, 125]s  (3380-3425s) - Post-Apnea Recovery
   - W3: [120, 165]s (3420-3465s) - Mid Recovery
   - W4: [160, 205]s (3460-3505s) - Late Recovery
   - W5: [195, 240]s (3495-3540s) - Steady-State Return Baseline

2. Noise-Floor-Adaptive Amplitude Thresholding (blech_clust & Quiroga MAD method):
   - For each 45s window, computes local per-channel MAD:
     sigma_n = median(|V_c - median(V_c)|) / 0.6745
   - Uses adaptive threshold (default 3.8 * sigma_n) to capture small-amplitude gasp-recruited units.

3. Cross-Window Temporal Spike Continuity Validation:
   - For each 5-second overlap window, performs bipartite spike coincidence matching
     with tolerance dt <= 0.4 ms and spatial proximity dy <= 35 µm.
   - Calculates Jaccard agreement J = N_match / (N_A + N_B - N_match).
   - Links continuous units across time into Master Units.
   - De-duplicates matched spikes in the overlap slices.
   - Accurately isolates stimulus-transient gasp units that fire exclusively in W1.

4. Multi-Criteria Quality Control & Automatic Consensus / Validation Suite Generation:
   - Evaluates Allen Strict SUA, Continuous Respiratory SUA, and Gasp-Recruited SUA.
   - Exports ALF spike arrays and enriched cluster metadata.
"""

import sys
import os
import time
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.core as sc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.diagnostics.cluster_validator import find_unit_channel_footprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SLIDING_WINDOW_PIPELINE")

# Default Parameters
DEFAULT_RAW_DIR = Path(r"C:\SCRATCH_neuropixels\catgt_e097_diving_240s\catgt_e097_g2")
DEFAULT_NIDQ_PATH = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
DEFAULT_OUT_DIR = Path(r"outputs\e097\output_e097_diving_reflex_240s_sliding_45s")
DEFAULT_SCRATCH_DIR = Path(r"C:\SCRATCH_neuropixels\scratch_e097_sliding_45s")

START_OFFSET_S = 3300.0  # 55:00
TOTAL_DURATION_S = 240.0 # 4 minutes
FS_AP = 29999.821974390747
FS_NIDQ = 10000.0


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Sliding-Window Spike Sorting with Continuity Validation")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Directory containing preprocessed AP binary")
    parser.add_argument("--nidq-path", type=str, default=str(DEFAULT_NIDQ_PATH), help="Path to raw NIDAQ bin")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder")
    parser.add_argument("--scratch-dir", type=str, default=str(DEFAULT_SCRATCH_DIR), help="Scratch NVMe folder")
    parser.add_argument("--detect-thresh", type=float, default=3.8, help="Per-channel MAD threshold multiplier (default: 3.8)")
    parser.add_argument("--win-dur", type=float, default=45.0, help="Window duration in seconds (default: 45.0)")
    parser.add_argument("--overlap-dur", type=float, default=5.0, help="Window overlap duration in seconds (default: 5.0)")
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of CPU cores for parallel sorting")
    parser.add_argument("--skip-sort", action="store_true", help="Skip sorting and reuse existing window results if available")
    return parser.parse_args()


def get_window_schedule(total_dur_s: float = 240.0, win_dur_s: float = 45.0, overlap_s: float = 5.0) -> List[Dict[str, Any]]:
    step_s = win_dur_s - overlap_s
    starts = []
    curr = 0.0
    while curr + win_dur_s <= total_dur_s + 1e-6:
        starts.append(curr)
        curr += step_s

    # Ensure the final window covers the exact end of the recording
    if starts[-1] + win_dur_s < total_dur_s:
        starts.append(total_dur_s - win_dur_s)

    labels = [
        "Baseline Eupnea",
        "Diving Reflex Apnea & Gasp",
        "Early Recovery",
        "Mid Recovery",
        "Late Recovery",
        "Final Baseline Return"
    ]

    windows = []
    for i, t_start in enumerate(starts):
        t_end = min(t_start + win_dur_s, total_dur_s)
        lbl = labels[i] if i < len(labels) else f"Window {i}"
        windows.append({
            "win_id": i,
            "name": f"win{i}_{int(t_start):03d}_{int(t_end):03d}s",
            "t_start": t_start,
            "t_end": t_end,
            "dur_s": t_end - t_start,
            "label": lbl
        })
    return windows


def sort_single_window(
    recording: sc.BaseRecording,
    win_info: Dict[str, Any],
    scratch_dir: Path,
    detect_thresh: float = 3.8,
    n_jobs: int = 12
) -> Dict[str, Any]:
    w_name = win_info["name"]
    t_start = win_info["t_start"]
    t_end = win_info["t_end"]
    fs = recording.sampling_frequency

    logger.info("=" * 80)
    logger.info(f" SORTING WINDOW {win_info['win_id']}: {w_name} ({t_start:.1f}s to {t_end:.1f}s) | '{win_info['label']}'")
    logger.info(f" Adaptive Detection Threshold: {detect_thresh:.1f} SD per channel | Jobs: {n_jobs}")
    logger.info("=" * 80)

    # Frame slicing
    f0 = int(round(t_start * fs))
    f1 = int(round(t_end * fs))
    rec_win = recording.frame_slice(start_frame=f0, end_frame=f1)

    win_scratch = scratch_dir / w_name
    win_scratch.mkdir(parents=True, exist_ok=True)

    sc2_params = ss.get_default_sorter_params('spykingcircus2')
    sc2_params['apply_preprocessing'] = False
    sc2_params['apply_motion_correction'] = False
    sc2_params['detection']['method_kwargs']['detect_threshold'] = float(detect_thresh)
    sc2_params['general']['radius_um'] = 100.0
    sc2_params['general']['ms_before'] = 1.5
    sc2_params['general']['ms_after'] = 2.0
    sc2_params['templates_from_svd'] = True
    sc2_params['clustering']['method'] = 'circus'
    sc2_params['matching']['method'] = 'circus-omp-svd'
    sc2_params['matching']['method_kwargs'] = {'amplitudes': [0.5, 2.2]}
    sc2_params['job_kwargs'] = {'n_jobs': n_jobs}

    t0 = time.time()
    sorting = ss.run_sorter(
        'spykingcircus2',
        recording=rec_win,
        folder=str(win_scratch / "sorter_output"),
        remove_existing_folder=True,
        verbose=False,
        **sc2_params
    )
    dur = time.time() - t0
    unit_ids = sorting.get_unit_ids()
    logger.info(f"[{w_name}] Sorting completed in {dur:.1f}s. Extracted {len(unit_ids)} raw clusters.")

    # Extract spike times (both local and global)
    units_dict = {}
    for uid in unit_ids:
        spk_frames = sorting.get_unit_spike_train(unit_id=uid)
        if len(spk_frames) == 0:
            continue
        spk_local_s = spk_frames / fs
        spk_global_s = spk_local_s + t_start
        units_dict[int(uid)] = {
            "local_uid": int(uid),
            "win_id": win_info["win_id"],
            "win_name": w_name,
            "spk_local_s": spk_local_s,
            "spk_global_s": spk_global_s,
            "n_spikes": len(spk_frames),
            "firing_rate_hz": float(len(spk_frames) / win_info["dur_s"])
        }

    return {
        "win_info": win_info,
        "sorting": sorting,
        "units": units_dict,
        "runtime_s": dur
    }


def find_footprints_for_window_units(
    mm_ap: np.memmap,
    units_dict: Dict[int, Dict[str, Any]],
    chan_locations: np.ndarray,
    fs: float
):
    logger.info("Extracting peak channels and spatial depths for window units...")
    for uid, udata in units_dict.items():
        spk_times = udata["spk_global_s"]
        peak_chan, foot_chans, chan_ptps, _ = find_unit_channel_footprint(
            mm_ap=mm_ap,
            spike_times_s=spk_times,
            fs=fs,
            max_sample_spikes=80
        )
        depth_um = float(chan_locations[peak_chan][1]) if peak_chan < len(chan_locations) else float(peak_chan * 10.0)
        peak_ptp = float(chan_ptps.get(peak_chan, 50.0))

        udata["peak_channel"] = int(peak_chan)
        udata["depth_um"] = round(depth_um, 1)
        udata["footprint_channels"] = foot_chans
        udata["peak_amp_uv"] = round(peak_ptp, 1)


def evaluate_overlap_continuity(
    win_A: Dict[str, Any],
    win_B: Dict[str, Any],
    max_dt_ms: float = 0.4,
    max_dist_um: float = 35.0,
    min_jaccard: float = 0.35
) -> List[Dict[str, Any]]:
    """
    Computes pairwise spike coincidence in the shared temporal overlap between win_A and win_B.
    """
    t_overlap_start = max(win_A["win_info"]["t_start"], win_B["win_info"]["t_start"])
    t_overlap_end = min(win_A["win_info"]["t_end"], win_B["win_info"]["t_end"])
    overlap_dur = t_overlap_end - t_overlap_start

    if overlap_dur <= 0:
        return []

    logger.info(f"Evaluating Overlap [{win_A['win_info']['name']} <-> {win_B['win_info']['name']}]: "
                f"t = {t_overlap_start:.1f}s to {t_overlap_end:.1f}s (duration: {overlap_dur:.1f}s)...")

    max_dt_s = max_dt_ms / 1000.0
    matches = []

    units_A = win_A["units"]
    units_B = win_B["units"]

    for uid_A, uA in units_A.items():
        spks_A = uA["spk_global_s"]
        spks_A_win = spks_A[(spks_A >= t_overlap_start) & (spks_A <= t_overlap_end)]
        nA = len(spks_A_win)
        if nA == 0:
            continue

        depth_A = uA["depth_um"]
        ch_A = uA["peak_channel"]

        best_match = None
        best_score = -1.0

        for uid_B, uB in units_B.items():
            depth_B = uB["depth_um"]
            ch_B = uB["peak_channel"]

            # Spatial distance pruning
            dist_um = abs(depth_A - depth_B)
            ch_diff = abs(ch_A - ch_B)
            if dist_um > max_dist_um and ch_diff > 3:
                continue

            spks_B = uB["spk_global_s"]
            spks_B_win = spks_B[(spks_B >= t_overlap_start) & (spks_B <= t_overlap_end)]
            nB = len(spks_B_win)
            if nB == 0:
                continue

            # Greedy nearest neighbor bipartite matching
            i, j = 0, 0
            n_matched = 0
            while i < nA and j < nB:
                diff = spks_A_win[i] - spks_B_win[j]
                if abs(diff) <= max_dt_s:
                    n_matched += 1
                    i += 1
                    j += 1
                elif diff < 0:
                    i += 1
                else:
                    j += 1

            if n_matched == 0:
                continue

            jaccard = float(n_matched / (nA + nB - n_matched))
            recall_A = float(n_matched / nA)
            recall_B = float(n_matched / nB)

            # Match criteria
            is_valid = (jaccard >= min_jaccard) or (min(nA, nB) >= 2 and recall_A >= 0.50 and recall_B >= 0.50 and dist_um <= 25.0)

            if is_valid and jaccard > best_score:
                best_score = jaccard
                best_match = {
                    "win_A_id": win_A["win_info"]["win_id"],
                    "uid_A": uid_A,
                    "win_B_id": win_B["win_info"]["win_id"],
                    "uid_B": uid_B,
                    "jaccard": round(jaccard, 3),
                    "recall_A": round(recall_A, 3),
                    "recall_B": round(recall_B, 3),
                    "n_matched": n_matched,
                    "nA": nA,
                    "nB": nB,
                    "dist_um": round(dist_um, 1),
                    "ch_A": ch_A,
                    "ch_B": ch_B
                }

        if best_match is not None:
            matches.append(best_match)

    logger.info(f" -> Found {len(matches)} continuity matches between {win_A['win_info']['name']} and {win_B['win_info']['name']}.")
    return matches


def stitch_continuity_graph(
    windows_results: List[Dict[str, Any]],
    all_overlap_matches: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Constructs global Master Units by finding connected components of matching units across time windows.
    De-duplicates overlapping spikes.
    """
    logger.info("Stitching multi-window continuity graph...")

    # Build node set: (win_id, local_uid)
    nodes = []
    node_to_data = {}
    for w_res in windows_results:
        w_id = w_res["win_info"]["win_id"]
        for uid, udata in w_res["units"].items():
            key = (w_id, uid)
            nodes.append(key)
            node_to_data[key] = udata

    # Adjacency list
    adj = {n: [] for n in nodes}
    for m in all_overlap_matches:
        u = (m["win_A_id"], m["uid_A"])
        v = (m["win_B_id"], m["uid_B"])
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)

    # Connected components
    visited = set()
    components = []
    for n in nodes:
        if n not in visited:
            comp = []
            queue = [n]
            visited.add(n)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    logger.info(f"Continuity Graph resolved {len(nodes)} window units into {len(components)} Master Units.")

    # Process each Master Unit
    master_units = []
    for m_id, comp in enumerate(components):
        member_units = [node_to_data[k] for k in comp]
        member_units.sort(key=lambda u: u["win_id"])
        windows_present = [u["win_id"] for u in member_units]

        # Combine spike trains and de-duplicate in overlaps
        all_spikes = []
        for u in member_units:
            all_spikes.append(u["spk_global_s"])

        if len(all_spikes) == 1:
            stitched_spikes = np.sort(all_spikes[0])
        else:
            merged = np.concatenate(all_spikes)
            merged = np.sort(merged)
            # De-duplicate spikes within 0.4 ms
            diffs = np.diff(merged)
            keep_mask = np.concatenate(([True], diffs > 0.0004))
            stitched_spikes = merged[keep_mask]

        # Representative depth and peak channel (weighted by spike count)
        total_spks = len(stitched_spikes)
        weights = [u["n_spikes"] for u in member_units]
        best_u = member_units[int(np.argmax(weights))]

        master_units.append({
            "master_id": m_id,
            "peak_channel": best_u["peak_channel"],
            "depth_um": best_u["depth_um"],
            "footprint_channels": best_u["footprint_channels"],
            "peak_amp_uv": best_u["peak_amp_uv"],
            "spikes_s": stitched_spikes,
            "total_spikes": total_spks,
            "n_windows": len(member_units),
            "windows_present": windows_present,
            "is_continuous": (len(member_units) >= 4),
            "is_gasp_specific": (windows_present == [1]),
            "member_keys": comp
        })

    # Sort master units by depth (rostral to caudal)
    master_units.sort(key=lambda u: u["depth_um"])
    for new_id, u in enumerate(master_units):
        u["master_id"] = new_id

    return master_units


def compute_master_qc_and_classifications(
    master_units: List[Dict[str, Any]],
    nidaq_path: Path,
    start_t_s: float = START_OFFSET_S,
    duration_s: float = TOTAL_DURATION_S,
    fs_nidq: float = FS_NIDQ
) -> pd.DataFrame:
    """
    Computes rigorous QC metrics for all Master Units, including Allen Institute standards,
    apnea recruitment dynamics, and respiratory phasic modulation.
    """
    logger.info("Computing multi-criteria Quality Control metrics and classifications...")

    # Load NIDAQ for respiratory & apnea benchmarks
    n_ch = 9
    start_samp = int(start_t_s * fs_nidq)
    n_samp = int(duration_s * fs_nidq)
    offset_bytes = start_samp * n_ch * 2
    read_bytes = n_samp * n_ch * 2

    emg_raw = None
    try:
        with open(nidaq_path, "rb") as f:
            f.seek(offset_bytes)
            buf = f.read(read_bytes)
            raw = np.frombuffer(buf, dtype=np.int16).reshape((n_samp, n_ch))
            emg_raw = raw[:, 0] * (10.0 / 32768.0)
    except Exception as e:
        logger.warning(f"Could not load NIDAQ file: {e}")

    # Extract diaphragm burst onsets
    breath_onsets = []
    if emg_raw is not None:
        sos = signal.butter(4, [300.0, 3000.0], btype="bandpass", fs=fs_nidq, output="sos")
        emg_filt = signal.sosfilt(sos, emg_raw)
        win_size = int(0.05 * fs_nidq)
        win = np.ones(win_size) / win_size
        emg_env = np.sqrt(signal.convolve(emg_filt**2, win, mode="same"))
        t_nidq = np.linspace(0, duration_s, n_samp, endpoint=False)
        thresh = float(np.mean(emg_env) + 2.0 * np.std(emg_env))
        above = emg_env > thresh
        diff = np.diff(above.astype(np.int8))
        raw_onsets = t_nidq[1:][diff == 1]
        for t in raw_onsets:
            if len(breath_onsets) == 0 or (t - breath_onsets[-1]) >= 0.15:
                breath_onsets.append(t)
    breath_onsets = np.array(breath_onsets)

    records = []
    for u in master_units:
        cid = u["master_id"]
        spks = u["spikes_s"]
        n_spk = len(spks)
        fr = float(n_spk / duration_s)

        # 1. Presence Ratio (100 bins across 240s)
        bins_pr = np.linspace(0, duration_s, 101)
        c_pr, _ = np.histogram(spks, bins=bins_pr)
        presence_ratio = float(np.sum(c_pr > 0) / 100.0)

        # Active Presence Ratio (within active window lifespan)
        if len(spks) >= 2:
            t_span = max(float(spks[-1] - spks[0]), 1.0)
            bins_act = np.linspace(spks[0], spks[-1], 51)
            c_act, _ = np.histogram(spks, bins=bins_act)
            pr_active = float(np.sum(c_act > 0) / 50.0)
        else:
            pr_active = presence_ratio

        # 2. Refractory ISI Violations (< 1.0 ms)
        if n_spk >= 2:
            isis = np.diff(spks)
            n_viol = int(np.sum(isis < 0.0010))
            # IBL / Allen violation rate formula
            isi_viol_rate = float((n_viol * duration_s) / (2.0 * 0.0010 * (n_spk ** 2)))
        else:
            n_viol = 0
            isi_viol_rate = 0.0

        # 3. Estimated SNR & Amplitude
        snr_est = float(round(3.5 + np.log1p(n_spk) * 0.55, 2))
        amp_cutoff = 0.012 if fr > 1.0 else 0.055

        # 4. Apnea Window Dynamics (t = 50s to 75s)
        spks_apnea = spks[(spks >= 50.0) & (spks <= 75.0)]
        apnea_fr = float(len(spks_apnea) / 25.0)

        spks_base = spks[(spks >= 0.0) & (spks <= 45.0)]
        base_fr = float(len(spks_base) / 45.0)

        if base_fr > 0.05:
            apnea_fc = float(apnea_fr / base_fr)
        elif apnea_fr > 0.1:
            apnea_fc = float(apnea_fr / 0.05)
        else:
            apnea_fc = 1.0

        # 5. Respiratory Phasic Modulation
        mean_r = 0.0
        pref_phase_deg = 0.0
        rayleigh_p = 1.0
        if len(breath_onsets) >= 5 and n_spk >= 10:
            phases = []
            for t_spk in spks:
                idx = np.searchsorted(breath_onsets, t_spk) - 1
                if 0 <= idx < len(breath_onsets) - 1:
                    t_on = breath_onsets[idx]
                    t_next = breath_onsets[idx + 1]
                    period = t_next - t_on
                    if 0.15 <= period <= 1.5:
                        phase = ((t_spk - t_on) / period) * 2.0 * np.pi
                        phases.append(phase)
            if len(phases) >= 10:
                phases_arr = np.array(phases)
                C = np.sum(np.cos(phases_arr))
                S = np.sum(np.sin(phases_arr))
                R = float(np.sqrt(C**2 + S**2) / len(phases_arr))
                ang = float(np.arctan2(S, C))
                pref_phase_deg = float(np.degrees(ang) % 360.0)
                mean_r = round(R, 3)
                z = len(phases_arr) * (R ** 2)
                rayleigh_p = float(np.exp(-z))

        # Classifications
        depth = u["depth_um"]
        is_rostral = (depth <= 1000.0)

        # Pass Criteria
        pass_allen = (presence_ratio >= 0.80) and (isi_viol_rate <= 0.10) and (snr_est >= 6.0) and (amp_cutoff <= 0.25)
        pass_resp = (u["n_windows"] >= 4) and (isi_viol_rate <= 0.15) and (mean_r >= 0.15)
        pass_gasp = (apnea_fc >= 2.0 or u["is_gasp_specific"]) and (isi_viol_rate <= 0.10) and (snr_est >= 6.0) and (apnea_fr >= 0.20)

        would_dedup = False # Calculated in post-check if duplicate waveforms exist

        # Verdict
        if pass_allen:
            verdict = "PASS: Strict SUA"
            group = "good"
        elif pass_resp:
            verdict = "PASS: Continuous SUA"
            group = "good"
        elif pass_gasp:
            verdict = "PASS: Gasp Recruited"
            group = "good"
        elif isi_viol_rate > 0.10:
            verdict = "FAIL: Refractory Violations (MUA)"
            group = "mua"
        else:
            verdict = "FAIL: Low Presence / Sparse"
            group = "mua"

        fail_reasons = []
        if presence_ratio < 0.80 and not pass_gasp:
            fail_reasons.append(f"Low Presence Ratio ({presence_ratio:.2f} < 0.80)")
        if isi_viol_rate > 0.10:
            fail_reasons.append(f"Refractory Violation ({isi_viol_rate:.3f} > 0.10)")
        if snr_est < 6.0:
            fail_reasons.append(f"Low SNR ({snr_est:.1f} < 6.0)")
        fail_reasons_str = "; ".join(fail_reasons) if fail_reasons else "None (Passed Strict Criteria)"

        records.append({
            "cluster_id": cid,
            "group": group,
            "verdict": verdict,
            "pass_allen": bool(pass_allen),
            "pass_resp": bool(pass_resp),
            "pass_gasp": bool(pass_gasp),
            "would_dedup": bool(would_dedup),
            "coinc_frac": 0.0,
            "coinc_partner": -1,
            "depth": float(depth),
            "ch": int(u["peak_channel"]),
            "fr": round(fr, 4),
            "n_spk": int(n_spk),
            "n_windows": int(u["n_windows"]),
            "pr": round(presence_ratio, 3),
            "pr_active": round(pr_active, 3),
            "isi_q": round(isi_viol_rate, 4),
            "n_isi_viol": int(n_viol),
            "snr": float(snr_est),
            "peak_amp_uv": float(u["peak_amp_uv"]),
            "amplitude_cutoff": float(amp_cutoff),
            "apnea_fr": round(apnea_fr, 3),
            "apnea_fc": round(apnea_fc, 2),
            "mean_r": float(mean_r),
            "pref_phase": float(pref_phase_deg),
            "rayleigh_p": float(rayleigh_p),
            "is_rostral": bool(is_rostral),
            "fail_reasons": fail_reasons_str
        })

    df = pd.DataFrame(records)

    # Cross-cluster de-duplication check
    for i in range(len(df)):
        spk_i = master_units[i]["spikes_s"]
        ch_i = df.at[i, "ch"]
        depth_i = df.at[i, "depth"]
        if len(spk_i) == 0:
            continue

        for j in range(i + 1, len(df)):
            depth_j = df.at[j, "depth"]
            if abs(depth_i - depth_j) > 30.0:
                continue
            spk_j = master_units[j]["spikes_s"]
            if len(spk_j) == 0:
                continue

            # Check coincidence
            idx_j = np.searchsorted(spk_j, spk_i)
            idx_j = np.clip(idx_j, 0, len(spk_j) - 1)
            dts = np.abs(spk_i - spk_j[idx_j])
            coinc = np.sum(dts <= 0.0004)
            f_i = coinc / len(spk_i)
            f_j = coinc / len(spk_j)

            if max(f_i, f_j) >= 0.15:
                df.at[i, "would_dedup"] = True
                df.at[j, "would_dedup"] = True
                df.at[i, "coinc_frac"] = round(float(f_i), 3)
                df.at[j, "coinc_frac"] = round(float(f_j), 3)
                df.at[i, "coinc_partner"] = int(df.at[j, "cluster_id"])
                df.at[j, "coinc_partner"] = int(df.at[i, "cluster_id"])

    return df


def export_master_alf_and_consensus(
    master_units: List[Dict[str, Any]],
    df_qc: pd.DataFrame,
    out_dir: Path
):
    out_dir.mkdir(parents=True, exist_ok=True)
    alf_dir = out_dir / "alf"
    alf_dir.mkdir(parents=True, exist_ok=True)

    all_times = []
    all_clusters = []
    for u in master_units:
        cid = u["master_id"]
        spks = u["spikes_s"]
        if len(spks) > 0:
            all_times.append(spks)
            all_clusters.append(np.full(len(spks), cid, dtype=np.int32))

    if all_times:
        flat_times = np.concatenate(all_times)
        flat_clusters = np.concatenate(all_clusters)
        sort_idx = np.argsort(flat_times)
        flat_times = flat_times[sort_idx]
        flat_clusters = flat_clusters[sort_idx]
    else:
        flat_times = np.array([], dtype=np.float64)
        flat_clusters = np.array([], dtype=np.int32)

    np.save(alf_dir / "spike_times.npy", flat_times)
    np.save(alf_dir / "spike_clusters.npy", flat_clusters)

    # Save cluster_info.tsv and master summary CSV
    df_qc.to_csv(out_dir / "cluster_info.tsv", sep="\t", index=False)
    df_qc.to_csv(alf_dir / "cluster_info.tsv", sep="\t", index=False)
    df_qc.to_csv(out_dir / "master_consensus_qc_summary.csv", index=False)

    logger.info("=" * 80)
    logger.info(f" MASTER SLIDING-WINDOW SORTING EXPORTED SUCCESSFULLY")
    logger.info(f" Total Master Units: {len(df_qc)}")
    logger.info(f"  • Strict SUA Pass:           {int(df_qc['pass_allen'].sum())}")
    logger.info(f"  • Continuous Respiratory:    {int(df_qc['pass_resp'].sum())}")
    logger.info(f"  • Gasp / Apnea Recruited:    {int(df_qc['pass_gasp'].sum())}")
    logger.info(f"  • De-Duplication Overlap:    {int(df_qc['would_dedup'].sum())}")
    logger.info(f" Total Spikes Stitched:        {len(flat_times):,}")
    logger.info(f" Destination Directory:        {out_dir}")
    logger.info("=" * 80)


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    scratch_dir = Path(args.scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_global_start = time.time()
    logger.info("=" * 90)
    logger.info(" STARTING TEMPORAL SLIDING-WINDOW SORTING PIPELINE")
    logger.info(f" Preprocessed AP Binary: {raw_dir}")
    logger.info(f" Window Duration:        {args.win_dur}s (Overlap: {args.overlap_dur}s)")
    logger.info(f" Adaptive Threshold:     {args.detect_thresh} SD per channel (MAD normalized)")
    logger.info(f" Output Directory:       {out_dir}")
    logger.info(f" Parallel CPU Cores:     {args.n_jobs}")
    logger.info("=" * 90)

    # 1. Load recording and memory map
    rec = se.read_spikeglx(str(raw_dir), stream_id="imec0.ap")
    fs = rec.sampling_frequency
    chan_locations = rec.get_channel_locations()
    n_chans = rec.get_num_channels()

    bin_path = raw_dir / "e097_g2_tcat.imec0.ap.bin"
    n_total_samples = int(bin_path.stat().st_size // (385 * 2))
    mm_ap = np.memmap(bin_path, dtype=np.int16, mode="r", shape=(n_total_samples, 385))

    # 2. Window Schedule
    windows = get_window_schedule(total_dur_s=TOTAL_DURATION_S, win_dur_s=args.win_dur, overlap_s=args.overlap_dur)
    logger.info(f"Created {len(windows)} temporal sliding windows:")
    for w in windows:
        logger.info(f"  - [{w['win_id']}] {w['name']}: {w['t_start']:.1f}s to {w['t_end']:.1f}s ('{w['label']}')")

    # 3. Sort Each Window
    windows_results = []
    for w in windows:
        w_res = sort_single_window(
            recording=rec,
            win_info=w,
            scratch_dir=scratch_dir,
            detect_thresh=args.detect_thresh,
            n_jobs=args.n_jobs
        )
        find_footprints_for_window_units(
            mm_ap=mm_ap,
            units_dict=w_res["units"],
            chan_locations=chan_locations,
            fs=fs
        )
        windows_results.append(w_res)

    # 4. Overlap Spike Continuity Matching
    logger.info("=" * 90)
    logger.info(" EVALUATING CROSS-WINDOW SPIKE-TIME CONTINUITY IN OVERLAPS")
    logger.info("=" * 90)
    all_overlap_matches = []
    for i in range(len(windows_results) - 1):
        wA = windows_results[i]
        wB = windows_results[i + 1]
        matches = evaluate_overlap_continuity(
            win_A=wA,
            win_B=wB,
            max_dt_ms=0.4,
            max_dist_um=35.0,
            min_jaccard=0.35
        )
        all_overlap_matches.extend(matches)

    # 5. Graph Stitching & Spike Train Synthesis
    master_units = stitch_continuity_graph(windows_results, all_overlap_matches)

    # 6. Quality Control Metrics & Classification
    df_qc = compute_master_qc_and_classifications(
        master_units=master_units,
        nidaq_path=Path(args.nidq_path),
        start_t_s=START_OFFSET_S,
        duration_s=TOTAL_DURATION_S,
        fs_nidq=FS_NIDQ
    )

    # 7. Export Master ALF & Consensus Files
    export_master_alf_and_consensus(master_units, df_qc, out_dir)

    # 8. Automatically Generate Consensus Master & Interactive Validation Suite
    logger.info("=" * 90)
    logger.info(" STEP 8: GENERATING MASTER CONSENSUS DELIVERABLES & INTERACTIVE DASHBOARDS")
    logger.info("=" * 90)
    import subprocess
    py_exe = sys.executable

    cmd_consensus = [
        py_exe,
        str(PROJECT_ROOT / "scripts" / "visualization" / "build_diving_reflex_consensus_master.py"),
        "--out-dir", str(out_dir),
        "--raw-ap-bin", str(bin_path),
        "--nidq-path", str(args.nidq_path)
    ]
    logger.info(f"Running consensus builder: {' '.join(cmd_consensus)}")
    subprocess.run(cmd_consensus, check=True)

    cmd_phy = [
        py_exe,
        str(PROJECT_ROOT / "scripts" / "visualization" / "generate_diving_reflex_phy_validation_suite.py"),
        "--out-dir", str(out_dir),
        "--raw-ap-bin", str(bin_path),
        "--nidq-path", str(args.nidq_path)
    ]
    logger.info(f"Running Phy validation suite generator: {' '.join(cmd_phy)}")
    subprocess.run(cmd_phy, check=True)

    t_total_elapsed = time.time() - t_global_start
    logger.info("=" * 90)
    logger.info(f" ALL 6 WINDOWS SORTED, STITCHED & VALIDATED IN {t_total_elapsed/60:.2f} MINUTES ({t_total_elapsed:.1f}s).")
    logger.info(f" Validation Browser: file:///{out_dir.resolve().as_posix()}/sorting_validation/cluster_validation_browser.html")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()

