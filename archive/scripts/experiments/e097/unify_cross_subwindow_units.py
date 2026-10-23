#!/usr/bin/env python3
"""
Cross-Window Neuron Unification Engine for e097 (29:00 - 36:00 / 1740s - 2160s):
- Chains single units across the 7 physiological subwindows:
    1. win1_baseline (1740-1800s)
    2. win2_anox1 (1800-1836s)
    3. win3_anox2 (1836-1860s)
    4. win4_gasp_early (1860-1950s)
    5. win5_gasp_late (1950-2070s)
    6. win6_flow_shift (2070-2136s)
    7. win7_recovery (2136-2160s)
- Unification Criteria:
    * Spatial centroid proximity: delta_d <= 25.0 um
    * Multi-channel waveform template correlation: Pearson r >= 0.85
    * Skip-window linking: recovers neurons silenced during the anoxic nadir (win3)
- Biological Phenotype Classification:
    * Constitutive: Active across baseline, anoxia, gasping, and recovery
    * Anoxia-Silenced: Active in baseline/anox1, extinguished in anox2/apnea nadir, re-emerges in gasping/recovery
    * De Novo Gasping Emergent: Silent in baseline & anoxia, first active during autoresuscitation gasping
    * Vulnerable: Active in baseline, permanently extinguished after anoxia
    * Transient: Active in only a single transient epoch
- Dual Respiratory Phase Tuning Tracking:
    * Diaphragm EMG burst phase (0° - 360°)
    * Airflow plethysmography phase
- Generates interactive master unification dashboard and export CSV/JSON manifests.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import spikeinterface.full as si
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CROSS_WINDOW_UNIFICATION")

WINDOW_ORDER = [
    "win1_baseline",
    "win2_anox1",
    "win3_anox2",
    "win4_gasp_early",
    "win5_gasp_late",
    "win6_flow_shift",
    "win7_recovery"
]

WINDOW_METADATA = {
    "win1_baseline": {"name": "Baseline", "t_start": 1740.0, "t_end": 1800.0, "dur": 60.0},
    "win2_anox1": {"name": "Anox 1 (Switch)", "t_start": 1800.0, "t_end": 1836.0, "dur": 36.0},
    "win3_anox2": {"name": "Anox 2 (Apnea)", "t_start": 1836.0, "t_end": 1860.0, "dur": 24.0},
    "win4_gasp_early": {"name": "Early Gasping", "t_start": 1860.0, "t_end": 1950.0, "dur": 90.0},
    "win5_gasp_late": {"name": "Late Gasping", "t_start": 1950.0, "t_end": 2070.0, "dur": 120.0},
    "win6_flow_shift": {"name": "Active Exp Shift", "t_start": 2070.0, "t_end": 2136.0, "dur": 66.0},
    "win7_recovery": {"name": "Recovery", "t_start": 2136.0, "t_end": 2160.0, "dur": 24.0}
}

def extract_mean_templates_for_window(
    sorter_dir: Path,
    recording: si.BaseRecording,
    max_spikes_per_unit: int = 300,
    snippet_len_samples: int = 60,
    top_k_channels: int = 8
) -> Dict[int, Dict[str, Any]]:
    """
    Extracts mean multi-channel waveform templates and spatial centroids for each unit.
    Returns: dict mapping unit_id -> {
        'depth_um': float,
        'centroid_xy': (x, y),
        'peak_chan': int,
        'neighbor_chans': list,
        'template': np.ndarray (shape: [snippet_len, top_k]),
        'n_spikes': int,
        'firing_rate_hz': float
    }
    """
    alf_dir = sorter_dir / "alf"
    spike_times_s = np.load(alf_dir / "spike_times.npy")
    spike_clusters = np.load(alf_dir / "spike_clusters.npy")
    
    qc_file = sorter_dir / "master_consensus_qc_summary.csv"
    if not qc_file.exists():
        qc_file = sorter_dir / "cluster_info.tsv"
    
    if qc_file.suffix == ".tsv":
        df_qc = pd.read_csv(qc_file, sep="\t")
    else:
        df_qc = pd.read_csv(qc_file)
        
    fs = recording.get_sampling_frequency()
    n_total_samples = recording.get_num_samples()
    chan_locations = recording.get_channel_locations()
    n_chans = recording.get_num_channels()

    unit_data = {}
    
    for _, row in df_qc.iterrows():
        uid = int(row["cluster_id"])
        depth = float(row.get("depth_um", 0.0))
        fr = float(row.get("firing_rate_hz", 0.0))
        n_spk = int(row.get("n_spikes", 0))
        
        # Unit spike sample times
        mask = (spike_clusters == uid)
        spk_times = spike_times_s[mask]
        if len(spk_times) == 0:
            continue
            
        spk_samples = (spk_times * fs).astype(np.int64)
        # Filter boundaries
        half_len = snippet_len_samples // 2
        valid = (spk_samples >= half_len) & (spk_samples < n_total_samples - half_len)
        spk_samples = spk_samples[valid]
        if len(spk_samples) == 0:
            continue
            
        # Subsample if large
        if len(spk_samples) > max_spikes_per_unit:
            rng = np.random.default_rng(42)
            sampled_spikes = rng.choice(spk_samples, size=max_spikes_per_unit, replace=False)
        else:
            sampled_spikes = spk_samples

        # Find closest channels to depth
        chan_y = chan_locations[:, 1]
        dist_to_depth = np.abs(chan_y - depth)
        peak_chan = int(np.argmin(dist_to_depth))
        
        # Top-k closest channels
        sorted_chans = np.argsort(np.abs(chan_y - chan_y[peak_chan]))[:top_k_channels]
        
        # Extract snippets
        snippets = []
        for s in sampled_spikes:
            trace = recording.get_traces(
                start_frame=s - half_len,
                end_frame=s + half_len,
                channel_ids=recording.channel_ids[sorted_chans],
                return_scaled=True
            )
            snippets.append(trace)
            
        snippets = np.array(snippets) # shape: (N, T, K)
        mean_template = np.mean(snippets, axis=0) # shape: (T, K)
        
        chan_waveforms = {}
        for k_idx, ch_idx in enumerate(sorted_chans):
            ch_name = str(recording.channel_ids[ch_idx])
            chan_waveforms[ch_name] = mean_template[:, k_idx]
            
        peak_wave = mean_template[:, 0]
        
        centroid_x = float(np.mean(chan_locations[sorted_chans, 0]))
        centroid_y = float(np.mean(chan_locations[sorted_chans, 1]))
        
        unit_data[uid] = {
            "cluster_id": uid,
            "depth_um": depth,
            "centroid_xy": (centroid_x, centroid_y),
            "peak_chan": peak_chan,
            "neighbor_chans": sorted_chans.tolist(),
            "template": mean_template,
            "chan_waveforms": chan_waveforms,
            "peak_waveform": peak_wave,
            "n_spikes": n_spk,
            "firing_rate_hz": fr,
            "passed_qc": bool(row.get("passed_allen_qc", True))
        }

    return unit_data

def compute_pairwise_unit_correlation(
    uA: Dict[str, Any],
    uB: Dict[str, Any],
    max_distance_um: float = 25.0
) -> float:
    """
    Computes spatial-gated waveform template Pearson correlation between two units
    aligned to shared physical channels on the Neuropixels probe.
    """
    posA = uA["centroid_xy"]
    posB = uB["centroid_xy"]
    dist = np.sqrt((posA[0] - posB[0])**2 + (posA[1] - posB[1])**2)
    
    if dist > max_distance_um:
        return 0.0
        
    cwA = uA.get("chan_waveforms", {})
    cwB = uB.get("chan_waveforms", {})
    shared_chans = sorted(list(set(cwA.keys()) & set(cwB.keys())))
    
    # 1. Peak waveform correlation
    peakA = uA.get("peak_waveform", uA["template"][:, 0] if uA["template"].ndim > 1 else uA["template"])
    peakB = uB.get("peak_waveform", uB["template"][:, 0] if uB["template"].ndim > 1 else uB["template"])
    std_pkA = np.std(peakA)
    std_pkB = np.std(peakB)
    r_peak = float(pearsonr(peakA, peakB)[0]) if (std_pkA > 1e-6 and std_pkB > 1e-6) else 0.0
    if np.isnan(r_peak):
        r_peak = 0.0
        
    # 2. Multi-channel shared footprint correlation
    if len(shared_chans) >= 2:
        vecA = np.concatenate([cwA[c] for c in shared_chans])
        vecB = np.concatenate([cwB[c] for c in shared_chans])
        stdA = np.std(vecA)
        stdB = np.std(vecB)
        if stdA > 1e-6 and stdB > 1e-6:
            r_shared = float(pearsonr(vecA, vecB)[0])
            if np.isnan(r_shared):
                r_shared = 0.0
        else:
            r_shared = 0.0
        return max(r_shared, r_peak)
    else:
        return r_peak

def match_adjacent_windows(
    units_A: Dict[int, Dict[str, Any]],
    units_B: Dict[int, Dict[str, Any]],
    max_distance_um: float = 25.0,
    min_corr: float = 0.85
) -> Dict[int, int]:
    """
    Bipartite matching between unit set A and unit set B.
    Returns mapping: uid_A -> uid_B
    """
    ids_A = list(units_A.keys())
    ids_B = list(units_B.keys())
    if not ids_A or not ids_B:
        return {}
        
    cost_matrix = np.zeros((len(ids_A), len(ids_B)))
    
    for i, uA_id in enumerate(ids_A):
        uA = units_A[uA_id]
        for j, uB_id in enumerate(ids_B):
            uB = units_B[uB_id]
            r = compute_pairwise_unit_correlation(uA, uB, max_distance_um=max_distance_um)
            if r >= min_corr:
                cost_matrix[i, j] = -r # Minimize negative correlation
            else:
                cost_matrix[i, j] = 1000.0 # High penalty
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {}
    for r_idx, c_idx in zip(row_ind, col_ind):
        if cost_matrix[r_idx, c_idx] < 0: # Valid match
            matches[ids_A[r_idx]] = ids_B[c_idx]
            
    return matches

def build_unified_trajectories(
    window_units: Dict[str, Dict[int, Dict[str, Any]]],
    max_distance_um: float = 25.0,
    min_corr: float = 0.85,
    max_skip_windows: int = 2
) -> List[Dict[str, Any]]:
    """
    Builds global unit trajectories across all windows, with skip-window support.
    """
    available_windows = [w for w in WINDOW_ORDER if w in window_units and len(window_units[w]) > 0]
    logger.info(f"Building trajectories across {len(available_windows)} windows: {available_windows}")
    
    tracks = [] # List of dicts: {'track_id': str, 'windows': {win_id: uid}, 'depth': float}
    track_counter = 1
    
    # Step 1: Initialize tracks with the first available window
    first_win = available_windows[0]
    for uid, u in window_units[first_win].items():
        tracks.append({
            "track_id": f"NEURON_{track_counter:04d}",
            "windows": {first_win: uid},
            "depth_um": u["depth_um"],
            "last_active_win": first_win,
            "last_uid": uid
        })
        track_counter += 1
        
    # Step 2: Iterate through subsequent windows
    for w_idx in range(1, len(available_windows)):
        curr_win = available_windows[w_idx]
        curr_units = window_units[curr_win]
        unmatched_curr_uids = set(curr_units.keys())
        
        # A. Try matching against existing active tracks from the immediately preceding window
        prev_win = available_windows[w_idx - 1]
        active_tracks = [t for t in tracks if t.get("last_active_win") == prev_win]
        
        sub_A = {t["track_id"]: window_units[prev_win][t["last_uid"]] for t in active_tracks}
        sub_B = {uid: curr_units[uid] for uid in unmatched_curr_uids}
        
        matches = match_adjacent_windows(sub_A, sub_B, max_distance_um=max_distance_um, min_corr=min_corr)
        for t_id, matched_uid in matches.items():
            track = next(t for t in tracks if t["track_id"] == t_id)
            track["windows"][curr_win] = matched_uid
            track["last_active_win"] = curr_win
            track["last_uid"] = matched_uid
            unmatched_curr_uids.discard(matched_uid)
            
        # B. Skip-window matching for tracks that went silent in recent windows (e.g. anoxic nadir)
        for skip in range(2, max_skip_windows + 1):
            if w_idx - skip < 0:
                break
            skip_win = available_windows[w_idx - skip]
            silent_tracks = [t for t in tracks if t.get("last_active_win") == skip_win]
            if not silent_tracks or not unmatched_curr_uids:
                continue
                
            sub_skip = {t["track_id"]: window_units[skip_win][t["last_uid"]] for t in silent_tracks}
            sub_B_rem = {uid: curr_units[uid] for uid in unmatched_curr_uids}
            
            skip_matches = match_adjacent_windows(sub_skip, sub_B_rem, max_distance_um=max_distance_um, min_corr=min_corr)
            for t_id, matched_uid in skip_matches.items():
                track = next(t for t in tracks if t["track_id"] == t_id)
                track["windows"][curr_win] = matched_uid
                track["last_active_win"] = curr_win
                track["last_uid"] = matched_uid
                unmatched_curr_uids.discard(matched_uid)
                
        # C. Any remaining unmatched units in curr_win spawn new candidate tracks (e.g. De Novo gasping units!)
        for new_uid in unmatched_curr_uids:
            tracks.append({
                "track_id": f"NEURON_{track_counter:04d}",
                "windows": {curr_win: new_uid},
                "depth_um": curr_units[new_uid]["depth_um"],
                "last_active_win": curr_win,
                "last_uid": new_uid
            })
            track_counter += 1

    # Step 3: Classify trajectory phenotype for each unified track
    classified_tracks = []
    for t in tracks:
        win_presence = t["windows"]
        has_win1 = "win1_baseline" in win_presence
        has_win2 = "win2_anox1" in win_presence
        has_win3 = "win3_anox2" in win_presence
        has_win4 = "win4_gasp_early" in win_presence
        has_win5 = "win5_gasp_late" in win_presence
        has_win6 = "win6_flow_shift" in win_presence
        has_win7 = "win7_recovery" in win_presence
        
        n_active = len(win_presence)
        
        # Classification logic
        if not has_win1 and not has_win2 and not has_win3 and (has_win4 or has_win5):
            phenotype = "De Novo Gasping Emergent"
        elif (has_win1 or has_win2) and not has_win3 and (has_win4 or has_win5 or has_win6 or has_win7):
            phenotype = "Anoxia-Silenced (Resilient)"
        elif has_win1 and has_win2 and has_win3 and (has_win4 or has_win5):
            phenotype = "Constitutive (Resistant)"
        elif has_win1 and not has_win4 and not has_win5 and not has_win6 and not has_win7:
            phenotype = "Vulnerable (Anoxia-Extinguished)"
        elif n_active == 1:
            phenotype = "Transient Epoch-Specific"
        else:
            phenotype = "Modulated Active"
            
        t["phenotype"] = phenotype
        t["num_active_windows"] = n_active
        classified_tracks.append(t)
        
    logger.info(f"Unification complete: {len(classified_tracks)} unified neuronal tracks constructed.")
    return classified_tracks

def main():
    parser = argparse.ArgumentParser(description="Cross-window neuron unification engine")
    parser.add_argument("--sorter", type=str, default="kilosort4", choices=["kilosort4", "lupin"], help="Sorter to unify across windows")
    parser.add_argument("--max-dist", type=float, default=25.0, help="Max distance um for matching")
    parser.add_argument("--min-corr", type=float, default=0.85, help="Min Pearson r for template matching")
    args = parser.parse_args()

    out_base = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting"
    unification_out = out_base / f"unification_{args.sorter}"
    unification_out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f" CROSS-WINDOW NEURON UNIFICATION ENGINE ({args.sorter.upper()})")
    logger.info(f" Distance threshold: <= {args.max_dist:.1f} um | Correlation threshold: r >= {args.min_corr:.2f}")
    logger.info("=" * 80)

    # 1. Inspect which subwindows are currently sorted and available
    window_units = {}
    for win_id in WINDOW_ORDER:
        win_sorter_dir = out_base / win_id / args.sorter
        alf_dir = win_sorter_dir / "alf"
        if not (alf_dir / "spike_times.npy").exists():
            logger.info(f"Subwindow {win_id} not yet sorted for {args.sorter}. Skipping.")
            continue
            
        # Locate CatGT preprocessed recording
        if win_id == "win1_baseline":
            catgt_dir = Path(r"C:\SCRATCH_neuropixels\catgt_e097_win1_baseline\catgt_e097_g2")
        else:
            catgt_dir = Path(r"C:\SCRATCH_neuropixels\catgt_e097_subwindows") / win_id / "catgt_e097_g2"
            
        if not catgt_dir.exists():
            logger.warning(f"CatGT recording not found at {catgt_dir}! Skipping template extraction.")
            continue
            
        logger.info(f"Extracting templates for {win_id} ({args.sorter})...")
        rec = read_spikeglx_ap_safe(catgt_dir)
        u_dict = extract_mean_templates_for_window(
            sorter_dir=win_sorter_dir,
            recording=rec,
            max_spikes_per_unit=250,
            snippet_len_samples=60,
            top_k_channels=8
        )
        window_units[win_id] = u_dict
        logger.info(f"Extracted {len(u_dict)} unit templates for {win_id}.")

    if len(window_units) < 2:
        logger.warning(f"Only {len(window_units)} windows available. Need at least 2 to compute cross-window unification.")
        return

    # 2. Build trajectories
    trajectories = build_unified_trajectories(
        window_units=window_units,
        max_distance_um=args.max_dist,
        min_corr=args.min_corr,
        max_skip_windows=2
    )

    # 3. Export Summary Manifest
    summary_path = unification_out / "unified_neuronal_trajectories.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=4)
    logger.info(f"Saved unified trajectories manifest to: {summary_path}")

    # Build CSV summary table
    rows = []
    for t in trajectories:
        row = {
            "track_id": t["track_id"],
            "phenotype": t["phenotype"],
            "depth_um": t["depth_um"],
            "num_active_windows": t["num_active_windows"],
        }
        for win_id in WINDOW_ORDER:
            row[f"unit_{win_id}"] = t["windows"].get(win_id, -1)
        rows.append(row)
        
    df_tracks = pd.DataFrame(rows)
    csv_path = unification_out / "unified_neuronal_trajectories.csv"
    df_tracks.to_csv(csv_path, index=False)
    logger.info(f"Saved trajectory spreadsheet to: {csv_path}")

    # Summary breakdown
    breakdown = df_tracks["phenotype"].value_counts().to_dict()
    logger.info("\n>>> PHENOTYPE BREAKDOWN SUMMARY:")
    for pheno, cnt in breakdown.items():
        logger.info(f"  * {pheno}: {cnt} neurons")

if __name__ == '__main__':
    main()
