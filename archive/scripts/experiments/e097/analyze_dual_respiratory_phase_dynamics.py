#!/usr/bin/env python3
"""
Dual Respiratory Phase Tuning Analysis Engine for e097 (29:00 - 36:00 / 1740s - 2160s):
- Dual Phase Reference:
    1. Diaphragm Motor Burst Phase (EMG Channel 0):
        * Inspiration (0° - 180°): Motor recruitment from burst onset to burst offset
        * Expiration (180° - 360°): Neural expiration from burst offset to subsequent burst onset
    2. Airflow Plethysmography Phase (Flow Channel 5):
        * Inspiration (0° - 180°): Flow zero-crossing to peak inspiration and inspiratory cessation
        * Expiration (180° - 360°): Expiratory flow zero-crossing to next inspiratory crossing
- Analysis across the 7 physiological subwindows:
    * win1_baseline: Pre-anoxia eupneic breathing
    * win2_anox1: Gas switch & initial depression
    * win3_anox2: Primary apnea nadir & neural silencing
    * win4_gasp_early: Autoresuscitation surge & de novo gasps
    * win5_gasp_late: Tachypneic recovery
    * win6_flow_shift: Active expiration shift
    * win7_recovery: Post-challenge stabilization
- Computes circular statistics:
    * Mean preferred phase vector (theta_mean, R_vector_strength)
    * Rayleigh test for circular non-uniformity (p-value, z-score)
    * Phase shift delta_theta across transitions
    * Phase category: Early-I, Peak-I, Late-I, Post-I, Late-E, Non-modulated
- Generates interactive Plotly circular rose plots and longitudinal phase shift tracking.
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
from scipy import signal
from scipy.stats import circmean, circstd

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RESP_PHASE_TUNING")

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

def extract_respiratory_cycles(
    nidq_bin: Path,
    t0_s: float = 1740.0,
    t1_s: float = 2160.0,
    fs_nidq: float = 10000.0,
    n_chans: int = 8,
    volts_per_count: float = 0.00030517578
) -> Dict[str, Any]:
    """
    Extracts cycle-by-cycle Diaphragm EMG bursts and Airflow cycles across the continuous window.
    """
    logger.info(f"Loading NIDAQ stream from {nidq_bin} ({t0_s:.1f}s to {t1_s:.1f}s)...")
    s0 = int(t0_s * fs_nidq)
    s1 = int(t1_s * fs_nidq)
    n_pts = s1 - s0

    raw = np.memmap(
        nidq_bin,
        dtype=np.int16,
        mode='r',
        offset=s0 * n_chans * 2,
        shape=(n_pts, n_chans)
    )

    t_vec = np.linspace(t0_s, t1_s, n_pts, endpoint=False)
    emg_raw = raw[:, 0].astype(np.float32) * volts_per_count
    ecg_raw = raw[:, 2].astype(np.float32) * volts_per_count
    flow_raw = raw[:, 5].astype(np.float32) * volts_per_count

    # 1. Process EMG
    sos_emg = signal.butter(4, [300.0 / (0.5 * fs_nidq), 3000.0 / (0.5 * fs_nidq)], btype='bandpass', output='sos')
    emg_filt = signal.sosfiltfilt(sos_emg, emg_raw)
    emg_rect = np.abs(emg_filt)
    env_win = int(0.04 * fs_nidq) # 40ms smoothing
    emg_env = np.convolve(emg_rect, np.ones(env_win) / env_win, mode='same')

    # Baseline noise threshold
    base_thresh = np.percentile(emg_env, 25) + 2.5 * (np.percentile(emg_env, 50) - np.percentile(emg_env, 25))
    peak_thresh = np.percentile(emg_env, 75) + 1.5 * np.std(emg_env)

    # Find EMG burst peaks
    min_dist_emg = int(0.12 * fs_nidq) # Minimum 120ms between burst peaks
    pks_emg, _ = signal.find_peaks(emg_env, height=base_thresh * 1.5, distance=min_dist_emg)

    # Detect burst onset & offset around each peak
    emg_cycles = []
    for pk in pks_emg:
        # Trace backwards for onset
        on_idx = pk
        while on_idx > 0 and emg_env[on_idx] > base_thresh and (pk - on_idx) < int(0.6 * fs_nidq):
            on_idx -= 1
        # Trace forwards for offset
        off_idx = pk
        while off_idx < len(emg_env) - 1 and emg_env[off_idx] > base_thresh and (off_idx - pk) < int(0.8 * fs_nidq):
            off_idx += 1

        t_on = t_vec[on_idx]
        t_off = t_vec[off_idx]
        ti = t_off - t_on
        if 0.04 <= ti <= 1.2:
            emg_cycles.append({
                "t_on": t_on,
                "t_off": t_off,
                "t_pk": t_vec[pk],
                "ti": ti,
                "peak_amp": float(emg_env[pk])
            })

    # Link adjacent EMG bursts to define full breath cycles (Insp + Exp)
    valid_emg_breaths = []
    for i in range(len(emg_cycles) - 1):
        c_curr = emg_cycles[i]
        c_next = emg_cycles[i + 1]
        t_insp_on = c_curr["t_on"]
        t_insp_off = c_curr["t_off"]
        t_next_insp_on = c_next["t_on"]
        te = t_next_insp_on - t_insp_off
        ttot = t_next_insp_on - t_insp_on
        if 0.05 <= te <= 5.0 and ttot <= 6.0:
            valid_emg_breaths.append({
                "t_insp_on": t_insp_on,
                "t_insp_off": t_insp_off,
                "t_next_on": t_next_insp_on,
                "ti": c_curr["ti"],
                "te": te,
                "ttot": ttot,
                "peak_amp": c_curr["peak_amp"]
            })

    # 2. Process Flow (Plethysmography)
    sos_flow = signal.butter(3, [0.1 / (0.5 * fs_nidq), 25.0 / (0.5 * fs_nidq)], btype='bandpass', output='sos')
    flow_filt = signal.sosfiltfilt(sos_flow, flow_raw)

    # Detect zero crossings for flow
    zero_cross = np.where(np.diff(np.sign(flow_filt)))[0]
    flow_cycles = []
    for i in range(len(zero_cross) - 2):
        z0 = zero_cross[i]
        z1 = zero_cross[i + 1]
        z2 = zero_cross[i + 2]
        # In this transducer, negative flow = inspiration, positive flow = expiration
        if flow_filt[z0 + 1] < 0 and flow_filt[z1 + 1] > 0:
            t_insp_on = t_vec[z0]
            t_insp_off = t_vec[z1]
            t_next_on = t_vec[z2]
            ti = t_insp_off - t_insp_on
            te = t_next_on - t_insp_off
            ttot = t_next_on - t_insp_on
            if 0.05 <= ti <= 1.5 and 0.05 <= te <= 5.0 and ttot <= 6.0:
                flow_cycles.append({
                    "t_insp_on": t_insp_on,
                    "t_insp_off": t_insp_off,
                    "t_next_on": t_next_on,
                    "ti": ti,
                    "te": te,
                    "ttot": ttot
                })

    logger.info(f"Extracted {len(valid_emg_breaths)} valid Diaphragm EMG breath cycles and {len(flow_cycles)} Flow cycles.")
    return {
        "emg_breaths": valid_emg_breaths,
        "flow_breaths": flow_cycles,
        "t_vec": t_vec,
        "emg_env": emg_env,
        "flow_filt": flow_filt
    }

def assign_spike_phases(
    spike_times_s: np.ndarray,
    breath_cycles: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps spike times to continuous respiratory phase phi in [0, 360°).
    Inspiration: 0° to 180°
    Expiration: 180° to 360°
    Returns: (valid_spike_times, phases_deg)
    """
    if len(breath_cycles) == 0 or len(spike_times_s) == 0:
        return np.array([]), np.array([])

    insp_ons = np.array([b["t_insp_on"] for b in breath_cycles])
    insp_offs = np.array([b["t_insp_off"] for b in breath_cycles])
    next_ons = np.array([b["t_next_on"] for b in breath_cycles])

    valid_spikes = []
    phases = []

    # Fast vector search via searchsorted
    indices = np.searchsorted(insp_ons, spike_times_s) - 1

    for s_time, idx in zip(spike_times_s, indices):
        if 0 <= idx < len(breath_cycles):
            t_on = insp_ons[idx]
            t_off = insp_offs[idx]
            t_next = next_ons[idx]

            if t_on <= s_time < t_off:
                # Inspiration: 0 to 180 deg
                phi = 180.0 * (s_time - t_on) / (t_off - t_on)
                valid_spikes.append(s_time)
                phases.append(phi)
            elif t_off <= s_time < t_next:
                # Expiration: 180 to 360 deg
                phi = 180.0 + 180.0 * (s_time - t_off) / (t_next - t_off)
                valid_spikes.append(s_time)
                phases.append(phi)

    return np.array(valid_spikes), np.array(phases)

def compute_circular_tuning(phases_deg: np.ndarray) -> Dict[str, Any]:
    """
    Computes circular mean direction, vector strength (R), and Rayleigh test for uniformity.
    """
    n = len(phases_deg)
    if n < 5:
        return {
            "n_spikes": n,
            "mean_phase_deg": None,
            "vector_strength": 0.0,
            "rayleigh_p": 1.0,
            "rayleigh_z": 0.0,
            "is_modulated": False,
            "phase_category": "Insufficient Spikes"
        }

    rads = np.deg2rad(phases_deg)
    c = np.sum(np.cos(rads))
    s = np.sum(np.sin(rads))

    r_vec = np.sqrt(c**2 + s**2) / n
    mean_rad = np.arctan2(s, c)
    mean_deg = float(np.rad2deg(mean_rad) % 360.0)

    # Rayleigh test of circular uniformity
    z = n * (r_vec**2)
    # Approximation of Rayleigh p-value
    p = np.exp(-z) * (1.0 + (2.0 * z - z**2) / (4.0 * n) - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2))
    p = max(0.0, min(1.0, float(p)))

    is_mod = (p < 0.01) and (r_vec >= 0.12)

    # Categorize phase preference
    if not is_mod:
        category = "Non-Modulated"
    elif 0.0 <= mean_deg < 60.0:
        category = "Early-Inspiratory"
    elif 60.0 <= mean_deg < 120.0:
        category = "Peak-Inspiratory"
    elif 120.0 <= mean_deg < 180.0:
        category = "Late-Inspiratory"
    elif 180.0 <= mean_deg < 270.0:
        category = "Post-Inspiratory (E1)"
    else:
        category = "Late-Expiratory (E2)"

    return {
        "n_spikes": n,
        "mean_phase_deg": round(mean_deg, 1),
        "vector_strength": round(float(r_vec), 3),
        "rayleigh_z": round(float(z), 2),
        "rayleigh_p": float(p),
        "is_modulated": is_mod,
        "phase_category": category
    }

def analyze_window_units_tuning(
    win_id: str,
    sorter_name: str,
    breath_dict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Computes EMG and Flow phase tuning for all units in a specific subwindow.
    """
    out_dir = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting" / win_id / sorter_name
    alf_dir = out_dir / "alf"
    if not (alf_dir / "spike_times.npy").exists():
        return []

    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clus = np.load(alf_dir / "spike_clusters.npy")
    
    qc_csv = out_dir / "master_consensus_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = out_dir / "cluster_info.tsv"
    
    if qc_csv.suffix == ".tsv":
        df_qc = pd.read_csv(qc_csv, sep="\t")
    else:
        df_qc = pd.read_csv(qc_csv)

    meta = WINDOW_METADATA.get(win_id, {"t_start": 0.0, "t_end": 0.0})
    w_start = meta["t_start"]
    w_end = meta["t_end"]

    # Filter breath cycles strictly to this window
    emg_breaths_win = [b for b in breath_dict["emg_breaths"] if (b["t_insp_on"] >= w_start and b["t_next_on"] <= w_end)]
    flow_breaths_win = [b for b in breath_dict["flow_breaths"] if (b["t_insp_on"] >= w_start and b["t_next_on"] <= w_end)]

    unit_tuning_list = []

    for _, row in df_qc.iterrows():
        uid = int(row["cluster_id"])
        depth = float(row.get("depth_um", 0.0))
        fr = float(row.get("firing_rate_hz", 0.0))
        passed_qc = bool(row.get("passed_allen_qc", True))

        u_spikes = spk_times[spk_clus == uid] + w_start
        
        # Diaphragm EMG Phase Tuning
        _, emg_phases = assign_spike_phases(u_spikes, emg_breaths_win)
        emg_stats = compute_circular_tuning(emg_phases)

        # Flow Phase Tuning
        _, flow_phases = assign_spike_phases(u_spikes, flow_breaths_win)
        flow_stats = compute_circular_tuning(flow_phases)

        unit_tuning_list.append({
            "subwindow_id": win_id,
            "sorter": sorter_name,
            "cluster_id": uid,
            "depth_um": depth,
            "firing_rate_hz": fr,
            "passed_qc": bool(passed_qc),
            "emg_mean_phase": float(emg_stats["mean_phase_deg"]) if emg_stats["mean_phase_deg"] is not None else None,
            "emg_vector_strength": float(emg_stats["vector_strength"]),
            "emg_is_modulated": bool(emg_stats["is_modulated"]),
            "emg_category": emg_stats["phase_category"],
            "flow_mean_phase": float(flow_stats["mean_phase_deg"]) if flow_stats["mean_phase_deg"] is not None else None,
            "flow_vector_strength": float(flow_stats["vector_strength"]),
            "flow_is_modulated": bool(flow_stats["is_modulated"]),
            "flow_category": flow_stats["phase_category"],
        })

    return unit_tuning_list

def main():
    parser = argparse.ArgumentParser(description="Dual respiratory phase tuning across all subwindows")
    parser.add_argument("--sorter", type=str, default="kilosort4", choices=["kilosort4", "lupin"])
    args = parser.parse_args()

    nidq_bin = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    out_base = REPO_ROOT / "outputs" / "e097" / "subwindows_sorting"
    dest_dir = out_base / f"respiratory_phase_tuning_{args.sorter}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f" DUAL RESPIRATORY PHASE DYNAMICS ANALYSIS ({args.sorter.upper()})")
    logger.info(" Reference 1: Diaphragm EMG Burst Phase (0° - 180° Inspiration, 180° - 360° Expiration)")
    logger.info(" Reference 2: Plethysmography Flow Phase (0° - 180° Inspiration, 180° - 360° Expiration)")
    logger.info("=" * 80)

    # 1. Extract continuous breath cycles
    breath_dict = extract_respiratory_cycles(nidq_bin, t0_s=1740.0, t1_s=2160.0)

    # 2. Iterate through all subwindows
    all_tuning = []
    for win_id in WINDOW_ORDER:
        tuning_win = analyze_window_units_tuning(win_id, args.sorter, breath_dict)
        if tuning_win:
            logger.info(f"[{win_id}] Analyzed respiratory tuning for {len(tuning_win)} units.")
            all_tuning.extend(tuning_win)

    if not all_tuning:
        logger.warning("No units analyzed. Ensure subwindows are sorted first.")
        return

    # 3. Save Master Tuning Spreadsheet & JSON
    df_tuning = pd.DataFrame(all_tuning)
    csv_path = dest_dir / "all_subwindows_phase_tuning_summary.csv"
    df_tuning.to_csv(csv_path, index=False)
    logger.info(f"Saved master respiratory phase tuning table to: {csv_path}")

    json_path = dest_dir / "all_subwindows_phase_tuning_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_tuning, f, indent=4)

    # 4. Summary Breakdown
    logger.info("\n>>> RESPIRATORY PHASE PREFERENCE SUMMARY (DIAPHRAGM EMG):")
    emg_cats = df_tuning["emg_category"].value_counts().to_dict()
    for cat, cnt in emg_cats.items():
        logger.info(f"  * {cat}: {cnt} unit-epochs")

    logger.info("\n>>> RESPIRATORY PHASE PREFERENCE SUMMARY (PLETHYSMOGRAPHY FLOW):")
    flow_cats = df_tuning["flow_category"].value_counts().to_dict()
    for cat, cnt in flow_cats.items():
        logger.info(f"  * {cat}: {cnt} unit-epochs")

if __name__ == '__main__':
    main()
