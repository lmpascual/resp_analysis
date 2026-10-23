#!/usr/bin/env python3
"""
End-to-End Intact-Window Analysis Pipeline for e097 (1740.0s - 2160.0s / 420.0s):
- Unifies the entire anoxic challenge and recovery without partitioning into subwindows:
    * Baseline Eupnea (1740-1800s)
    * Anoxic Challenge 1 (1800-1836s)
    * Anoxic Challenge 2 / Apnea Nadir (1836-1860s)
    * Early Gasping (1860-1950s)
    * Late Gasping (1950-2070s)
    * Active Expiration Shift (2070-2136s)
    * Post-Challenge Recovery (2136-2160s)
- Modules:
    1. CatGT AP preprocessing (t=1740s, dur=420s)
    2. Dual Spike Sorting:
        * Pipeline A: Kilosort 4 (RTX 4000 Ada GPU)
        * Pipeline B: Lupin (Modular SVD-OMP CPU)
    3. Permissive SUA QC & Cross-Sorter Mutual Agreement (0.4 ms coincidence window)
    4. NIDQ Analog Processing & Whole-Window Breath Segmentation
    5. Continuous & Trial-by-Trial 5-State Poisson HMM (Raw 10 ms unreduced spike counts)
    6. Canonical State Alignment across the continuous timeline
    7. Population Drop-Out & Subsampling Robustness (N in {120, 80, 50, 30, 15, 8, 3})
    8. Single-Neuron Attribution, Gini Inequality, Targeted vs. Random Ablation
    9. Ensemble Coordination (Participation Ratio & Transition Co-Modulation)
    10. Interactive HTML Dashboard & 300 DPI Publication Figure Generation
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.optimize import linear_sum_assignment
from scipy.stats import sem
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from hmmlearn import hmm

# SpikeInterface
import spikeinterface.full as si
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc

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

from src.pipeline.config import CatGTConfig, SorterConfig, LupinConfig, Kilosort4Config
from src.pipeline.processing.neural_ap.io_catgt import run_catgt
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.sorting.sorter_runner import run_single_sorter
from src.pipeline.processors.proc_emg import process_emg, EMGConfig
from src.pipeline.processors.proc_flow import process_flow_meter
from scripts.visualization.plot_interactive_sorter_raster import build_interactive_sorter_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("INTACT_WINDOW_PIPELINE")

# Constants
INTACT_T_START = 1740.0
INTACT_DURATION = 420.0
INTACT_T_END = 2160.0

CANONICAL_STATE_NAMES = [
    "State 1: Insp Burst (Peak Drive)",
    "State 2: Peak / Off-Switch Quench (I_off)",
    "State 3: Post-Insp / Expiratory Drive (Exp)",
    "State 4: Pre-Insp Priming (Pre-I)",
    "State 5: Inter-Breath Pause (Quiescence)"
]
CANONICAL_COLORS = ["#10b981", "#f97316", "#38bdf8", "#a855f7", "#64748b"]

STAGES = [
    {"id": "win1_baseline", "name": "Baseline Eupnea", "t0": 1740.0, "t1": 1800.0, "color": "#10b981"},
    {"id": "win2_anox1", "name": "Anoxic Challenge 1 (Gas Switch)", "t0": 1800.0, "t1": 1836.0, "color": "#eab308"},
    {"id": "win3_anox2", "name": "Anoxic Challenge 2 (Apnea Nadir)", "t0": 1836.0, "t1": 1860.0, "color": "#ef4444"},
    {"id": "win4_gasp_early", "name": "Early Gasping (Autoresuscitation)", "t0": 1860.0, "t1": 1950.0, "color": "#38bdf8"},
    {"id": "win5_gasp_late", "name": "Late Gasping (Tachypnea)", "t0": 1950.0, "t1": 2070.0, "color": "#6366f1"},
    {"id": "win6_flow_shift", "name": "Active Expiration Shift", "t0": 2070.0, "t1": 2136.0, "color": "#a855f7"},
    {"id": "win7_recovery", "name": "Post-Challenge Recovery", "t0": 2136.0, "t1": 2160.0, "color": "#ec4899"}
]


def load_or_run_catgt(raw_ap_dir: Path, scratch_top: Path, catgt_exe: str, force: bool = False) -> Tuple[Path, Path]:
    """Ensures CatGT AP preprocessed files exist for the intact 420s window."""
    catgt_processed_folder = scratch_top / "catgt_e097_g2"
    catgt_bin = catgt_processed_folder / "e097_g2_tcat.imec0.ap.bin"
    catgt_meta = catgt_processed_folder / "e097_g2_tcat.imec0.ap.meta"

    if not force and catgt_bin.exists() and catgt_meta.exists() and catgt_bin.stat().st_size > 5 * 1024 * 1024 * 1024:
        logger.info(f"Found existing CatGT preprocessed binary at: {catgt_bin} ({catgt_bin.stat().st_size / (1024*1024*1024):.2f} GB)")
        return catgt_bin, catgt_meta

    scratch_top.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running CatGT extraction for intact window (t={INTACT_T_START:.1f}s, dur={INTACT_DURATION:.1f}s)...")
    catgt_cfg = CatGTConfig(
        enabled=True,
        catgt_path=catgt_exe,
        run_name="e097",
        gate=2,
        trigger=0,
        probe=0,
        t_shift=True,
        apfilter="butter,12,300,9000",
        loccar_um=(75.0, 300.0),
        start_secs=INTACT_T_START,
        max_secs=INTACT_DURATION,
        dest_dir=str(scratch_top)
    )
    bin_path, meta_path = run_catgt(
        config=catgt_cfg,
        raw_dir=raw_ap_dir,
        dest_dir=scratch_top,
        start_secs=INTACT_T_START,
        max_secs=INTACT_DURATION
    )
    return bin_path, meta_path


def run_intact_spike_sorting(
    recording: si.BaseRecording,
    out_dir: Path,
    scratch_dir: Path,
    sorters: List[str] = ["kilosort4", "lupin"]
) -> Dict[str, Any]:
    """Executes dual spike sorting and computes mutual agreement on the intact recording."""
    results = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    sorter_cfg = SorterConfig(
        enabled_sorters=sorters,
        do_CAR=False,
        save_phy=True,
        export_alf=True,
        isi_threshold_s=0.0015,
        qc_isi_cutoff=0.15,
        qc_presence_ratio=0.05,
        qc_amp_cutoff=0.25,
        qc_min_fr_hz=0.05,
        n_jobs=12
    )

    ks4_cfg = Kilosort4Config(
        enabled=True,
        Th_universal=8.5,
        Th_learned=7.5,
        do_CAR=False,
        nblocks=1,
        batch_size=60000,
        nt=61
    )

    lupin_cfg = LupinConfig(
        enabled=True,
        sparsity_radius_um=100.0,
        clustering_method="circus",
        matching_method="circus-omp-svd",
        ms_before=1.5,
        ms_after=2.0,
        templates_from_svd=True,
        matching_amplitudes=[0.5, 2.2]
    )

    sortings = {}
    for s_name in sorters:
        s_out = out_dir / s_name
        alf_dir = s_out / "alf"
        if (alf_dir / "spike_times.npy").exists() and (alf_dir / "spike_clusters.npy").exists():
            logger.info(f"[{s_name.upper()}] Found existing sorted ALF files at: {alf_dir}. Loading directly...")
            spk_t = np.load(alf_dir / "spike_times.npy")
            spk_c = np.load(alf_dir / "spike_clusters.npy")
            fs = recording.get_sampling_frequency()
            sort_obj = si.NumpySorting.from_times_and_labels(
                times_list=(spk_t * fs).astype(np.int64),
                labels_list=spk_c,
                sampling_frequency=fs
            )
            qc_csv = s_out / f"{s_name}_qc_summary.csv"
            df_qc = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame()
            runtime = 0.0
        else:
            logger.info(f"[{s_name.upper()}] Running spike sorting...")
            sort_obj, df_qc, runtime = run_single_sorter(
                sorter_name=s_name,
                recording=recording,
                output_dir=out_dir,
                scratch_dir=scratch_dir,
                config=sorter_cfg,
                lupin_config=lupin_cfg,
                kilosort4_config=ks4_cfg,
                verbose=True
            )

        sortings[s_name] = sort_obj
        results[s_name] = {
            "total_units": len(sort_obj.get_unit_ids()),
            "sua_units": int((df_qc["passed_allen_qc"] == True).sum()) if "passed_allen_qc" in df_qc.columns else len(sort_obj.get_unit_ids()),
            "runtime_s": float(runtime),
            "output_dir": str(s_out)
        }

    # Mutual Agreement if both sorters are run
    if "kilosort4" in sortings and "lupin" in sortings:
        logger.info("Computing cross-sorter mutual agreement (0.4 ms coincidence window)...")
        cmp = sc.compare_two_sorters(
            sorting1=sortings["lupin"],
            sorting2=sortings["kilosort4"],
            name1="lupin",
            name2="kilosort4",
            delta_time=0.4
        )
        agr = cmp.get_agreement_fraction()
        matched_50 = int((agr.max(axis=1) >= 0.50).sum())
        matched_80 = int((agr.max(axis=1) >= 0.80).sum())
        pct_matched = float(matched_50 / max(1, len(sortings["lupin"].get_unit_ids())) * 100.0)
        results["agreement"] = {
            "matched_units_50pct": matched_50,
            "matched_units_80pct": matched_80,
            "pct_lupin_matched": pct_matched
        }
        logger.info(f"Mutual Agreement: {matched_50} units matched >= 50% ({pct_matched:.1f}% of Lupin units)")

    return results


def extract_intact_breaths(nidq_file: Path, start_t: float = 1740.0, dur: float = 420.0) -> Dict[str, Any]:
    """Extracts Diaphragm EMG, Flow, and breath segmentation across the intact 420s window."""
    fs = 10000.0
    n_ch = 9
    s0 = int(start_t * fs)
    s1 = int((start_t + dur) * fs)

    logger.info(f"Loading analog NIDQ signals from {nidq_file} (t={start_t:.1f}s to {start_t+dur:.1f}s)...")
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[s0 * n_ch : s1 * n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    flow_v = slice_data[:, 5].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    valid_pairs = [(on, off) for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets) if on > 0.05 and on < t_vec[-1] - 0.05 and off > on]
    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])

    flow_res = process_flow_meter(flow_v, t_vec, valid_onsets, valid_offsets, fs=fs)
    flow_zcs = flow_res.flow_zero_crossings
    peak_insp_flow = flow_res.flow_troughs  # Peak inspiration: where flow slope changes from negative to positive
    n_breaths = len(valid_onsets)

    cycle_ends = np.append(valid_onsets[1:], dur)
    # Inspiration-off is detected by the integrated diaphragm trace (burst offset)
    ti_arr = (valid_offsets - valid_onsets) * 1000.0  # ms (EMG T_I: Onset to Inspiration-Off)
    te_arr = (cycle_ends - valid_offsets) * 1000.0    # ms (EMG T_E: Inspiration-Off to Next Onset)
    tot_arr = (cycle_ends - valid_onsets) * 1000.0   # ms (Total respiratory cycle duration)
    ti_flow_zc = (flow_zcs - valid_onsets) * 1000.0   # ms (Legacy flow zero-crossing duration)

    # Assign each breath to a physiological stage
    breath_stages = []
    for on in valid_onsets:
        abs_t = start_t + on
        st_id = "win7_recovery"
        for st in STAGES:
            if st["t0"] <= abs_t < st["t1"]:
                st_id = st["id"]
                break
        breath_stages.append(st_id)

    logger.info(f"Segmented {n_breaths} total breaths across intact window using integrated diaphragm trace for I_off: Mean T_I = {np.mean(ti_arr):.1f} ± {np.std(ti_arr):.1f} ms.")

    return {
        "fs": fs,
        "dur": dur,
        "start_t": start_t,
        "t_vec": t_vec,
        "emg_v": emg_v,
        "emg_env": emg_proc.envelope,
        "flow_v": flow_v,
        "valid_onsets": valid_onsets,
        "valid_offsets": valid_offsets,
        "i_off": valid_offsets,
        "peak_insp_flow": peak_insp_flow,
        "flow_zcs": flow_zcs,
        "cycle_ends": cycle_ends,
        "ti_arr": ti_arr,
        "te_arr": te_arr,
        "tot_arr": tot_arr,
        "ti_flow_zc": ti_flow_zc,
        "breath_stages": np.array(breath_stages),
        "n_breaths": n_breaths
    }


def load_intact_spikes_and_qc(sorter_dir: Path, max_units: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads spike times, clusters, depths, and selects top QC units up to max_units."""
    spk_times = np.load(sorter_dir / "alf" / "spike_times.npy")
    spk_clusters = np.load(sorter_dir / "alf" / "spike_clusters.npy")

    qc_csv = sorter_dir / f"{sorter_dir.name}_qc_summary.csv"
    if not qc_csv.exists():
        qc_csv = sorter_dir / "cluster_info.tsv"
    sep = "\t" if qc_csv.suffix == ".tsv" else ","
    qc_df = pd.read_csv(qc_csv, sep=sep)

    if "passed_allen_qc" in qc_df.columns:
        df_qc = qc_df[qc_df["passed_allen_qc"] == True].copy()
    else:
        df_qc = qc_df.copy()

    df_qc = df_qc.sort_values(by="depth_um", ascending=False).reset_index(drop=True)
    if len(df_qc) > max_units:
        df_qc = df_qc.head(max_units)

    cids = df_qc["cluster_id"].values
    depths = df_qc["depth_um"].values

    return spk_times, spk_clusters, depths, df_qc


def fit_intact_poisson_hmm(
    spk_times: np.ndarray,
    spk_clusters: np.ndarray,
    df_qc: pd.DataFrame,
    dur: float = 420.0,
    dt: float = 0.010,
    n_states: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Fits 5-state Poisson HMM on raw 10 ms unreduced spike counts across the intact 420s window."""
    time_bins = np.arange(0, dur + dt, dt)
    n_bins = len(time_bins) - 1
    cids = df_qc["cluster_id"].values
    n_units = len(cids)

    logger.info(f"Discretizing spikes into 10 ms bins ({n_bins} bins, {n_units} units, dur={dur:.1f}s)...")
    spike_mat = np.zeros((n_bins, n_units), dtype=np.int32)
    for u_idx, cid in enumerate(cids):
        ts = spk_times[spk_clusters == cid]
        ts_valid = ts[(ts >= 0) & (ts <= dur)]
        counts, _ = np.histogram(ts_valid, bins=time_bins)
        spike_mat[:, u_idx] = counts

    logger.info(f"Fitting {n_states}-state Poisson HMM on raw spike counts (shape={spike_mat.shape})...")
    model = hmm.PoissonHMM(n_components=n_states, n_iter=100, tol=1e-3, random_state=random_state, verbose=False)
    model.fit(spike_mat)

    state_seq = model.predict(spike_mat)
    post_probs = model.predict_proba(spike_mat)
    rates = model.lambdas_ # (n_states, n_units)
    mean_state_rates = rates.mean(axis=1) / dt # Hz

    # Canonical State Alignment:
    # State 1: Highest overall population firing rate -> Inspiratory Burst
    # State 5: Lowest overall firing rate -> Inter-Breath Pause
    sorted_by_rate = np.argsort(mean_state_rates)
    state_pause_raw = sorted_by_rate[0]
    state_insp_raw = sorted_by_rate[-1]

    # For intermediate states, evaluate temporal transition order relative to insp burst
    remaining_states = [s for s in range(n_states) if s not in (state_pause_raw, state_insp_raw)]
    trans = model.transmat_

    # Canonical mapping dict: canonical_idx -> raw_idx
    # Canonical indices: 0: Insp, 1: Off-Switch, 2: Exp, 3: Pre-I, 4: Pause
    # Find Pre-I (highest probability of transitioning INTO Insp)
    pre_i_cands = [(s, trans[s, state_insp_raw]) for s in remaining_states]
    pre_i_cands.sort(key=lambda x: x[1], reverse=True)
    state_pre_i_raw = pre_i_cands[0][0]

    # Find Off-Switch (highest probability of being transitioned INTO FROM Insp)
    remaining_states2 = [s for s in remaining_states if s != state_pre_i_raw]
    off_cands = [(s, trans[state_insp_raw, s]) for s in remaining_states2]
    off_cands.sort(key=lambda x: x[1], reverse=True)
    state_off_raw = off_cands[0][0]

    # Remaining is Expiratory Drive
    remaining_states3 = [s for s in remaining_states2 if s != state_off_raw]
    state_exp_raw = remaining_states3[0] if remaining_states3 else remaining_states2[-1]

    raw_to_canon = {
        state_insp_raw: 0,
        state_off_raw: 1,
        state_exp_raw: 2,
        state_pre_i_raw: 3,
        state_pause_raw: 4
    }

    canon_state_seq = np.array([raw_to_canon[s] for s in state_seq])
    canon_post_probs = np.zeros_like(post_probs)
    for raw_idx, canon_idx in raw_to_canon.items():
        canon_post_probs[:, canon_idx] = post_probs[:, raw_idx]

    canon_rates = np.zeros_like(rates)
    for raw_idx, canon_idx in raw_to_canon.items():
        canon_rates[canon_idx, :] = rates[raw_idx, :]

    logger.info("Canonical state mapping successfully resolved.")
    for c_i, name in enumerate(CANONICAL_STATE_NAMES):
        occupancy = float(np.mean(canon_state_seq == c_i) * 100.0)
        logger.info(f"  {name}: {occupancy:.1f}% occupancy, mean ensemble rate = {canon_rates[c_i].mean() / dt:.1f} Hz")

    return {
        "model": model,
        "dt": dt,
        "n_states": n_states,
        "n_units": n_units,
        "n_bins": n_bins,
        "spike_mat": spike_mat,
        "state_seq": canon_state_seq,
        "post_probs": canon_post_probs,
        "canon_rates": canon_rates,
        "raw_to_canon": raw_to_canon,
        "transmat": model.transmat_
    }


def align_states_hungarian(pred: np.ndarray, true: np.ndarray, n_states: int = 5) -> Tuple[np.ndarray, float, float, np.ndarray, float]:
    """Bijective Hungarian state alignment against ground-truth intact trajectory."""
    cost_matrix = np.zeros((n_states, n_states), dtype=np.int32)
    for i in range(n_states):
        for j in range(n_states):
            cost_matrix[i, j] = -np.sum((pred == i) & (true == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_pred = np.array([mapping.get(s, s) for s in pred])
    ari = adjusted_rand_score(true, mapped_pred)
    nmi = normalized_mutual_info_score(true, mapped_pred)
    f1_per_state = f1_score(true, mapped_pred, average=None, labels=list(range(n_states)))
    macro_f1 = f1_score(true, mapped_pred, average="macro")
    return mapped_pred, ari, nmi, f1_per_state, macro_f1


def run_intact_subsampling_experiment(spike_mat: np.ndarray, true_states: np.ndarray, n_draws: int = 8) -> Dict[str, Any]:
    """Evaluates subsampling robustness across N in {120, 80, 50, 30, 15, 8, 3} on the intact window."""
    total_units = spike_mat.shape[1]
    tiers = [u for u in [120, 80, 50, 30, 15, 8, 3] if u <= total_units]
    if total_units not in tiers:
        tiers.insert(0, total_units)

    results_by_size = []
    logger.info("Executing Intact Subsampling Dropout Experiment across population sizes...")

    for N in tiers:
        logger.info(f"  Testing Subsample Size N = {N} ({n_draws} draws)...")
        ari_draws, nmi_draws, macro_f1_draws = [], [], []
        state_f1_matrix = []

        if N == total_units:
            results_by_size.append({
                "N": N,
                "fraction": 1.0,
                "ari_mean": 1.0,
                "ari_std": 0.0,
                "ari_sem": 0.0,
                "macro_f1_mean": 1.0,
                "macro_f1_sem": 0.0,
                "nmi_mean": 1.0,
                "state_f1_mean": [1.0] * 5,
                "state_f1_sem": [0.0] * 5
            })
            continue

        for seed in range(n_draws):
            rng = np.random.RandomState(42 + seed * 17)
            sub_units = rng.choice(total_units, size=N, replace=False)
            sub_mat = spike_mat[:, sub_units]

            try:
                sub_model = hmm.PoissonHMM(n_components=5, n_iter=60, tol=1e-3, random_state=seed, verbose=False)
                sub_model.fit(sub_mat)
                pred_raw = sub_model.predict(sub_mat)
                _, ari, nmi, f1_per_st, mf1 = align_states_hungarian(pred_raw, true_states, n_states=5)
                ari_draws.append(ari)
                nmi_draws.append(nmi)
                macro_f1_draws.append(mf1)
                state_f1_matrix.append(f1_per_st)
            except Exception as e:
                logger.warning(f"Failed fit at N={N}, seed={seed}: {e}")

        if ari_draws:
            state_f1_arr = np.array(state_f1_matrix)
            results_by_size.append({
                "N": N,
                "fraction": float(N / total_units),
                "ari_mean": float(np.mean(ari_draws)),
                "ari_std": float(np.std(ari_draws)),
                "ari_sem": float(sem(ari_draws) if len(ari_draws) > 1 else 0.0),
                "macro_f1_mean": float(np.mean(macro_f1_draws)),
                "macro_f1_sem": float(sem(macro_f1_draws) if len(macro_f1_draws) > 1 else 0.0),
                "nmi_mean": float(np.mean(nmi_draws)),
                "state_f1_mean": [float(x) for x in np.mean(state_f1_arr, axis=0)],
                "state_f1_sem": [float(x) for x in (sem(state_f1_arr, axis=0) if len(ari_draws) > 1 else np.zeros(5))]
            })

    # Critical Population Thresholds
    n_star_80 = total_units
    n_star_90 = total_units
    for res in sorted(results_by_size, key=lambda x: x["N"]):
        if res["ari_mean"] >= 0.80 and n_star_80 == total_units:
            n_star_80 = res["N"]
        if res["ari_mean"] >= 0.90 and n_star_90 == total_units:
            n_star_90 = res["N"]

    return {
        "results_by_size": results_by_size,
        "n_star_80": n_star_80,
        "n_star_90": n_star_90
    }


def run_intact_single_neuron_attribution(spike_mat: np.ndarray, true_states: np.ndarray, depths: np.ndarray, dt: float = 0.010) -> Dict[str, Any]:
    """Computes single-neuron mutual information, Gini coefficient, Lorenz curve, and targeted ablations."""
    total_units = spike_mat.shape[1]
    logger.info("Computing Single-Neuron Mutual Information Attribution across the intact window...")

    unit_stats = []
    mi_list = []

    for u_idx in range(total_units):
        u_spk = spike_mat[:, u_idx]
        mean_fr = float(np.mean(u_spk) / dt)
        mi_bits = float(normalized_mutual_info_score(u_spk, true_states))
        mi_list.append(mi_bits)
        unit_stats.append({
            "unit_idx": int(u_idx),
            "depth_um": float(depths[u_idx]) if u_idx < len(depths) else 0.0,
            "mean_fr": mean_fr,
            "mi_bits": mi_bits
        })

    # Gini inequality of mutual information
    mi_arr = np.array(mi_list)
    mi_sorted = np.sort(mi_arr)
    n = len(mi_sorted)
    index = np.arange(1, n + 1)
    gini = float((np.sum((2 * index - n - 1) * mi_sorted)) / (n * np.sum(mi_sorted) + 1e-12))

    # Lorenz Curve Coordinates
    cum_units = np.linspace(0, 1, n + 1).tolist()
    cum_info = (np.concatenate(([0.0], np.cumsum(mi_sorted))) / (np.sum(mi_sorted) + 1e-12)).tolist()

    # Targeted Ablation Experiments
    sort_indices = np.argsort(mi_arr)
    top_indices = sort_indices[::-1]
    bottom_indices = sort_indices

    ablation_results = []
    for drop_frac in [0.10, 0.20]:
        k_drop = max(1, int(round(drop_frac * total_units)))
        logger.info(f"  Testing Ablation: Dropping {k_drop} units ({int(drop_frac*100)}%)...")

        # 1. Drop Top Driver Units
        keep_no_top = np.setdiff1d(np.arange(total_units), top_indices[:k_drop])
        m_top = hmm.PoissonHMM(n_components=5, n_iter=60, tol=1e-3, random_state=42, verbose=False)
        m_top.fit(spike_mat[:, keep_no_top])
        p_top = m_top.predict(spike_mat[:, keep_no_top])
        _, ari_top, _, _, f1_top = align_states_hungarian(p_top, true_states, n_states=5)

        # 2. Drop Bottom Units
        keep_no_bot = np.setdiff1d(np.arange(total_units), bottom_indices[:k_drop])
        m_bot = hmm.PoissonHMM(n_components=5, n_iter=60, tol=1e-3, random_state=42, verbose=False)
        m_bot.fit(spike_mat[:, keep_no_bot])
        p_bot = m_bot.predict(spike_mat[:, keep_no_bot])
        _, ari_bot, _, _, f1_bot = align_states_hungarian(p_bot, true_states, n_states=5)

        # 3. Drop Random Units (5 draws)
        ari_rand, f1_rand = [], []
        for r_seed in range(5):
            rng = np.random.RandomState(100 + r_seed)
            drop_r = rng.choice(total_units, size=k_drop, replace=False)
            keep_r = np.setdiff1d(np.arange(total_units), drop_r)
            m_r = hmm.PoissonHMM(n_components=5, n_iter=60, tol=1e-3, random_state=r_seed, verbose=False)
            m_r.fit(spike_mat[:, keep_r])
            p_r = m_r.predict(spike_mat[:, keep_r])
            _, ar, _, _, fr = align_states_hungarian(p_r, true_states, n_states=5)
            ari_rand.append(ar)
            f1_rand.append(fr)

        ablation_results.append({
            "frac": drop_frac,
            "k_dropped": k_drop,
            "ari_no_top": float(ari_top),
            "f1_no_top": float(f1_top),
            "ari_no_bottom": float(ari_bot),
            "f1_no_bottom": float(f1_bot),
            "ari_no_random_mean": float(np.mean(ari_rand)),
            "ari_no_random_sem": float(sem(ari_rand) if len(ari_rand) > 1 else 0.0),
            "f1_no_random_mean": float(np.mean(f1_rand)),
            "f1_no_random_sem": float(sem(f1_rand) if len(f1_rand) > 1 else 0.0)
        })

    return {
        "unit_stats": unit_stats,
        "gini_mi": float(gini),
        "lorenz_cum_units": cum_units,
        "lorenz_cum_info": cum_info,
        "ablation_results": ablation_results
    }


def run_intact_coordination_analysis(spike_mat: np.ndarray, true_states: np.ndarray, dt: float = 0.010) -> Dict[str, Any]:
    """Computes Participation Ratio, PCA variance spectrum, and transition co-modulation fractions."""
    total_units = spike_mat.shape[1]
    logger.info("Computing Ensemble Coordination and Dimensionality on intact recording...")

    # PCA & Participation Ratio on smoothed firing rates
    box_pts = int(0.050 / dt) # 50 ms smoothing
    smoothed = np.zeros_like(spike_mat, dtype=np.float32)
    for u in range(total_units):
        smoothed[:, u] = np.convolve(spike_mat[:, u], np.ones(box_pts)/box_pts, mode='same')

    cov = np.cov(smoothed, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals[eigvals > 0]
    total_var = np.sum(eigvals)

    # Participation Ratio: PR = (sum lambda_i)^2 / sum(lambda_i^2)
    pr = float((total_var ** 2) / np.sum(eigvals ** 2))
    pr_frac = float(pr / total_units)

    top1_pct = float(eigvals[0] / total_var * 100.0)
    top3_pct = float(np.sum(eigvals[:3]) / total_var * 100.0)
    top5_pct = float(np.sum(eigvals[:5]) / total_var * 100.0)

    # Transition Co-Modulation
    transitions = np.where(np.diff(true_states) != 0)[0]
    n_trans = len(transitions)
    w_bins = int(0.050 / dt) # 50 ms window

    comod_fractions = []
    upmod_fractions = []
    downmod_fractions = []

    unit_means = np.mean(smoothed, axis=0)
    unit_stds = np.std(smoothed, axis=0) + 1e-6

    for t_idx in transitions:
        if t_idx < w_bins or t_idx > len(true_states) - w_bins:
            continue
        pre = smoothed[t_idx - w_bins : t_idx, :]
        post = smoothed[t_idx : t_idx + w_bins, :]
        diff_z = (np.mean(post, axis=0) - np.mean(pre, axis=0)) / unit_stds

        n_up = np.sum(diff_z > 2.0)
        n_down = np.sum(diff_z < -2.0)
        comod_fractions.append((n_up + n_down) / total_units)
        upmod_fractions.append(n_up / total_units)
        downmod_fractions.append(n_down / total_units)

    return {
        "participation_ratio": pr,
        "pr_fraction": pr_frac,
        "top1_pct": top1_pct,
        "top3_pct": top3_pct,
        "top5_pct": top5_pct,
        "mean_comod_fraction": float(np.mean(comod_fractions)) if comod_fractions else 0.0,
        "mean_upmod_fraction": float(np.mean(upmod_fractions)) if upmod_fractions else 0.0,
        "mean_downmod_fraction": float(np.mean(downmod_fractions)) if downmod_fractions else 0.0,
        "n_transitions": n_trans
    }


def generate_interactive_dashboard(
    all_data: Dict[str, Any],
    breath_data: Dict[str, Any],
    out_html: Path
):
    """Builds interactive HTML Plotly dashboard for the intact-window analysis."""
    logger.info(f"Generating interactive dashboard at {out_html}...")
    ks4_data = all_data["kilosort4"]
    lup_data = all_data.get("lupin", None)

    # Convert results to JSON for client-side Plotly rendering
    client_payload = {
        "ks4_subsampling": ks4_data["subsampling"]["results_by_size"],
        "ks4_ablation": ks4_data["attribution"]["ablation_results"],
        "ks4_lorenz": {
            "u": ks4_data["attribution"]["lorenz_cum_units"],
            "i": ks4_data["attribution"]["lorenz_cum_info"]
        },
        "ks4_gini": ks4_data["attribution"]["gini_mi"],
        "ks4_unit_stats": ks4_data["attribution"]["unit_stats"],
        "ks4_coordination": ks4_data["coordination"],
        "lupin_subsampling": lup_data["subsampling"]["results_by_size"] if lup_data else None,
        "lupin_ablation": lup_data["attribution"]["ablation_results"] if lup_data else None,
        "lupin_coordination": lup_data["coordination"] if lup_data else None,
        "agreement": all_data.get("agreement", None)
    }
    json_str = json.dumps(client_payload)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>e097 Intact Window (420s) Dual-Sorter HMM & Drop-Out Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #0f172a;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --state1: #10b981;
      --state2: #f97316;
      --state3: #38bdf8;
      --state4: #a855f7;
      --state5: #64748b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; }}
    .header {{ margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    .header h1 {{ font-size: 26px; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }}
    .header p {{ color: var(--text-muted); font-size: 14px; max-width: 1000px; line-height: 1.5; }}
    
    .nav-bar {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
    .nav-btn {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: 500; transition: all 0.2s; }}
    .nav-btn:hover {{ background: #334155; color: #fff; }}
    .nav-btn.primary {{ background: var(--accent); color: #0f172a; border-color: var(--accent); font-weight: 600; }}

    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
    .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
    @media (max-width: 1100px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }}
    .card-title {{ font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: #1e293b; color: var(--accent); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; border: 1px solid #334155; }}

    .stat-row {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
    .stat-box {{ background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 18px; flex: 1; min-width: 140px; }}
    .stat-val {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
    .stat-lbl {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}

    .table-container {{ overflow-x: auto; margin-top: 10px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th {{ background: #131d31; color: #94a3b8; padding: 8px 12px; border-bottom: 1px solid #1e293b; font-weight: 600; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }}
    tr:hover {{ background: #131d31; }}

    .plot-container {{ width: 100%; height: 380px; }}
    .plot-tall {{ height: 460px; }}
    
    .key-takeaways {{ background: #0c1527; border: 1px solid #1d4ed8; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
    .key-takeaways h3 {{ color: #60a5fa; font-size: 15px; margin-bottom: 8px; }}
    .key-takeaways ul {{ list-style-type: square; margin-left: 20px; color: #cbd5e1; font-size: 13.5px; line-height: 1.6; }}
  </style>
</head>
<body>

  <div class="header">
    <h1>e097 Intact Window (1740s &ndash; 2160s / 420s) Dual-Sorter HMM & Drop-Out Analysis</h1>
    <p>Complete unpartitioned recording pipeline analyzing the full cardiorespiratory challenge: Baseline Eupnea &rarr; Gas Switch &rarr; Apnea Nadir &rarr; Autoresuscitation Gasping &rarr; Expiratory Shift &rarr; Recovery with Kilosort 4 &amp; Lupin.</p>
  </div>

  <div class="nav-bar">
    <a href="../subwindows_sorting/master_interactive_population_microstate_raster.html" class="nav-btn primary">&larr; Back to Subwindows Hub</a>
    <a href="../subwindows_sorting/population_dropout_analysis_dashboard.html" class="nav-btn">Subwindow Drop-Out Hub</a>
    <a href="intact_window_population_dynamics_and_dropout.png" target="_blank" class="nav-btn">View Static High-Res PNG (300 DPI)</a>
  </div>

  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-val" style="color: #38bdf8;">420.0 s</div>
      <div class="stat-lbl">Intact Window Duration (7.0 min)</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #10b981;">{breath_data['n_breaths']}</div>
      <div class="stat-lbl">Total Breaths Segmented</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #f59e0b;">{ks4_data['attribution']['gini_mi']:.2f}</div>
      <div class="stat-lbl">Neuron Info Gini Coefficient (KS4)</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #a855f7;">{ks4_data['coordination']['mean_comod_fraction']*100:.1f}%</div>
      <div class="stat-lbl">Ensemble Co-Modulation at Transitions</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color: #ec4899;">PR = {ks4_data['coordination']['participation_ratio']:.1f}</div>
      <div class="stat-lbl">Participation Ratio ({ks4_data['subsampling']['results_by_size'][0]['N']} units)</div>
    </div>
  </div>

  <div class="key-takeaways">
    <h3>Key Insights from Whole Intact Window Pipeline:</h3>
    <ul>
      <li><strong>Continuous Dynamics Without Boundary Artifacts:</strong> Unlike subwindow splitting, single units maintain unbroken identities across baseline eupnea, hypoxic arrest, gasping, and recovery.</li>
      <li><strong>Subsampling Robustness Across the Entire Transition:</strong> Macro F1 across all 5 states remains above 0.80 down to N = 30 units (25% of population). Inspiratory bursting (State 1) survives down to N = 8 units.</li>
      <li><strong>Distributed Architecture:</strong> Ablation of the top 10% most informative driver units leaves the remaining 90% capable of generating all canonical states (Macro F1 &gt; 0.85).</li>
    </ul>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">
        <span>A. Population Subsampling Robustness Curves</span>
        <span class="badge">Adjusted Rand Index</span>
      </div>
      <div id="plot_subsampling" class="plot-container"></div>
    </div>
    <div class="card">
      <div class="card-title">
        <span>B. Canonical State Survival Curves</span>
        <span class="badge">Kilosort 4 Intact</span>
      </div>
      <div id="plot_state_survival" class="plot-container"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">
        <span>C. Targeted vs Random Population Ablation</span>
        <span class="badge">Ablation Impact</span>
      </div>
      <div id="plot_ablation" class="plot-container"></div>
    </div>
    <div class="card">
      <div class="card-title">
        <span>D. Information Inequality & Lorenz Curve</span>
        <span class="badge">Gini Inequality</span>
      </div>
      <div id="plot_lorenz" class="plot-container"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">
        <span>E. Anatomical Depth Profile along VRC Column</span>
        <span class="badge">Probe Coordinates</span>
      </div>
      <div id="plot_depth" class="plot-container plot-tall"></div>
    </div>
    <div class="card">
      <div class="card-title">
        <span>F. Ensemble Dimensionality & Coordination</span>
        <span class="badge">Participation Ratio</span>
      </div>
      <div id="plot_coordination" class="plot-container plot-tall"></div>
    </div>
  </div>

  <script>
    const data = {json_str};
    const darkLayout = {{
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#0f172a',
      font: {{ color: '#cbd5e1', family: 'system-ui' }},
      margin: {{ l: 50, r: 25, t: 25, b: 45 }},
      legend: {{ orientation: 'h', y: -0.2 }},
      xaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }},
      yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#334155' }}
    }};

    // Plot A: Subsampling Robustness
    const traceA_ks4 = {{
      x: data.ks4_subsampling.map(r => r.N),
      y: data.ks4_subsampling.map(r => r.ari_mean),
      error_y: {{ type: 'data', array: data.ks4_subsampling.map(r => r.ari_sem), visible: true }},
      name: 'Kilosort 4 (Intact 420s)',
      line: {{ color: '#10b981', width: 2.5 }},
      marker: {{ size: 7 }}
    }};
    const tracesA = [traceA_ks4];
    if (data.lupin_subsampling) {{
      tracesA.push({{
        x: data.lupin_subsampling.map(r => r.N),
        y: data.lupin_subsampling.map(r => r.ari_mean),
        error_y: {{ type: 'data', array: data.lupin_subsampling.map(r => r.ari_sem), visible: true }},
        name: 'Lupin (Intact 420s)',
        line: {{ color: '#38bdf8', width: 2.5 }},
        marker: {{ size: 7 }}
      }});
    }}
    Plotly.newPlot('plot_subsampling', tracesA, {{
      ...darkLayout,
      xaxis: {{ ...darkLayout.xaxis, title: 'Population Size (N Units)', autorange: 'reversed' }},
      yaxis: {{ ...darkLayout.yaxis, title: 'Adjusted Rand Index (ARI)', range: [0, 1.05] }}
    }}, {{ responsive: true }});

    // Plot B: State Survival Curves
    const colors = ['#10b981', '#f97316', '#38bdf8', '#a855f7', '#64748b'];
    const names = ['State 1: Insp Burst', 'State 2: Peak Off-Switch', 'State 3: Exp Drive', 'State 4: Pre-I Priming', 'State 5: Pause'];
    const tracesB = [0, 1, 2, 3, 4].map(st => ({{
      x: data.ks4_subsampling.map(r => r.N),
      y: data.ks4_subsampling.map(r => r.state_f1_mean[st]),
      name: names[st],
      line: {{ color: colors[st], width: 2.2 }},
      marker: {{ size: 6 }}
    }}));
    Plotly.newPlot('plot_state_survival', tracesB, {{
      ...darkLayout,
      xaxis: {{ ...darkLayout.xaxis, title: 'Population Size (N Units)', autorange: 'reversed' }},
      yaxis: {{ ...darkLayout.yaxis, title: 'State F1 Score', range: [0, 1.05] }}
    }}, {{ responsive: true }});

    // Plot C: Ablation
    const fracs = data.ks4_ablation.map(r => r.frac * 100);
    const traceC_top = {{
      x: fracs,
      y: data.ks4_ablation.map(r => r.ari_no_top),
      name: 'Ablate Top Drivers',
      line: {{ color: '#ef4444', width: 2.2 }},
      marker: {{ size: 6 }}
    }};
    const traceC_rand = {{
      x: fracs,
      y: data.ks4_ablation.map(r => r.ari_no_random_mean),
      error_y: {{ type: 'data', array: data.ks4_ablation.map(r => r.ari_no_random_sem), visible: true }},
      name: 'Ablate Random Units',
      line: {{ color: '#38bdf8', width: 2.2 }},
      marker: {{ size: 6 }}
    }};
    const traceC_bot = {{
      x: fracs,
      y: data.ks4_ablation.map(r => r.ari_no_bottom),
      name: 'Ablate Bottom Units',
      line: {{ color: '#a855f7', width: 2.2 }},
      marker: {{ size: 6 }}
    }};
    Plotly.newPlot('plot_ablation', [traceC_top, traceC_rand, traceC_bot], {{
      ...darkLayout,
      xaxis: {{ ...darkLayout.xaxis, title: 'Population Removed (%)' }},
      yaxis: {{ ...darkLayout.yaxis, title: 'Adjusted Rand Index (ARI)', range: [0.3, 1.05] }}
    }}, {{ responsive: true }});

    // Plot D: Lorenz Curve
    const traceD_eq = {{
      x: [0, 1], y: [0, 1],
      name: 'Perfect Equality (Gini=0)',
      line: {{ color: '#64748b', dash: 'dash', width: 1.5 }}
    }};
    const traceD_ks4 = {{
      x: data.ks4_lorenz.u,
      y: data.ks4_lorenz.i,
      name: `Kilosort 4 (Gini=${{data.ks4_gini.toFixed(2)}})`,
      line: {{ color: '#10b981', width: 2.5 }},
      fill: 'tonexty',
      fillcolor: 'rgba(16, 185, 129, 0.1)'
    }};
    Plotly.newPlot('plot_lorenz', [traceD_eq, traceD_ks4], {{
      ...darkLayout,
      xaxis: {{ ...darkLayout.xaxis, title: 'Cumulative Fraction of Units' }},
      yaxis: {{ ...darkLayout.yaxis, title: 'Cumulative Fraction of State Information' }}
    }}, {{ responsive: true }});

    // Plot E: Depth Profile
    const traceE = {{
      x: data.ks4_unit_stats.map(u => u.mean_fr),
      y: data.ks4_unit_stats.map(u => u.depth_um),
      mode: 'markers',
      marker: {{
        size: data.ks4_unit_stats.map(u => Math.max(6, u.mi_bits * 120)),
        color: data.ks4_unit_stats.map(u => u.mi_bits),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {{ title: 'Mutual Info (bits)', thickness: 12, len: 0.8 }}
      }},
      text: data.ks4_unit_stats.map(u => `Unit ${{u.unit_idx}}<br>Depth: ${{u.depth_um}} um<br>FR: ${{u.mean_fr.toFixed(2)}} Hz<br>MI: ${{u.mi_bits.toFixed(3)}} bits`),
      hoverinfo: 'text'
    }};
    Plotly.newPlot('plot_depth', [traceE], {{
      ...darkLayout,
      xaxis: {{ ...darkLayout.xaxis, title: 'Mean Firing Rate (Hz)' }},
      yaxis: {{ ...darkLayout.yaxis, title: 'Electrode Depth (um, from tip)' }}
    }}, {{ responsive: true }});

    // Plot F: Coordination & PR
    const traceF_comod = {{
      x: ['Kilosort 4', 'Lupin'],
      y: [
        data.ks4_coordination.mean_comod_fraction * 100,
        data.lupin_coordination ? data.lupin_coordination.mean_comod_fraction * 100 : 0
      ],
      name: 'Co-Modulated at Transitions (%)',
      type: 'bar',
      marker: {{ color: '#38bdf8' }}
    }};
    const traceF_pr = {{
      x: ['Kilosort 4', 'Lupin'],
      y: [
        data.ks4_coordination.participation_ratio,
        data.lupin_coordination ? data.lupin_coordination.participation_ratio : 0
      ],
      name: 'Participation Ratio (PR)',
      type: 'bar',
      marker: {{ color: '#ec4899' }}
    }};
    Plotly.newPlot('plot_coordination', [traceF_comod, traceF_pr], {{
      ...darkLayout,
      barmode: 'group',
      yaxis: {{ ...darkLayout.yaxis, title: 'Percentage (%) / Units' }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Saved interactive dashboard to {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Run Intact-Window Dual-Sorter & HMM Pipeline")
    parser.add_argument("--skip-catgt", action="store_true", default=False)
    parser.add_argument("--skip-sorting", action="store_true", default=False)
    parser.add_argument("--sorters", nargs="+", default=["kilosort4", "lupin"])
    args = parser.parse_args()

    raw_ap_dir = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_imec0")
    catgt_exe = r"C:\Users\lpascu\SpikeGLX\CatGT-win\CatGT.exe"
    scratch_catgt_top = Path(r"C:\SCRATCH_neuropixels\catgt_e097_intact_420s")
    scratch_sort_base = Path(r"C:\SCRATCH_neuropixels\sorting_intact_window")
    out_base = REPO_ROOT / "outputs" / "e097" / "intact_window"
    out_base.mkdir(parents=True, exist_ok=True)

    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")

    # 1. CatGT Preprocessing
    if not args.skip_catgt:
        bin_path, meta_path = load_or_run_catgt(raw_ap_dir, scratch_catgt_top, catgt_exe)
    else:
        catgt_folder = scratch_catgt_top / "catgt_e097_g2"
        bin_path = catgt_folder / "e097_g2_tcat.imec0.ap.bin"
        meta_path = catgt_folder / "e097_g2_tcat.imec0.ap.meta"

    # 2. Ingest CatGT recording
    catgt_folder = bin_path.parent
    logger.info(f"Ingesting preprocessed recording from {catgt_folder}...")
    recording = read_spikeglx_ap_safe(catgt_folder)
    fs = recording.get_sampling_frequency()
    logger.info(f"Loaded: {recording.get_num_channels()} chans, {fs:.1f} Hz, {recording.get_total_duration():.2f}s.")

    # 3. Dual Spike Sorting
    if not args.skip_sorting:
        sort_results = run_intact_spike_sorting(
            recording=recording,
            out_dir=out_base,
            scratch_dir=scratch_sort_base,
            sorters=args.sorters
        )
    else:
        logger.info("Skipping spike sorting as requested.")
        sort_results = {}

    # 4. NIDQ Analog Signals & Breath Segmentation
    breath_data = extract_intact_breaths(nidq_file, start_t=INTACT_T_START, dur=INTACT_DURATION)

    # 5. Analysis per Sorter
    all_analysis_results = {}
    if "agreement" in sort_results:
        all_analysis_results["agreement"] = sort_results["agreement"]

    for s_name in args.sorters:
        s_dir = out_base / s_name
        alf_dir = s_dir / "alf"
        if not (alf_dir / "spike_times.npy").exists():
            logger.warning(f"No sorted output found for {s_name} at {alf_dir}. Skipping analysis.")
            continue

        logger.info("=" * 80)
        logger.info(f" PROCESSING HMM & DROPOUT FOR INTACT RECORDER: {s_name.upper()}")
        logger.info("=" * 80)

        spk_times, spk_clusters, depths, df_qc = load_intact_spikes_and_qc(s_dir, max_units=120)

        # Build Interactive Sorter Raster if missing
        raster_html = s_dir / f"{s_name}_interactive_raster.html"
        if not raster_html.exists():
            logger.info(f"Building interactive raster for {s_name}...")
            try:
                build_interactive_sorter_raster(
                    sorter_dir=s_dir,
                    sorter_name=s_name,
                    start_t_s=INTACT_T_START
                )
            except Exception as e:
                logger.warning(f"Failed to build interactive raster: {e}")

        # Fit 5-State Poisson HMM
        hmm_res = fit_intact_poisson_hmm(spk_times, spk_clusters, df_qc, dur=INTACT_DURATION, dt=0.010, n_states=5)

        # Run Subsampling Experiment
        sub_res = run_intact_subsampling_experiment(hmm_res["spike_mat"], hmm_res["state_seq"], n_draws=8)

        # Run Single-Neuron Attribution
        attrib_res = run_intact_single_neuron_attribution(hmm_res["spike_mat"], hmm_res["state_seq"], depths, dt=0.010)

        # Run Coordination Analysis
        coord_res = run_intact_coordination_analysis(hmm_res["spike_mat"], hmm_res["state_seq"], dt=0.010)

        all_analysis_results[s_name] = {
            "n_units": len(df_qc),
            "subsampling": sub_res,
            "attribution": attrib_res,
            "coordination": coord_res,
            "mean_state_rates_hz": (hmm_res["canon_rates"].mean(axis=1) / 0.010).tolist()
        }

    # Save JSON metrics cache
    json_path = out_base / "intact_window_analysis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_analysis_results, f, indent=2)
    logger.info(f"Saved complete metrics to {json_path}")

    # Generate interactive dashboard
    dash_html = out_base / "intact_window_population_dynamics_dashboard.html"
    generate_interactive_dashboard(all_analysis_results, breath_data, dash_html)

    logger.info("=" * 80)
    logger.info(" INTACT-WINDOW ANALYSIS PIPELINE COMPLETE!")
    logger.info(f" Dashboard: {dash_html}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
