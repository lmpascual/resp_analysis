#!/usr/bin/env python3
"""
Generate Interactive Phy-Like Validation Suite for E097 Diving Reflex (240s).
Mirrors: outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run/phy_validation_suite

Features:
1. Multi-cluster iframe browser: cluster_validation_browser.html
2. Deep-dive unit HTML pages: units/unit_{uid:04d}_ch{peak_chan}.html
3. Panel 1 Trace Inspector & Interactive Waveform Zoom with 1x, 1.5x, 2x, 3x, ➕ Bigger, ➖ Smaller, ↺ Reset buttons
4. Red circle action potential trough alignment & amber triangle ISI violation markers
5. Multi-criteria filtering: All (400), Strict SUA (99), Continuous SUA (122), Gasp Recruited (9), Dedup Split Candidates (3), Rostral Continuum (18), Failed QC (269)
6. Dual export to both sorting_validation/ and phy_validation_suite/
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.diagnostics.cluster_validator import (
    UnitValidationBundle,
    find_unit_channel_footprint,
    extract_representative_trace_snippet,
    compute_respiratory_phase_vector,
    compute_isi_and_violations,
    compute_amplitude_stationarity,
    compute_time_resolved_isi_pdfs,
    compute_autocorrelogram,
    compute_amplitude_cutoff,
    compute_waveform_kinetics,
)
from src.pipeline.diagnostics.cluster_html_report import generate_single_unit_html

ROOT_DIR = Path(r"c:\Users\lpascu\resp_analysis")
OUT_DIR = ROOT_DIR / "outputs" / "e097" / "output_e097_diving_reflex_240s_full_probe"
SORTING_VAL_DIR = OUT_DIR / "sorting_validation"
PHY_VAL_DIR = OUT_DIR / "phy_validation_suite"

RAW_AP_BIN = Path(r"C:\SCRATCH_neuropixels\catgt_e097_diving_240s\catgt_e097_g2\e097_g2_tcat.imec0.ap.bin")
NIDQ_PATH = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")

START_T_S = 3300.0
DURATION_S = 240.0
FS_NIDQ = 10000.0
FS_AP = 29999.821974390747
N_CHANS_AP = 385


def load_nidaq_bursts() -> Tuple[np.ndarray, np.ndarray, str]:
    """Fast-seek NIDAQ binary for the 240s window and detect diaphragm EMG onsets."""
    print("Loading NIDAQ segment from raw binary...")
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
    flow_raw = raw[:, 5] * v_conv

    # Downsample to 1 kHz for burst landmark detection
    ds = 10
    fs_ds = FS_NIDQ / ds
    emg_ds = emg_raw[::ds]
    time_ds = np.linspace(0, DURATION_S, len(emg_ds), endpoint=False)

    # High-pass filter & envelope
    sos = signal.butter(4, 100.0, btype="highpass", fs=fs_ds, output="sos")
    emg_hp = signal.sosfilt(sos, emg_ds)
    win_len = int(0.05 * fs_ds)
    win = np.ones(win_len) / win_len
    env = np.sqrt(signal.convolve(emg_hp**2, win, mode="same"))

    # Peak detection for respiratory bursts
    baseline_std = np.std(env[: int(30.0 * fs_ds)])
    threshold = np.median(env) + 2.0 * baseline_std
    peaks, _ = signal.find_peaks(env, height=threshold, distance=int(0.25 * fs_ds))

    onsets = []
    offsets = []
    for pk in peaks:
        # Seek backwards for onset (below 25% peak)
        pk_val = env[pk]
        cutoff = 0.25 * pk_val
        left = pk
        while left > 0 and env[left] > cutoff and (pk - left) < int(0.3 * fs_ds):
            left -= 1
        right = pk
        while right < len(env) - 1 and env[right] > cutoff and (right - pk) < int(0.5 * fs_ds):
            right += 1
        t_on = time_ds[left]
        t_off = time_ds[right]
        if t_off > t_on + 0.05:
            onsets.append(t_on)
            offsets.append(t_off)

    breath_onsets = np.array(onsets, dtype=np.float64)
    breath_offsets = np.array(offsets, dtype=np.float64)
    prov = "Diaphragm EMG Burst Detection (Baseline Eupnea vs Diving Apnea)"
    print(f"Detected {len(breath_onsets)} breath cycles across {DURATION_S}s window.")
    return breath_onsets, breath_offsets, prov


def build_browser_html(units_manifest: List[Dict], dest_path: Path):
    """Generates the multi-cluster validation browser matching LP21 architecture."""
    manifest_json = json.dumps(units_manifest)
    first_file = units_manifest[0]["file_name"] if len(units_manifest) > 0 else ""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E097 Diving Reflex Spike Sorting Validation Browser</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        .navbar {
            background: #1e293b;
            border-bottom: 1px solid #334155;
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            z-index: 100;
        }
        .nav-left {
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }
        .nav-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .selector-box {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        select {
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            max-width: 480px;
        }
        select:focus {
            border-color: #38bdf8;
        }
        .btn {
            background: #334155;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.80rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn:hover {
            background: #475569;
        }
        .btn-active {
            background: #0284c7 !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
        }
        .nav-links {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-left: auto;
        }
        .link-badge {
            background: #0369a1;
            color: #ffffff;
            text-decoration: none;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 6px;
            transition: background 0.15s;
            border: 1px solid #0284c7;
            white-space: nowrap;
        }
        .link-badge:hover {
            background: #0284c7;
        }
        .iframe-container {
            flex: 1;
            width: 100%;
            height: 100%;
            border: none;
            background: #f8fafc;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .shortcuts-hint {
            font-size: 0.78rem;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        kbd {
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 3px;
            padding: 2px 5px;
            font-size: 0.72rem;
            color: #cbd5e1;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="nav-left">
            <span class="nav-title">
                <span>🔬</span> Spike Sorting Validation Browser
            </span>
            <div class="selector-box">
                <label for="unitSelect" style="font-size:0.85rem;color:#94a3b8;">Cluster:</label>
                <select id="unitSelect" onchange="loadSelectedUnit()"></select>
                <button class="btn" onclick="prevUnit()" title="Previous Unit (Left Arrow or J)">&#9664; Prev</button>
                <button class="btn" onclick="nextUnit()" title="Next Unit (Right Arrow or K)">Next &#9654;</button>
            </div>
            <div class="selector-box">
                <button class="btn btn-active" id="filterAll" onclick="applyFilter('all')">All (__N_ALL__)</button>
                <button class="btn" id="filterStrict" onclick="applyFilter('strict')">Strict SUA (__N_STRICT__)</button>
                <button class="btn" id="filterResp" onclick="applyFilter('resp')">Continuous SUA (__N_RESP__)</button>
                <button class="btn" id="filterGasp" onclick="applyFilter('gasp')">Gasp Recruited (__N_GASP__)</button>
                <button class="btn" id="filterDedup" onclick="applyFilter('dedup')">Dedup Overlap (__N_DEDUP__)</button>
                <button class="btn" id="filterRostral" onclick="applyFilter('rostral')">Rostral Continuum (__N_ROSTRAL__)</button>
                <button class="btn" id="filterFail" onclick="applyFilter('fail')">Failed QC (__N_FAIL__)</button>
            </div>
        </div>
        <div class="nav-links">
            <div class="shortcuts-hint">
                <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> or <kbd>J</kbd> / <kbd>K</kbd> to cycle
            </div>
            <a href="../consensus_master/lupin_qc_interactive_raster.html" target="_blank" class="link-badge">📊 Master Raster (6-Panel)</a>
            <a href="../consensus_master/04_gasping_transition_rasters/gasping_transition_dashboard.html" target="_blank" class="link-badge">🫁 Transition Gallery</a>
            <a href="../consensus_master/02_single_cell_phase_rasters/gallery_all_units_chronological.html" target="_blank" class="link-badge">⏱️ Chronological Gallery</a>
        </div>
    </div>
    <div class="iframe-container">
        <iframe id="unitFrame" src="__FIRST_FILE__"></iframe>
    </div>

    <script>
    const manifest = __MANIFEST_JSON__;
    let currentFilter = 'all';
    let filteredUnits = [...manifest];
    let currentIndex = 0;

    const selectEl = document.getElementById('unitSelect');
    const frameEl = document.getElementById('unitFrame');

    function populateDropdown() {
        selectEl.innerHTML = '';
        filteredUnits.forEach((u, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            let statusTag = u.qc_label;
            opt.textContent = `Unit ${String(u.cluster_id).padStart(4, '0')} | Ch ${u.peak_channel} (${Math.round(u.depth_um)}um) | ${u.firing_rate_hz.toFixed(1)}Hz | ${statusTag}`;
            selectEl.appendChild(opt);
        });
        if (filteredUnits.length > 0) {
            currentIndex = 0;
            selectEl.selectedIndex = 0;
            frameEl.src = filteredUnits[0].file_name;
        } else {
            frameEl.src = 'about:blank';
        }
    }

    function loadSelectedUnit() {
        const idx = parseInt(selectEl.value, 10);
        if (idx >= 0 && idx < filteredUnits.length) {
            currentIndex = idx;
            frameEl.src = filteredUnits[currentIndex].file_name;
        }
    }

    function nextUnit() {
        if (filteredUnits.length === 0) return;
        currentIndex = (currentIndex + 1) % filteredUnits.length;
        selectEl.selectedIndex = currentIndex;
        frameEl.src = filteredUnits[currentIndex].file_name;
    }

    function prevUnit() {
        if (filteredUnits.length === 0) return;
        currentIndex = (currentIndex - 1 + filteredUnits.length) % filteredUnits.length;
        selectEl.selectedIndex = currentIndex;
        frameEl.src = filteredUnits[currentIndex].file_name;
    }

    function applyFilter(filterType) {
        currentFilter = filterType;
        document.querySelectorAll('.btn').forEach(b => {
            if (['filterAll', 'filterStrict', 'filterResp', 'filterGasp', 'filterDedup', 'filterRostral', 'filterFail'].includes(b.id)) {
                b.classList.remove('btn-active');
            }
        });
        if (filterType === 'all') {
            document.getElementById('filterAll').classList.add('btn-active');
            filteredUnits = [...manifest];
        } else if (filterType === 'strict') {
            document.getElementById('filterStrict').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_allen_strict);
        } else if (filterType === 'resp') {
            document.getElementById('filterResp').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_respiratory_continuous);
        } else if (filterType === 'gasp') {
            document.getElementById('filterGasp').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_gasp_recruited);
        } else if (filterType === 'dedup') {
            document.getElementById('filterDedup').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.would_be_deduplicated);
        } else if (filterType === 'rostral') {
            document.getElementById('filterRostral').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.is_rostral_continuum);
        } else if (filterType === 'fail') {
            document.getElementById('filterFail').classList.add('btn-active');
            filteredUnits = manifest.filter(u => !u.passed_respiratory_continuous && !u.passed_gasp_recruited);
        }
        populateDropdown();
    }

    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {
            nextUnit();
        } else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {
            prevUnit();
        }
    });

    populateDropdown();
    </script>
</body>
</html>
"""
    n_all = len(units_manifest)
    n_strict = sum(1 for u in units_manifest if u.get("passed_allen_strict"))
    n_resp = sum(1 for u in units_manifest if u.get("passed_respiratory_continuous"))
    n_gasp = sum(1 for u in units_manifest if u.get("passed_gasp_recruited"))
    n_dedup = sum(1 for u in units_manifest if u.get("would_be_deduplicated"))
    n_rostral = sum(1 for u in units_manifest if u.get("is_rostral_continuum"))
    n_fail = sum(1 for u in units_manifest if not u.get("passed_respiratory_continuous") and not u.get("passed_gasp_recruited"))

    html = (
        html.replace("__MANIFEST_JSON__", manifest_json)
        .replace("__FIRST_FILE__", first_file)
        .replace("__N_ALL__", str(n_all))
        .replace("__N_STRICT__", str(n_strict))
        .replace("__N_RESP__", str(n_resp))
        .replace("__N_GASP__", str(n_gasp))
        .replace("__N_DEDUP__", str(n_dedup))
        .replace("__N_ROSTRAL__", str(n_rostral))
        .replace("__N_FAIL__", str(n_fail))
    )
    dest_path.write_text(html, encoding="utf-8")
    print(f"Browser HTML saved to: {dest_path}")


def main():
    global OUT_DIR, SORTING_VAL_DIR, PHY_VAL_DIR, RAW_AP_BIN, NIDQ_PATH
    import argparse
    parser = argparse.ArgumentParser(description="Generate Interactive Phy-Like Validation Suite")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="Output folder containing sorted units")
    parser.add_argument("--raw-ap-bin", type=str, default=str(RAW_AP_BIN), help="Path to preprocessed AP binary")
    parser.add_argument("--nidq-path", type=str, default=str(NIDQ_PATH), help="Path to raw NIDAQ bin")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    SORTING_VAL_DIR = OUT_DIR / "sorting_validation"
    PHY_VAL_DIR = OUT_DIR / "phy_validation_suite"
    RAW_AP_BIN = Path(args.raw_ap_bin)
    NIDQ_PATH = Path(args.nidq_path)

    print("=" * 80)
    print(" GENERATING INTERACTIVE PHY-LIKE CLUSTER VALIDATION SUITE (E097 DIVING REFLEX)")
    print(f" Source: {OUT_DIR}")
    print(f" Dest: {SORTING_VAL_DIR}")
    print("=" * 80)

    SORTING_VAL_DIR.mkdir(parents=True, exist_ok=True)
    val_units_dir = SORTING_VAL_DIR / "units"
    val_units_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ALF spike trains & cluster metadata
    spk_times = np.load(OUT_DIR / "alf" / "spike_times.npy")
    spk_clusts = np.load(OUT_DIR / "alf" / "spike_clusters.npy")
    df_clusters = pd.read_csv(OUT_DIR / "cluster_info.tsv", sep="\t")
    print(f"Loaded {len(df_clusters)} clusters and {len(spk_times):,} spikes.")

    # Harmonize column aliases
    if "pass_allen" in df_clusters.columns and "passed_allen_strict" not in df_clusters.columns:
        df_clusters["passed_allen_strict"] = df_clusters["pass_allen"]
    if "pass_resp" in df_clusters.columns and "passed_respiratory_continuous" not in df_clusters.columns:
        df_clusters["passed_respiratory_continuous"] = df_clusters["pass_resp"]
    if "pass_gasp" in df_clusters.columns and "passed_gasp_recruited" not in df_clusters.columns:
        df_clusters["passed_gasp_recruited"] = df_clusters["pass_gasp"]
    if "would_dedup" in df_clusters.columns and "would_be_deduplicated" not in df_clusters.columns:
        df_clusters["would_be_deduplicated"] = df_clusters["would_dedup"]
    if "coinc_frac" in df_clusters.columns and "coincidence_fraction" not in df_clusters.columns:
        df_clusters["coincidence_fraction"] = df_clusters["coinc_frac"]
    if "coinc_partner" in df_clusters.columns and "most_coincident_partner" not in df_clusters.columns:
        df_clusters["most_coincident_partner"] = df_clusters["coinc_partner"]
    if "depth" in df_clusters.columns and "depth_um" not in df_clusters.columns:
        df_clusters["depth_um"] = df_clusters["depth"]
    if "ch" in df_clusters.columns and "peak_channel" not in df_clusters.columns:
        df_clusters["peak_channel"] = df_clusters["ch"]
    if "fr" in df_clusters.columns and "firing_rate_hz" not in df_clusters.columns:
        df_clusters["firing_rate_hz"] = df_clusters["fr"]
    if "n_spk" in df_clusters.columns and "n_spikes" not in df_clusters.columns:
        df_clusters["n_spikes"] = df_clusters["n_spk"]
    if "is_rostral" in df_clusters.columns and "is_rostral_continuum" not in df_clusters.columns:
        df_clusters["is_rostral_continuum"] = df_clusters["is_rostral"]
    if "apnea_fc" in df_clusters.columns and "apnea_fold_change" not in df_clusters.columns:
        df_clusters["apnea_fold_change"] = df_clusters["apnea_fc"]
    if "apnea_fr" in df_clusters.columns and "apnea_fr_hz" not in df_clusters.columns:
        df_clusters["apnea_fr_hz"] = df_clusters["apnea_fr"]

    # 2. Load Respiratory Onsets
    breath_onsets, breath_offsets, prov = load_nidaq_bursts()

    # 3. Channel Geometry (True Probe Coordinates: Rostral y=0um to Caudal y=2865um)
    try:
        import spikeinterface.extractors as se
        raw_parent = RAW_AP_BIN.parent
        rec = se.read_spikeglx(str(raw_parent), stream_id="imec0.ap")
        locs = rec.get_channel_locations()
        channel_depths = {ch: float(locs[ch, 1]) for ch in range(len(locs))}
        if 384 not in channel_depths:
            channel_depths[384] = 0.0
        print(f"Loaded Neuropixels probe geometry: {len(locs)} channels (Span: {locs[:,1].min():.0f}um to {locs[:,1].max():.0f}um)")
    except Exception as e:
        print(f"Fallback probe geometry ({e}): using standard spacing.")
        channel_depths = {ch: float((ch // 2) * 15.0) for ch in range(385)}

    # 4. Open AP Binary Memory Map
    ap_file_bytes = RAW_AP_BIN.stat().st_size
    n_samples = ap_file_bytes // (2 * N_CHANS_AP)
    print(f"AP Binary: {RAW_AP_BIN.name} ({n_samples:,} samples, {n_samples / FS_AP:.1f}s)")
    mm_ap = np.memmap(RAW_AP_BIN, dtype=np.int16, mode="r", shape=(n_samples, N_CHANS_AP))

    # Pre-index spike times by cluster_id for high-speed O(1) lookup
    print("Indexing spike times by cluster ID...")
    sort_idx = np.argsort(spk_clusts)
    sorted_clusts = spk_clusts[sort_idx]
    sorted_times = spk_times[sort_idx]
    unique_c, split_idx = np.unique(sorted_clusts, return_index=True)
    splits = np.split(sorted_times, split_idx[1:])
    cluster_spike_dict = {int(c): np.sort(st) for c, st in zip(unique_c, splits)}
    print(f"Indexed spike trains for {len(cluster_spike_dict)} clusters.")

    # Sort clusters: prioritize passing SUA units (Gasp, Allen, Continuous), then candidates, then descending FR
    def cluster_sort_priority(row):
        if row.get("passed_gasp_recruited", False):
            return (0, -float(row.get("apnea_fold_change", 1.0)))
        if row.get("passed_allen_strict", False):
            return (1, -float(row.get("firing_rate_hz", 1.0)))
        if row.get("passed_respiratory_continuous", False):
            return (2, -float(row.get("firing_rate_hz", 1.0)))
        if row.get("would_be_deduplicated", False):
            return (3, -float(row.get("coincidence_fraction", 0.0)))
        return (4, -float(row.get("firing_rate_hz", 1.0)))

    df_clusters["sort_key"] = [cluster_sort_priority(r) for _, r in df_clusters.iterrows()]
    df_sorted = df_clusters.sort_values(by="sort_key").reset_index(drop=True)

    # Generate for all clusters so every cluster in the dropdown is navigable!
    target_clusters = df_sorted

    manifest = []
    t_start_all = time.time()
    n_total = len(target_clusters)

    print(f"Processing all {n_total} clusters for full validation suite...")

    for i, (_, row) in enumerate(target_clusters.iterrows()):
        t_u0 = time.time()
        uid = int(row["cluster_id"])
        fr = float(row["firing_rate_hz"])
        n_spk = int(row["n_spikes"])
        is_strict = bool(row.get("passed_allen_strict", False))
        is_resp = bool(row.get("passed_respiratory_continuous", False))
        is_gasp = bool(row.get("passed_gasp_recruited", False))
        is_dedup = bool(row.get("would_be_deduplicated", False))
        is_rostral = bool(row.get("is_rostral_continuum", False))
        coinc_frac = float(row.get("coincidence_fraction", 0.0))
        partner = int(row.get("most_coincident_partner", -1))
        fail_reasons = str(row.get("fail_reasons", "Passed"))
        apnea_fr = float(row.get("apnea_fr_hz", 0.0))
        apnea_fc = float(row.get("apnea_fold_change", 1.0))

        u_spk_times = cluster_spike_dict.get(uid, np.array([], dtype=np.float64))
        if len(u_spk_times) < 5:
            continue

        # Footprint channels & peak soma
        peak_chan, footprint_chans, channel_ptps, template_wfs = find_unit_channel_footprint(
            mm_ap=mm_ap,
            spike_times_s=u_spk_times,
            fs=FS_AP,
            candidate_channels=None,
            n_footprint_channels=8,
            max_sample_spikes=100,
        )
        true_depth_um = float(channel_depths.get(peak_chan, (peak_chan // 2) * 15.0))

        # Determine optimal 1.5s burst snippet window
        # For gasp-recruited neurons, center in apnea window (45s - 115s)
        if is_gasp:
            gasp_spikes = u_spk_times[(u_spk_times >= 45.0) & (u_spk_times <= 115.0)]
            if len(gasp_spikes) > 0:
                # Find maximum density 1.5s window in gasp period
                best_t = float(gasp_spikes[0])
                snip_t_start = max(45.0, best_t - 0.2)
            else:
                snip_t_start = None
        else:
            snip_t_start = None

        snippet = extract_representative_trace_snippet(
            mm_ap=mm_ap,
            channels=footprint_chans,
            channel_depths=channel_depths,
            spike_times_s=u_spk_times,
            peak_channel=peak_chan,
            t_start_s=snip_t_start,
            window_duration_s=1.5,
            fs=FS_AP,
            refractory_ms=1.0,
            max_display_pts_per_chan=100000,
        )

        # Biophysical & statistical metrics
        rp_res = compute_respiratory_phase_vector(
            spike_times_s=u_spk_times,
            breath_onsets=breath_onsets,
            breath_offsets=breath_offsets,
            landmark_provenance=prov,
        )

        isi_res = compute_isi_and_violations(
            spike_times_s=u_spk_times,
            refractory_ms=1.0,
        )

        peak_sample_idxs = (u_spk_times * FS_AP).astype(np.int64)
        valid_p_idxs = peak_sample_idxs[(peak_sample_idxs >= 15) & (peak_sample_idxs < n_samples - 30)]
        if len(valid_p_idxs) > 0:
            samp_subset = valid_p_idxs[: min(2000, len(valid_p_idxs))]
            amps = [float(np.ptp(mm_ap[s - 15 : s + 30, peak_chan])) for s in samp_subset]
            spk_amps_uv = np.array(amps, dtype=np.float32)
            st_times = u_spk_times[: len(spk_amps_uv)]
        else:
            spk_amps_uv = np.array([50.0] * len(u_spk_times), dtype=np.float32)
            st_times = u_spk_times

        st_res = compute_amplitude_stationarity(st_times, spk_amps_uv)
        tr_isi_res = compute_time_resolved_isi_pdfs(u_spk_times, n_epochs=5)
        acg_res = compute_autocorrelogram(u_spk_times)
        cutoff_res = compute_amplitude_cutoff(spk_amps_uv)

        peak_template = template_wfs.get(peak_chan, np.zeros(75, dtype=np.float32))
        wf_res = compute_waveform_kinetics(
            template_wf=peak_template,
            peak_chan=peak_chan,
            footprint_chans=footprint_chans,
            channel_depths=channel_depths,
            all_template_wfs=template_wfs,
            fs=FS_AP,
            noise_sigma=12.0,
        )

        # Custom Badge and Label
        if is_gasp:
            qc_badge = f'<span style="background-color:#065f46;color:#34d399;border:1px solid #059669;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Gasp Recruited SUA: {apnea_fc:.1f}x Surge)</span>'
            qc_label = f"Gasp Recruited SUA ({apnea_fc:.1f}x) [PASS]"
            overall_pass = True
        elif is_strict:
            qc_badge = '<span style="background-color:#10b981;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Strict Allen SUA)</span>'
            qc_label = "Strict Allen SUA [PASS]"
            overall_pass = True
        elif is_resp:
            qc_badge = '<span style="background-color:#0284c7;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Continuous Resp SUA)</span>'
            qc_label = "Continuous Resp SUA [PASS]"
            overall_pass = True
        elif is_dedup:
            qc_badge = f'<span style="background-color:#831843;color:#f43f5e;border:1px solid #e11d48;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">FAIL (Dedup Candidate: {coinc_frac*100:.0f}% split with #{partner})</span>'
            qc_label = f"Dedup Split of #{partner} ({coinc_frac*100:.0f}%) [FAIL]"
            overall_pass = False
        else:
            clean_reasons = fail_reasons.replace("Fail: ", "")
            qc_badge = f'<span style="background-color:#7f1d1d;color:#f87171;border:1px solid #dc2626;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">FAIL ({clean_reasons})</span>'
            qc_label = f"{clean_reasons} [FAIL]"
            overall_pass = False

        bundle = UnitValidationBundle(
            cluster_id=uid,
            firing_rate_hz=fr,
            total_spikes=len(u_spk_times),
            depth_um=true_depth_um,
            shank=0,
            peak_channel=peak_chan,
            footprint_channels=footprint_chans,
            passed_allen_qc=overall_pass,
            respiratory_phase=rp_res,
            isi=isi_res,
            stationarity=st_res,
            time_resolved_isi=tr_isi_res,
            acg=acg_res,
            waveform=wf_res,
            amplitude_cutoff=cutoff_res,
            trace_snippet=snippet,
            qc_summary=qc_label,
        )
        bundle.custom_badge_html = qc_badge
        bundle.custom_cli_dir = str(OUT_DIR.relative_to(PROJECT_ROOT) if OUT_DIR.is_relative_to(PROJECT_ROOT) else OUT_DIR)

        unit_file_name = f"unit_{uid:04d}_ch{peak_chan}.html"
        unit_file_path = val_units_dir / unit_file_name
        generate_single_unit_html(bundle, unit_file_path)

        manifest.append({
            "cluster_id": uid,
            "firing_rate_hz": fr,
            "total_spikes": len(u_spk_times),
            "depth_um": true_depth_um,
            "shank": 0,
            "peak_channel": peak_chan,
            "passed_allen_strict": is_strict,
            "passed_respiratory_continuous": is_resp,
            "passed_gasp_recruited": is_gasp,
            "would_be_deduplicated": is_dedup,
            "is_rostral_continuum": is_rostral,
            "qc_label": qc_label,
            "file_name": f"units/{unit_file_name}",
        })

        t_elapsed = time.time() - t_u0
        if (i + 1) % 25 == 0 or (i + 1) == n_total or is_gasp or is_dedup:
            print(f"[{i+1:03d}/{n_total:03d}] Unit {uid:04d} | Ch {peak_chan:03d} ({true_depth_um:.0f}um) | {fr:.1f}Hz | {qc_label} ({t_elapsed:.2f}s)")

    # 5. Build Unified Browser in sorting_validation/
    browser_path = SORTING_VAL_DIR / "cluster_validation_browser.html"
    build_browser_html(manifest, browser_path)

    # 6. Replicate entire suite into phy_validation_suite/
    print("Mirroring suite into phy_validation_suite/...")
    PHY_VAL_DIR.mkdir(parents=True, exist_ok=True)
    phy_units_dir = PHY_VAL_DIR / "units"
    phy_units_dir.mkdir(parents=True, exist_ok=True)

    for f in val_units_dir.glob("*.html"):
        shutil.copy2(f, phy_units_dir / f.name)

    shutil.copy2(browser_path, PHY_VAL_DIR / "cluster_validation_browser.html")

    total_time = time.time() - t_start_all
    print("=" * 80)
    print(f" COMPLETE! Processed {len(manifest)} units in {total_time:.1f}s ({total_time/len(manifest):.2f}s/unit).")
    print(f" sorting_validation: file:///{browser_path.resolve().as_posix()}")
    print(f" phy_validation_suite: file:///{(PHY_VAL_DIR / 'cluster_validation_browser.html').resolve().as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
