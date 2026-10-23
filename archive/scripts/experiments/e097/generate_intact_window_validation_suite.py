#!/usr/bin/env python3
"""
Generate Dual-Sorter Interactive Validation Suite for E097 Intact Window (420.0s).
Window: 1740.0s - 2160.0s (420.0s Eupnea, Anoxia Challenge, Apnea, & Recovery)
Dual Sorters: Kilosort 4 (1,109 clusters) and Lupin (273 clusters)

Features:
1. Full Neuropixels probe footprint analysis & 1.5s native 30 kHz raw AP trace snippets.
2. Trace Inspector with trough alignment (red circles) & ISI violations (amber triangles).
3. Gain zoom controls (1x, 1.5x, 2x, 3x, Bigger, Smaller, Reset).
4. Dual respiratory landmark alignment (Diaphragm EMG onsets & updated integrated Diaphragm I_off, 959 breath cycles).
5. Circular respiratory phase vectors, ISI distributions, ACG, stationarity, and waveform kinetics.
6. SpikeInterface bipartite cross-sorter consensus matching (64 consensus units tagged in both suites).
7. Independent cluster validation browsers for both Kilosort 4 and Lupin with cross-navigation.
8. Top-level Validation Hub landing page linking both suites and subwindow benchmarks.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.signal as signal
import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.comparison as sc

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

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

# Experiment & Path Definitions
EXPERIMENT_ID = "e097_g2"
WINDOW_ID = "intact_window"
WINDOW_NAME = "Intact Window (420.0s Eupnea & Anoxia)"
START_T_S = 1740.0
DURATION_S = 420.0
FS_NIDQ = 10000.0
FS_AP = 29999.821974390747
N_CHANS_AP = 385

ROOT_DIR = Path(r"c:\Users\lpascu\resp_analysis")
INTACT_DIR = ROOT_DIR / "outputs" / "e097" / WINDOW_ID
KS4_DIR = INTACT_DIR / "kilosort4"
LUPIN_DIR = INTACT_DIR / "lupin"

RAW_AP_BIN = Path(r"C:\SCRATCH_neuropixels\catgt_e097_intact_420s\catgt_e097_g2\e097_g2_tcat.imec0.ap.bin")
CATGT_FOLDER = RAW_AP_BIN.parent
NIDQ_CACHE = INTACT_DIR / "intact_nidq_cache.npz"

TOP_VAL_DIR = INTACT_DIR / "sorting_validation"
KS4_VAL_DIR = KS4_DIR / "sorting_validation"
LUPIN_VAL_DIR = LUPIN_DIR / "sorting_validation"


def load_nidaq_bursts() -> Tuple[np.ndarray, np.ndarray, str]:
    """Loads breath onsets and offsets from preprocessed intact NIDQ cache (integrated diaphragm I_off)."""
    print(f"Loading NIDAQ landmarks from cache: {NIDQ_CACHE}...")
    d = np.load(NIDQ_CACHE)
    valid_onsets = d["valid_onsets"]
    valid_offsets = d["valid_offsets"]
    prov = "Diaphragm EMG Burst Detection (Integrated Diaphragm I_off)"
    print(f"Loaded {len(valid_onsets)} breath cycles across {DURATION_S}s window (~{len(valid_onsets)/(DURATION_S/60):.1f} bpm).")
    return valid_onsets, valid_offsets, prov


def load_probe_geometry() -> Tuple[Dict[int, float], Dict[float, int]]:
    """Loads physical Neuropixels channel depths and creates depth lookup."""
    try:
        rec = se.read_spikeglx(str(CATGT_FOLDER), stream_id="imec0.ap")
        locs = rec.get_channel_locations()
        channel_depths = {ch: float(locs[ch, 1]) for ch in range(len(locs))}
        if 384 not in channel_depths:
            channel_depths[384] = 0.0
        print(f"Loaded Neuropixels probe geometry: {len(locs)} channels (Span: {locs[:,1].min():.0f}um to {locs[:,1].max():.0f}um)")
    except Exception as e:
        print(f"Fallback probe geometry ({e}): using standard spacing.")
        channel_depths = {ch: float((ch // 2) * 15.0) for ch in range(385)}

    return channel_depths


def compute_cross_sorter_consensus() -> Tuple[Dict[int, Tuple[int, float]], Dict[int, Tuple[int, float]]]:
    """Computes bipartite SpikeInterface mutual agreement between Lupin and Kilosort 4."""
    print("Computing bipartite cross-sorter consensus...")
    ks4_st = np.load(KS4_DIR / "alf" / "spike_times.npy")
    ks4_sc = np.load(KS4_DIR / "alf" / "spike_clusters.npy")
    lup_st = np.load(LUPIN_DIR / "alf" / "spike_times.npy")
    lup_sc = np.load(LUPIN_DIR / "alf" / "spike_clusters.npy")

    sort_ks4 = si.NumpySorting.from_times_and_labels((ks4_st * FS_AP).astype(np.int64), ks4_sc, sampling_frequency=FS_AP)
    sort_lup = si.NumpySorting.from_times_and_labels((lup_st * FS_AP).astype(np.int64), lup_sc, sampling_frequency=FS_AP)

    cmp = sc.compare_two_sorters(sort_lup, sort_ks4, sorting1_name="lupin", sorting2_name="kilosort4", delta_time=0.4)
    agr = cmp.agreement_scores

    lup_to_ks4 = {}
    for u_lup in agr.index:
        best_ks4 = agr.loc[u_lup].idxmax()
        score = float(agr.loc[u_lup, best_ks4])
        if score >= 0.20:
            lup_to_ks4[int(u_lup)] = (int(best_ks4), score)

    ks4_to_lup = {}
    for u_ks4 in agr.columns:
        best_lup = agr[u_ks4].idxmax()
        score = float(agr.loc[best_lup, u_ks4])
        if score >= 0.20:
            ks4_to_lup[int(u_ks4)] = (int(best_lup), score)

    n_cons_lup = sum(1 for _, s in lup_to_ks4.values() if s >= 0.50)
    n_cons_ks4 = sum(1 for _, s in ks4_to_lup.values() if s >= 0.50)
    print(f"Cross-sorter consensus: {n_cons_lup} Lupin units and {n_cons_ks4} KS4 units matched with >=50% agreement.")
    
    # Save match mapping table
    matches_summary = {
        "lup_to_ks4": {str(k): [int(v[0]), round(float(v[1]), 4)] for k, v in lup_to_ks4.items()},
        "ks4_to_lup": {str(k): [int(v[0]), round(float(v[1]), 4)] for k, v in ks4_to_lup.items()}
    }
    with open(INTACT_DIR / "cross_sorter_matches.json", "w", encoding="utf-8") as f:
        json.dump(matches_summary, f, indent=2)

    return lup_to_ks4, ks4_to_lup


def build_browser_html(
    units_manifest: List[Dict],
    dest_path: Path,
    sorter_name: str,
    other_sorter_name: str,
    other_suite_relative_url: str
):
    """Generates the multi-cluster validation browser matching LP21 architecture."""
    manifest_json = json.dumps(units_manifest)
    first_file = units_manifest[0]["file_name"] if len(units_manifest) > 0 else ""

    n_all = len(units_manifest)
    n_consensus = sum(1 for u in units_manifest if u.get("is_consensus", False))
    n_strict = sum(1 for u in units_manifest if u.get("passed_allen_strict", False))
    n_resp = sum(1 for u in units_manifest if u.get("passed_respiratory_continuous", False))
    n_mua = sum(1 for u in units_manifest if u.get("is_mua", False))
    n_fail = sum(1 for u in units_manifest if not u.get("passed_allen_strict", False) and not u.get("passed_respiratory_continuous", False))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E097 Intact Window (420s) {sorter_name} Spike Sorting Validation Browser</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        .navbar {{
            background: #1e293b;
            border-bottom: 1px solid #334155;
            padding: 10px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            z-index: 100;
        }}
        .nav-left {{
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }}
        .nav-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .selector-box {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        select {{
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            outline: none;
            max-width: 520px;
        }}
        select:focus {{
            border-color: #38bdf8;
        }}
        .btn {{
            background: #334155;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.80rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .btn:hover {{
            background: #475569;
        }}
        .btn-active {{
            background: #0284c7 !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
        }}
        .btn-consensus {{
            border-color: #a855f7 !important;
            color: #c084fc !important;
        }}
        .btn-consensus.btn-active {{
            background: #7e22ce !important;
            color: #ffffff !important;
        }}
        .nav-links {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-left: auto;
        }}
        .link-badge {{
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
        }}
        .link-badge:hover {{
            background: #0284c7;
        }}
        .link-switch {{
            background: #4c1d95;
            border-color: #7c3aed;
        }}
        .link-switch:hover {{
            background: #6d28d9;
        }}
        .iframe-container {{
            flex: 1;
            width: 100%;
            height: 100%;
            border: none;
            background: #f8fafc;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .shortcuts-hint {{
            font-size: 0.78rem;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        kbd {{
            background: #0f172a;
            border: 1px solid #475569;
            border-radius: 3px;
            padding: 2px 5px;
            font-size: 0.72rem;
            color: #cbd5e1;
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <div class="nav-left">
            <span class="nav-title">
                <span>🔬</span> {sorter_name} Validation Browser (Intact 420s)
            </span>
            <div class="selector-box">
                <label for="unitSelect" style="font-size:0.85rem;color:#94a3b8;">Cluster:</label>
                <select id="unitSelect" onchange="loadSelectedUnit()"></select>
                <button class="btn" onclick="prevUnit()" title="Previous Unit (Left Arrow or J)">&#9664; Prev</button>
                <button class="btn" onclick="nextUnit()" title="Next Unit (Right Arrow or K)">Next &#9654;</button>
            </div>
            <div class="selector-box">
                <button class="btn btn-active" id="filterAll" onclick="applyFilter('all')">All ({n_all})</button>
                <button class="btn btn-consensus" id="filterConsensus" onclick="applyFilter('consensus')">Consensus ({n_consensus})</button>
                <button class="btn" id="filterStrict" onclick="applyFilter('strict')">Strict SUA ({n_strict})</button>
                <button class="btn" id="filterResp" onclick="applyFilter('resp')">Continuous SUA ({n_resp})</button>
                <button class="btn" id="filterMua" onclick="applyFilter('mua')">MUA / Candidate ({n_mua})</button>
                <button class="btn" id="filterFail" onclick="applyFilter('fail')">Failed QC ({n_fail})</button>
            </div>
        </div>
        <div class="nav-links">
            <div class="shortcuts-hint">
                <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> or <kbd>J</kbd> / <kbd>K</kbd> to cycle
            </div>
            <a href="{other_suite_relative_url}" class="link-badge link-switch">🔄 Switch to {other_sorter_name} Suite &rarr;</a>
            <a href="../../intact_window_raw_emg_flow_interactive.html" target="_blank" class="link-badge">🫁 Interactive EMG & Flow</a>
            <a href="../../intact_window_population_dynamics_dashboard.html" target="_blank" class="link-badge">📊 Population Dynamics</a>
            <a href="../../sorting_validation/index.html" class="link-badge" style="background:#334155;border-color:#64748b;">🏠 Hub</a>
        </div>
    </div>
    <div class="iframe-container">
        <iframe id="unitFrame" src="{first_file}"></iframe>
    </div>

    <script>
    const manifest = {manifest_json};
    let currentFilter = 'all';
    let filteredUnits = [...manifest];
    let currentIndex = 0;

    const selectEl = document.getElementById('unitSelect');
    const frameEl = document.getElementById('unitFrame');

    function populateDropdown() {{
        selectEl.innerHTML = '';
        filteredUnits.forEach((u, idx) => {{
            const opt = document.createElement('option');
            opt.value = idx;
            let statusTag = u.qc_label;
            opt.textContent = `Unit ${{String(u.cluster_id).padStart(4, '0')}} | Ch ${{u.peak_channel}} (${{Math.round(u.depth_um)}}um) | ${{u.firing_rate_hz.toFixed(1)}}Hz | ${{statusTag}}`;
            selectEl.appendChild(opt);
        }});
        if (filteredUnits.length > 0) {{
            currentIndex = 0;
            selectEl.selectedIndex = 0;
            frameEl.src = filteredUnits[0].file_name;
        }} else {{
            frameEl.src = 'about:blank';
        }}
    }}

    function loadSelectedUnit() {{
        const idx = parseInt(selectEl.value, 10);
        if (idx >= 0 && idx < filteredUnits.length) {{
            currentIndex = idx;
            frameEl.src = filteredUnits[currentIndex].file_name;
        }}
    }}

    function nextUnit() {{
        if (filteredUnits.length === 0) return;
        currentIndex = (currentIndex + 1) % filteredUnits.length;
        selectEl.selectedIndex = currentIndex;
        frameEl.src = filteredUnits[currentIndex].file_name;
    }}

    function prevUnit() {{
        if (filteredUnits.length === 0) return;
        currentIndex = (currentIndex - 1 + filteredUnits.length) % filteredUnits.length;
        selectEl.selectedIndex = currentIndex;
        frameEl.src = filteredUnits[currentIndex].file_name;
    }}

    function applyFilter(filterType) {{
        currentFilter = filterType;
        document.querySelectorAll('.btn').forEach(b => {{
            if (['filterAll', 'filterConsensus', 'filterStrict', 'filterResp', 'filterMua', 'filterFail'].includes(b.id)) {{
                b.classList.remove('btn-active');
            }}
        }});

        if (filterType === 'all') {{
            document.getElementById('filterAll').classList.add('btn-active');
            filteredUnits = [...manifest];
        }} else if (filterType === 'consensus') {{
            document.getElementById('filterConsensus').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.is_consensus);
        }} else if (filterType === 'strict') {{
            document.getElementById('filterStrict').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_allen_strict);
        }} else if (filterType === 'resp') {{
            document.getElementById('filterResp').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.passed_respiratory_continuous);
        }} else if (filterType === 'mua') {{
            document.getElementById('filterMua').classList.add('btn-active');
            filteredUnits = manifest.filter(u => u.is_mua);
        }} else if (filterType === 'fail') {{
            document.getElementById('filterFail').classList.add('btn-active');
            filteredUnits = manifest.filter(u => !u.passed_allen_strict && !u.passed_respiratory_continuous);
        }}

        populateDropdown();
    }}

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'ArrowRight' || e.key === 'k' || e.key === 'K') {{
            nextUnit();
        }} else if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {{
            prevUnit();
        }}
    }});

    // Listen for iframe navigation events if any
    window.addEventListener('message', (event) => {{
        if (event.data && event.data.action === 'nextUnit') {{
            nextUnit();
        }} else if (event.data && event.data.action === 'prevUnit') {{
            prevUnit();
        }}
    }});

    // Initialize
    populateDropdown();
    </script>
</body>
</html>
"""
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built interactive validation browser: {dest_path}")


def process_sorter_validation_suite(
    sorter_name: str,
    sorter_dir: Path,
    val_out_dir: Path,
    mm_ap: np.memmap,
    channel_depths: Dict[int, float],
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    prov: str,
    consensus_map: Dict[int, Tuple[int, float]],
    other_sorter_name: str,
    other_suite_url: str
) -> List[Dict]:
    """Processes all clusters for a sorter and creates single unit pages & browser."""
    print("=" * 80)
    print(f" PROCESSING VALIDATION SUITE FOR: {sorter_name.upper()}")
    print(f" Source: {sorter_dir}")
    print(f" Dest: {val_out_dir}")
    print("=" * 80)

    val_out_dir.mkdir(parents=True, exist_ok=True)
    units_dir = val_out_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ALF spikes & cluster info
    spk_times = np.load(sorter_dir / "alf" / "spike_times.npy")
    spk_clusts = np.load(sorter_dir / "alf" / "spike_clusters.npy")
    df_clusters = pd.read_csv(sorter_dir / "cluster_info.tsv", sep="\t")
    print(f"Loaded {len(df_clusters)} clusters and {len(spk_times):,} spikes.")

    # Index spike times by cluster_id
    sort_idx = np.argsort(spk_clusts)
    sorted_clusts = spk_clusts[sort_idx]
    sorted_times = spk_times[sort_idx]
    unique_c, split_idx = np.unique(sorted_clusts, return_index=True)
    splits = np.split(sorted_times, split_idx[1:])
    cluster_spike_dict = {int(c): np.sort(st) for c, st in zip(unique_c, splits)}

    # Depth lookup inverse for channel candidates
    all_depths = np.array(list(channel_depths.values()))
    all_chans = np.array(list(channel_depths.keys()))

    # Sort priority: Consensus first, then passing SUA, then FR
    def sort_priority(row):
        uid = int(row["cluster_id"])
        is_cons = uid in consensus_map and consensus_map[uid][1] >= 0.50
        passed_allen = bool(row.get("passed_allen_qc", False))
        fr = float(row.get("firing_rate_hz", 0.0))
        if is_cons:
            return (0, -consensus_map[uid][1], -fr)
        if passed_allen:
            return (1, 0, -fr)
        return (2, 0, -fr)

    df_clusters["sort_key"] = [sort_priority(r) for _, r in df_clusters.iterrows()]
    df_sorted = df_clusters.sort_values(by="sort_key").reset_index(drop=True)

    manifest = []
    t_start_all = time.time()
    n_total = len(df_sorted)

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        t_u0 = time.time()
        uid = int(row["cluster_id"])
        fr = float(row.get("firing_rate_hz", 0.0))
        passed_allen = bool(row.get("passed_allen_qc", False))
        depth_um = float(row.get("depth_um", 0.0))

        u_spk_times = cluster_spike_dict.get(uid, np.array([], dtype=np.float64))
        if len(u_spk_times) < 5:
            # Skip units with negligible spike count
            continue

        # Candidate channels around depth
        d_diff = np.abs(all_depths - depth_um)
        est_ch = all_chans[np.argmin(d_diff)]
        cands = np.clip(np.arange(est_ch - 25, est_ch + 26), 0, 383)

        # Footprint channels & peak soma
        peak_chan, footprint_chans, channel_ptps, template_wfs = find_unit_channel_footprint(
            mm_ap=mm_ap,
            spike_times_s=u_spk_times,
            fs=FS_AP,
            candidate_channels=cands,
            n_footprint_channels=8,
            max_sample_spikes=100,
        )
        true_depth_um = float(channel_depths.get(peak_chan, depth_um))

        # 1.5s representative trace snippet
        snippet = extract_representative_trace_snippet(
            mm_ap=mm_ap,
            channels=footprint_chans,
            channel_depths=channel_depths,
            spike_times_s=u_spk_times,
            peak_channel=peak_chan,
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
        n_samples = mm_ap.shape[0]
        valid_p_idxs = peak_sample_idxs[(peak_sample_idxs >= 15) & (peak_sample_idxs < n_samples - 30)]
        if len(valid_p_idxs) > 0:
            samp_subset = valid_p_idxs[: min(2000, len(valid_p_idxs))]
            slices = np.vstack([mm_ap[s - 15 : s + 30, peak_chan] for s in samp_subset])
            spk_amps_uv = np.ptp(slices, axis=1).astype(np.float32)
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

        # Cross-sorter consensus status
        is_consensus = False
        cons_partner = -1
        cons_score = 0.0
        if uid in consensus_map:
            cons_partner, cons_score = consensus_map[uid]
            if cons_score >= 0.50:
                is_consensus = True

        # Custom Badge and Label
        is_strict = passed_allen and isi_res.passed_allen_isi and cutoff_res.passed_cutoff_qc
        is_resp = passed_allen and fr >= 2.0 and isi_res.contamination_pct <= 5.0
        is_mua = not is_strict and not is_resp and isi_res.contamination_pct > 10.0

        if is_consensus:
            qc_badge = f'<span style="background-color:#581c87;color:#c084fc;border:1px solid #7e22ce;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">CONSENSUS ({other_sorter_name} #{cons_partner} • {cons_score*100:.0f}% match)</span>'
            qc_label = f"Consensus with {other_sorter_name} #{cons_partner} ({cons_score*100:.0f}%)"
            overall_pass = True
        elif is_strict:
            qc_badge = '<span style="background-color:#10b981;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Strict Allen SUA)</span>'
            qc_label = "Strict Allen SUA [PASS]"
            overall_pass = True
        elif is_resp:
            qc_badge = '<span style="background-color:#0284c7;color:#ffffff;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">PASS (Continuous Resp SUA)</span>'
            qc_label = "Continuous Resp SUA [PASS]"
            overall_pass = True
        elif is_mua:
            qc_badge = f'<span style="background-color:#78350f;color:#fcd34d;border:1px solid #d97706;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">MUA / Contaminated ({isi_res.contamination_pct:.1f}% viol)</span>'
            qc_label = f"MUA ({isi_res.contamination_pct:.1f}% viol)"
            overall_pass = False
        else:
            qc_badge = f'<span style="background-color:#7f1d1d;color:#f87171;border:1px solid #dc2626;padding:4px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;">FAIL (QC Rejected)</span>'
            qc_label = "QC Rejected [FAIL]"
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
        bundle.custom_cli_dir = str(val_out_dir.relative_to(PROJECT_ROOT) if val_out_dir.is_relative_to(PROJECT_ROOT) else val_out_dir)

        unit_file_name = f"unit_{uid:04d}_ch{peak_chan}.html"
        unit_file_path = units_dir / unit_file_name
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
            "is_consensus": is_consensus,
            "is_mua": is_mua,
            "consensus_partner": cons_partner,
            "consensus_score": round(cons_score, 3),
            "qc_label": qc_label,
            "file_name": f"units/{unit_file_name}",
        })

        t_elapsed = time.time() - t_u0
        if (i + 1) % 25 == 0 or (i + 1) == n_total or is_consensus:
            print(f"[{i+1:04d}/{n_total:04d}] Unit {uid:04d} | Ch {peak_chan:03d} ({true_depth_um:.0f}um) | {fr:.1f}Hz | {qc_label} ({t_elapsed:.2f}s)")

    # Build Unified Browser
    browser_path = val_out_dir / "cluster_validation_browser.html"
    build_browser_html(
        units_manifest=manifest,
        dest_path=browser_path,
        sorter_name=sorter_name,
        other_sorter_name=other_sorter_name,
        other_suite_relative_url=other_suite_url
    )

    t_total = time.time() - t_start_all
    print(f"[{sorter_name}] Completed {len(manifest)} units in {t_total:.1f}s ({t_total/max(1, len(manifest)):.2f}s/unit).")
    return manifest


def build_top_level_validation_hub(
    ks4_manifest: List[Dict],
    lupin_manifest: List[Dict],
    dest_path: Path
):
    """Builds a top-level hub landing page for the intact window validation suites."""
    n_ks4_tot = len(ks4_manifest)
    n_ks4_sua = sum(1 for u in ks4_manifest if u.get("passed_allen_strict", False) or u.get("passed_respiratory_continuous", False))
    n_ks4_cons = sum(1 for u in ks4_manifest if u.get("is_consensus", False))

    n_lup_tot = len(lupin_manifest)
    n_lup_sua = sum(1 for u in lupin_manifest if u.get("passed_allen_strict", False) or u.get("passed_respiratory_continuous", False))
    n_lup_cons = sum(1 for u in lupin_manifest if u.get("is_consensus", False))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>E097 Intact Window (420.0s) Validation Suite Hub</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {{
            background: #07090e;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 24px 30px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 26px;
            color: #38bdf8;
            font-weight: 800;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
            margin: 0;
            line-height: 1.6;
        }}
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #0f1523;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .card-title {{
            font-size: 20px;
            font-weight: 700;
            margin: 0;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
        }}
        .badge-ks4 {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .badge-lupin {{ background: rgba(0, 245, 212, 0.2); color: #00f5d4; border: 1px solid #00f5d4; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 16px 0;
            background: rgba(0,0,0,0.25);
            padding: 14px;
            border-radius: 10px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .stat-val {{
            font-size: 22px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }}
        .btn-launch {{
            display: block;
            text-align: center;
            background: #0284c7;
            color: #ffffff;
            font-weight: 700;
            font-size: 14px;
            padding: 12px 20px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 16px;
            transition: background 0.15s ease;
        }}
        .btn-launch:hover {{
            background: #0369a1;
        }}
        .btn-lupin {{
            background: #0f766e;
        }}
        .btn-lupin:hover {{
            background: #115e59;
        }}
        .consensus-banner {{
            background: linear-gradient(135deg, rgba(88, 28, 135, 0.4), rgba(126, 34, 206, 0.2));
            border: 1px solid #7e22ce;
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .consensus-text h3 {{
            margin: 0 0 6px 0;
            font-size: 17px;
            color: #d8b4fe;
        }}
        .consensus-text p {{
            margin: 0;
            font-size: 13px;
            color: #c084fc;
        }}
        .consensus-num {{
            font-size: 32px;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            color: #f3e8ff;
        }}
        .links-row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .link-pill {{
            background: #1e293b;
            color: #38bdf8;
            padding: 8px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid rgba(56, 189, 248, 0.25);
            transition: background 0.15s;
        }}
        .link-pill:hover {{
            background: #38bdf8;
            color: #07090e;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>E097 Intact Window (420.0s) Sorting Validation Hub</h1>
            <p class="subtitle">
                Temporal Range: 1740.0s – 2160.0s (420.0s Intact Eupnea, Anoxia Challenge, Apnea, & Recovery) | Dual-Sorter Rigorous Quality Assurance<br>
                Interactive Phy-Like Trace Inspection • Exact AP Trough Snapping • Refractory ISI Violations • Respiratory Phase Tuning (Diaphragm I_off)
            </p>
        </div>

        <div class="consensus-banner">
            <div class="consensus-text">
                <h3>🤝 Cross-Sorter Mutual Consensus Agreement</h3>
                <p>SpikeInterface bipartite spike-train cross-matching (&Delta;t = 0.4 ms, agreement &ge; 50%). Mutual biological confirmation across independent algorithms.</p>
            </div>
            <div class="consensus-num">{n_lup_cons} Units ({n_lup_cons/max(1, n_lup_tot)*100:.1f}%)</div>
        </div>

        <div class="grid-cards">
            <!-- Kilosort 4 Card -->
            <div class="card">
                <div>
                    <div class="card-header">
                        <h2 class="card-title" style="color: #fbbf24;">Kilosort 4 Suite</h2>
                        <span class="badge badge-ks4">GPU Accelerated</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 13px; margin: 0 0 14px 0;">
                        Full cluster inventory ({n_ks4_tot} total units) with Allen Institute ISI, amplitude cutoff, footprint SNR metrics, and respiratory phase distributions.
                    </p>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Total Units</div>
                            <div class="stat-val" style="color: #f1f5f9;">{n_ks4_tot}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Clean SUA</div>
                            <div class="stat-val" style="color: #10b981;">{n_ks4_sua}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Consensus</div>
                            <div class="stat-val" style="color: #c084fc;">{n_ks4_cons}</div>
                        </div>
                    </div>
                </div>
                <div>
                    <a href="../kilosort4/sorting_validation/cluster_validation_browser.html" class="btn-launch">
                        🚀 Launch Kilosort 4 Validation Browser ({n_ks4_tot} Units) &rarr;
                    </a>
                </div>
            </div>

            <!-- Lupin Card -->
            <div class="card">
                <div>
                    <div class="card-header">
                        <h2 class="card-title" style="color: #00f5d4;">Lupin (SVD-OMP) Suite</h2>
                        <span class="badge badge-lupin">Sparse Pursuit</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 13px; margin: 0 0 14px 0;">
                        Spatial-temporal sparse decomposition footprint inventory ({n_lup_tot} total units) with high SUA yield and robust unit stability.
                    </p>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-label">Total Units</div>
                            <div class="stat-val" style="color: #f1f5f9;">{n_lup_tot}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Clean SUA</div>
                            <div class="stat-val" style="color: #10b981;">{n_lup_sua}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Consensus</div>
                            <div class="stat-val" style="color: #c084fc;">{n_lup_cons}</div>
                        </div>
                    </div>
                </div>
                <div>
                    <a href="../lupin/sorting_validation/cluster_validation_browser.html" class="btn-launch btn-lupin">
                        🚀 Launch Lupin Validation Browser ({n_lup_tot} Units) &rarr;
                    </a>
                </div>
            </div>
        </div>

        <div style="background: #0f1523; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px 24px;">
            <h3 style="margin: 0 0 14px 0; font-size: 15px; color: #94a3b8; text-transform: uppercase;">Direct Analysis Shortcuts</h3>
            <div class="links-row">
                <a href="../intact_window_raw_emg_flow_interactive.html" class="link-pill">🫁 Interactive EMG & Airflow Dynamics</a>
                <a href="../intact_window_population_dynamics_dashboard.html" class="link-pill">📊 Intact Population Dynamics & Dropout</a>
                <a href="../intact_window_6state_population_dynamics_dashboard.html" class="link-pill">🧠 6-State Microstate Dynamics</a>
                <a href="../intact_window_cross_sorter_6state_concordance.png" target="_blank" class="link-pill">🤝 Cross-Sorter Concordance</a>
                <a href="../kilosort4/kilosort4_qc_interactive_raster.html" class="link-pill">📈 KS4 Full-Window QC Raster</a>
                <a href="../lupin/lupin_qc_interactive_raster.html" class="link-pill">📈 Lupin Full-Window QC Raster</a>
            </div>
        </div>
    </div>
</body>
</html>
"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    # Also create cluster_validation_browser.html at top level as copy
    alt_dest = dest_path.parent / "cluster_validation_browser.html"
    with open(alt_dest, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Built top-level validation hub: {dest_path}")


def main():
    print("=" * 80)
    print(" E097 INTACT WINDOW (420.0s) DUAL VALIDATION SUITE GENERATOR")
    print(f" Time Range: {START_T_S}s - {START_T_S + DURATION_S}s ({DURATION_S}s)")
    print("=" * 80)

    # 1. Load Diaphragm EMG bursts
    breath_onsets, breath_offsets, prov = load_nidaq_bursts()

    # 2. Load Probe Geometry
    channel_depths = load_probe_geometry()

    # 3. Cross-Sorter Consensus Mapping
    lup_to_ks4, ks4_to_lup = compute_cross_sorter_consensus()

    # 4. Open AP Binary Memmap
    ap_bytes = RAW_AP_BIN.stat().st_size
    n_samples = ap_bytes // (2 * N_CHANS_AP)
    print(f"Opening AP Binary: {RAW_AP_BIN.name} ({n_samples:,} samples, {n_samples / FS_AP:.2f}s)")
    mm_ap = np.memmap(RAW_AP_BIN, dtype=np.int16, mode="r", shape=(n_samples, N_CHANS_AP))

    # 5. Process Lupin Validation Suite (273 units)
    lupin_manifest = process_sorter_validation_suite(
        sorter_name="Lupin (SVD-OMP)",
        sorter_dir=LUPIN_DIR,
        val_out_dir=LUPIN_VAL_DIR,
        mm_ap=mm_ap,
        channel_depths=channel_depths,
        breath_onsets=breath_onsets,
        breath_offsets=breath_offsets,
        prov=prov,
        consensus_map=lup_to_ks4,
        other_sorter_name="Kilosort 4",
        other_suite_url="../../kilosort4/sorting_validation/cluster_validation_browser.html"
    )

    # 6. Process Kilosort 4 Validation Suite (1,109 units)
    ks4_manifest = process_sorter_validation_suite(
        sorter_name="Kilosort 4",
        sorter_dir=KS4_DIR,
        val_out_dir=KS4_VAL_DIR,
        mm_ap=mm_ap,
        channel_depths=channel_depths,
        breath_onsets=breath_onsets,
        breath_offsets=breath_offsets,
        prov=prov,
        consensus_map=ks4_to_lup,
        other_sorter_name="Lupin",
        other_suite_url="../../lupin/sorting_validation/cluster_validation_browser.html"
    )

    # 7. Build Top-Level Hub
    hub_path = TOP_VAL_DIR / "index.html"
    build_top_level_validation_hub(
        ks4_manifest=ks4_manifest,
        lupin_manifest=lupin_manifest,
        dest_path=hub_path
    )

    print("=" * 80)
    print(" ALL INTACT VALIDATION SUITES SUCCESSFULLY GENERATED!")
    print(f" Top-Level Hub: file:///{hub_path.resolve().as_posix()}")
    print(f" Kilosort 4 Suite: file:///{(KS4_VAL_DIR / 'cluster_validation_browser.html').resolve().as_posix()}")
    print(f" Lupin Suite: file:///{(LUPIN_VAL_DIR / 'cluster_validation_browser.html').resolve().as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
