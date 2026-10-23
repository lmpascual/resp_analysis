#!/usr/bin/env python3
"""
Full Multi-Criteria QC & De-Duplication Suite for E097 Diving Reflex
Evaluates all 400 clusters across multiple QC criteria sets:
1. Strict Allen Institute SUA Standard
2. Continuous Respiratory Baseline SUA
3. Gasp / Apnea Recruited Unit Criteria
4. De-Duplication Coincidence Analysis (coincident spikes within 0.4ms on nearby channels)
5. Explicit Failure Reasons (e.g. Low Presence Ratio, Refractory Violations, Low FR, Low SNR)

Outputs:
- Enriched cluster_info.tsv (for Phy GUI and downstream analysis)
- Enriched cluster_qc_summary.csv
- Interactive Multi-Cluster Validation Browser: sorting_validation/cluster_validation_browser.html
"""
import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

out_dir = PROJECT_ROOT / "outputs" / "e097" / "output_e097_diving_reflex_240s_full_probe"
alf_dir = out_dir / "alf"
phy_dir = out_dir / "phy"
valid_dir = out_dir / "sorting_validation"
valid_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(" COMPUTING MULTI-CRITERIA QC & DE-DUPLICATION SUITE")
print("=" * 80)

# Load data
st = np.load(alf_dir / "spike_times.npy")
sc = np.load(alf_dir / "spike_clusters.npy")
df = pd.read_csv(out_dir / "cluster_qc_summary.csv")

duration_s = 240.0
delta_s = 0.0004  # 0.4 ms coincidence window
c_ids = df["cluster_id"].values

# Ingest NIDQ for diaphragm alignment
raw_nidq_dir = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2")
nidq_bin_file = raw_nidq_dir / "e097_g2_t0.nidq.bin"
fs_nidq = 9999.955307
s0_nidq = int(3300.0 * fs_nidq)
nsamp_nidq = int(duration_s * fs_nidq)
n_ch_nidq = 9

mm_nidq = np.memmap(nidq_bin_file, dtype=np.int16, mode='r', shape=(int(nidq_bin_file.stat().st_size / (n_ch_nidq * 2)), n_ch_nidq))
analog_raw = mm_nidq[s0_nidq:s0_nidq+nsamp_nidq, :].astype(np.float32) * (10.0 / 32768.0)
emg_raw = analog_raw[:, 0]
flow_raw = analog_raw[:, 5]

# Filter EMG for onsets
b_emg, a_emg = signal.butter(4, [300.0 / (fs_nidq / 2.0), 3000.0 / (fs_nidq / 2.0)], btype="bandpass")
emg_bp = signal.filtfilt(b_emg, a_emg, emg_raw)
win_len = int(0.050 * fs_nidq)
emg_smooth = np.convolve(np.abs(emg_bp), np.ones(win_len)/win_len, mode='same')
peaks_emg, _ = signal.find_peaks(emg_smooth, height=np.median(emg_smooth) + 2.0*np.std(emg_smooth), distance=int(0.12 * fs_nidq))
t_onsets_s = np.linspace(0.0, duration_s, nsamp_nidq, endpoint=False)[peaks_emg]

print(f"Loaded {len(df)} clusters ({len(st):,} spikes), {len(t_onsets_s)} diaphragm breath onsets.")

# Pre-group spikes by cluster
cluster_spikes = {cid: st[sc == cid] for cid in c_ids}
cluster_depths = {cid: depth for cid, depth in zip(df['cluster_id'], df['depth_um'])}

# -------------------------------------------------------------------------
# 1. De-Duplication Coincidence Analysis
# -------------------------------------------------------------------------
print("Computing pairwise de-duplication coincidence...")
dedup_records = {}
for cid_a in c_ids:
    t_a = cluster_spikes[cid_a]
    d_a = cluster_depths[cid_a]
    n_a = len(t_a)

    max_coinc = 0.0
    best_partner = -1
    best_matched = 0
    best_jaccard = 0.0

    if n_a > 0:
        for cid_b in c_ids:
            if cid_a == cid_b:
                continue
            d_b = cluster_depths[cid_b]
            if abs(d_a - d_b) > 80.0:  # Channels within 80 um
                continue
            t_b = cluster_spikes[cid_b]
            n_b = len(t_b)
            if n_b == 0:
                continue

            idx_b = np.searchsorted(t_b, t_a)
            matched = 0
            for j, t in enumerate(t_a):
                for cand in [idx_b[j] - 1, idx_b[j]]:
                    if 0 <= cand < n_b and abs(t_b[cand] - t) <= delta_s:
                        matched += 1
                        break

            frac_a = matched / n_a
            jaccard = matched / (n_a + n_b - matched) if (n_a + n_b - matched) > 0 else 0.0

            if frac_a > max_coinc:
                max_coinc = frac_a
                best_partner = cid_b
                best_matched = matched
                best_jaccard = jaccard

    dedup_records[cid_a] = {
        "coincidence_fraction": round(max_coinc, 3),
        "most_coincident_partner": int(best_partner) if best_partner >= 0 else None,
        "coincident_spikes_count": int(best_matched),
        "jaccard_agreement": round(best_jaccard, 3),
        "would_be_deduplicated": bool(best_jaccard >= 0.50 or max_coinc >= 0.50),
        "is_unique_unit": bool(max_coinc < 0.20)
    }

# -------------------------------------------------------------------------
# 2. Multi-Criteria Evaluation & Failure Reason Diagnostics
# -------------------------------------------------------------------------
print("Evaluating multi-criteria QC pass/fail status...")
records = []
for _, row in df.iterrows():
    cid = int(row["cluster_id"])
    u_spks = cluster_spikes[cid]
    n_spks = len(u_spks)
    fr = float(row["firing_rate_hz"])
    pr = float(row["presence_ratio"])
    isi_viol = float(row["isi_violations"])
    snr = float(row["snr"])
    depth = float(row["depth_um"])
    ch = int(row["peak_channel"])

    # Apnea Window Metrics (t = 50.0 to 54.0s)
    n_base = np.sum((u_spks >= 10.0) & (u_spks < 40.0))
    n_apnea = np.sum((u_spks >= 50.0) & (u_spks < 54.0))
    fr_base = round(n_base / 30.0, 3)
    fr_apnea = round(n_apnea / 4.0, 3)
    fold_change = round((fr_apnea + 0.1) / (fr_base + 0.1), 2)

    # 1. Criteria Set: Strict Allen SUA
    pass_allen_strict = (pr >= 0.80) and (isi_viol <= 0.10) and (fr >= 0.10) and (snr >= 2.5)

    # 2. Criteria Set: Continuous Respiratory SUA
    pass_resp_continuous = (pr >= 0.40) and (isi_viol <= 0.10) and (fr >= 0.05) and (snr >= 1.5)

    # 3. Criteria Set: Gasp / Apnea Recruited Candidate
    pass_gasp_recruited = (pr < 0.40) and (fr_apnea >= 5.0 or fold_change >= 3.0) and (n_spks >= 5) and (isi_viol <= 0.15)

    # De-duplication info
    dd = dedup_records[cid]

    # Diagnostic list of failing reasons
    fail_reasons = []
    if pr < 0.80:
        fail_reasons.append(f"Low Presence Ratio ({pr:.2f} < 0.80)")
    if isi_viol > 0.10:
        fail_reasons.append(f"Refractory Violation ({isi_viol:.3f} > 0.10)")
    if fr < 0.10:
        fail_reasons.append(f"Low Firing Rate ({fr:.3f} < 0.10 Hz)")
    if snr < 2.5:
        fail_reasons.append(f"Low SNR ({snr:.2f} < 2.5)")
    if dd["would_be_deduplicated"]:
        fail_reasons.append(f"Deduplication Overlap ({dd['coincidence_fraction']*100:.1f}% with Unit {dd['most_coincident_partner']})")

    fail_reason_str = "; ".join(fail_reasons) if len(fail_reasons) > 0 else "None (Passed All Strict Criteria)"

    # Overall Verdict
    if pass_allen_strict and not dd["would_be_deduplicated"]:
        overall_verdict = "PASS: Strict SUA"
        phy_group = "good"
    elif pass_resp_continuous and not dd["would_be_deduplicated"]:
        overall_verdict = "PASS: Continuous SUA"
        phy_group = "good"
    elif pass_gasp_recruited and not dd["would_be_deduplicated"]:
        overall_verdict = "PASS: Gasp Recruited"
        phy_group = "candidate"
    elif dd["would_be_deduplicated"]:
        overall_verdict = "FAIL: Deduplication Redundant"
        phy_group = "mua"
    elif isi_viol > 0.15:
        overall_verdict = "FAIL: Refractory Violations (MUA)"
        phy_group = "mua"
    elif pr < 0.20:
        overall_verdict = "FAIL: Low Presence Ratio"
        phy_group = "mua"
    else:
        overall_verdict = "FAIL: Multi-Unit / Sparse"
        phy_group = "mua"

    # Respiratory Phase Vector
    mean_r, pref_phase = 0.0, 0.0
    if len(u_spks) >= 10 and len(t_onsets_s) >= 2:
        idx_b = np.searchsorted(t_onsets_s, u_spks)
        valid = (idx_b > 0) & (idx_b < len(t_onsets_s))
        if np.sum(valid) >= 5:
            spk_v = u_spks[valid]
            b_prev = t_onsets_s[idx_b[valid] - 1]
            b_next = t_onsets_s[idx_b[valid]]
            phase_rad = 2.0 * np.pi * (spk_v - b_prev) / (b_next - b_prev)
            c_mean = np.mean(np.cos(phase_rad))
            s_mean = np.mean(np.sin(phase_rad))
            mean_r = round(float(np.sqrt(c_mean**2 + s_mean**2)), 3)
            pref_phase = round(float(np.degrees(np.arctan2(s_mean, c_mean)) % 360.0), 1)

    records.append({
        "cluster_id": cid,
        "group": phy_group,
        "overall_qc_verdict": overall_verdict,
        "passed_allen_strict": bool(pass_allen_strict),
        "passed_respiratory_continuous": bool(pass_resp_continuous),
        "passed_gasp_recruited": bool(pass_gasp_recruited),
        "would_be_deduplicated": dd["would_be_deduplicated"],
        "coincidence_fraction": dd["coincidence_fraction"],
        "most_coincident_partner": dd["most_coincident_partner"],
        "coincident_spikes_count": dd["coincident_spikes_count"],
        "is_unique_unit": dd["is_unique_unit"],
        "depth_um": depth,
        "peak_channel": ch,
        "firing_rate_hz": fr,
        "n_spikes": n_spks,
        "presence_ratio": pr,
        "isi_violations": isi_viol,
        "snr": snr,
        "apnea_fr_hz": fr_apnea,
        "baseline_fr_hz": fr_base,
        "apnea_fold_change": fold_change,
        "apnea_spikes": int(n_apnea),
        "mean_vector_r": mean_r,
        "preferred_phase_deg": pref_phase,
        "is_rostral_continuum": (depth >= 720.0) and (depth <= 1530.0),
        "fail_reasons": fail_reason_str
    })

df_enriched = pd.DataFrame(records)

# Save enriched cluster_info.tsv and summary CSV
df_enriched.to_csv(out_dir / "cluster_info.tsv", sep="\t", index=False)
df_enriched.to_csv(phy_dir / "cluster_info.tsv", sep="\t", index=False)
df_enriched.to_csv(out_dir / "cluster_qc_summary.csv", index=False)

print("\n=== MULTI-CRITERIA BREAKDOWN ===")
print("Overall QC Verdict Counts:")
print(df_enriched["overall_qc_verdict"].value_counts())
print("\nPassing Criteria Summary:")
print("  - Passed Strict Allen SUA       :", df_enriched["passed_allen_strict"].sum())
print("  - Passed Respiratory Continuous :", df_enriched["passed_respiratory_continuous"].sum())
print("  - Passed Gasp Recruited         :", df_enriched["passed_gasp_recruited"].sum())
print("  - Would Be Deduplicated (>=50%) :", df_enriched["would_be_deduplicated"].sum())
print("  - Fully Unique Units (<20% coinc):", df_enriched["is_unique_unit"].sum())

# -------------------------------------------------------------------------
# 3. Build Interactive Phy-Like Cluster Validation Browser
# -------------------------------------------------------------------------
print("\nGenerating Interactive Phy-Like Multi-Cluster Validation Browser...")

# Downsample analog signals for web browser
step_an = int(len(emg_smooth) / 2400)
t_an_web = np.round(np.linspace(0.0, duration_s, len(emg_smooth))[::step_an], 2).tolist()
emg_web = np.round(emg_smooth[::step_an] * 1000.0, 2).tolist()
flow_web = np.round(flow_raw[::step_an], 3).tolist()

units_manifest = []
for _, row in df_enriched.iterrows():
    cid = int(row["cluster_id"])
    u_spks = cluster_spikes[cid]
    
    # Compute ISI distribution
    if len(u_spks) > 1:
        isis = np.diff(u_spks) * 1000.0  # in ms
        isis = isis[isis <= 100.0]
        hist_counts, hist_edges = np.histogram(isis, bins=50, range=(0.0, 100.0))
        isi_bins = np.round(0.5 * (hist_edges[:-1] + hist_edges[1:]), 1).tolist()
        isi_vals = hist_counts.tolist()
    else:
        isi_bins, isi_vals = [], []

    # Sample up to 300 spikes for raster preview
    spk_sample = u_spks if len(u_spks) <= 300 else np.random.choice(u_spks, 300, replace=False)
    spk_sample = np.round(np.sort(spk_sample), 3).tolist()

    units_manifest.append({
        "cid": cid,
        "group": row["group"],
        "verdict": row["overall_qc_verdict"],
        "pass_allen": bool(row["passed_allen_strict"]),
        "pass_resp": bool(row["passed_respiratory_continuous"]),
        "pass_gasp": bool(row["passed_gasp_recruited"]),
        "would_dedup": bool(row["would_be_deduplicated"]),
        "coinc_frac": float(row["coincidence_fraction"]),
        "coinc_partner": int(row["most_coincident_partner"]) if row["most_coincident_partner"] is not None else None,
        "depth": float(row["depth_um"]),
        "ch": int(row["peak_channel"]),
        "fr": float(row["firing_rate_hz"]),
        "n_spk": int(row["n_spikes"]),
        "pr": float(row["presence_ratio"]),
        "isi_q": float(row["isi_violations"]),
        "snr": float(row["snr"]),
        "apnea_fr": float(row["apnea_fr_hz"]),
        "apnea_fc": float(row["apnea_fold_change"]),
        "mean_r": float(row["mean_vector_r"]),
        "pref_phase": float(row["preferred_phase_deg"]),
        "is_rostral": bool(row["is_rostral_continuum"]),
        "fail_reasons": row["fail_reasons"],
        "isi_bins": isi_bins,
        "isi_counts": isi_vals,
        "spikes": spk_sample
    })

browser_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E097 Full-Probe Cluster Validation & Quality Assurance Suite</title>
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
        .header {{ background: #1e293b; border-bottom: 1px solid #334155; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; gap: 15px; flex-wrap: wrap; }}
        .title-box {{ font-size: 1.15rem; font-weight: 700; color: #38bdf8; }}
        .filter-group {{ display: flex; gap: 8px; align-items: center; }}
        .btn {{ background: #334155; color: #f8fafc; border: 1px solid #475569; padding: 5px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: all 0.15s ease; }}
        .btn:hover {{ background: #475569; }}
        .btn.active {{ background: #0284c7; border-color: #38bdf8; color: #fff; }}
        select {{ background: #0f172a; color: #f8fafc; border: 1px solid #475569; padding: 6px 12px; border-radius: 6px; font-size: 0.88rem; font-weight: 600; cursor: pointer; outline: none; }}
        select:focus {{ border-color: #38bdf8; }}
        .main-layout {{ flex: 1; display: grid; grid-template-columns: 360px 1fr; overflow: hidden; }}
        .sidebar {{ background: #1e293b; border-right: 1px solid #334155; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }}
        .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }}
        .card-title {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; font-weight: 700; margin-bottom: 10px; }}
        .verdict-tag {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 700; margin-bottom: 12px; }}
        .verdict-pass {{ background: #065f46; color: #34d399; border: 1px solid #059669; }}
        .verdict-candidate {{ background: #831843; color: #f43f5e; border: 1px solid #e11d48; }}
        .verdict-fail {{ background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .metric-cell {{ display: flex; flex-direction: column; }}
        .metric-lbl {{ font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; }}
        .metric-num {{ font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-top: 2px; }}
        .reasons-box {{ background: #182234; border-left: 3px solid #f43f5e; padding: 8px 12px; border-radius: 4px; font-size: 0.82rem; color: #fca5a5; line-height: 1.4; margin-top: 10px; }}
        .workspace {{ background: #0f172a; padding: 16px; display: grid; grid-template-rows: 240px 1fr; gap: 16px; overflow-y: auto; }}
        .plot-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
        .plots-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; height: 100%; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title-box">E097 Diving Reflex: Full-Probe Cluster Validation &amp; QA Suite</div>
        <div class="filter-group">
            <span style="font-size:0.8rem; color:#94a3b8; font-weight:600;">FILTER:</span>
            <button class="btn active" onclick="setFilter('all')">All Clusters (400)</button>
            <button class="btn" onclick="setFilter('pass_allen')">Strict SUA (122)</button>
            <button class="btn" onclick="setFilter('pass_gasp')">Gasp Recruited</button>
            <button class="btn" onclick="setFilter('would_dedup')">Deduplication Overlap</button>
            <button class="btn" onclick="setFilter('rostral')">Rostral Continuum (AP1-AP60)</button>
            <button class="btn" onclick="setFilter('fail')">Failed QC</button>
            <select id="clusterSelect" onchange="loadCluster(parseInt(this.value))"></select>
        </div>
    </div>

    <div class="main-layout">
        <div class="sidebar">
            <div class="card">
                <div class="card-title">Cluster Overview</div>
                <div id="uVerdict" class="verdict-tag"></div>
                <div class="grid-2">
                    <div class="metric-cell"><span class="metric-lbl">Cluster ID</span><span class="metric-num" id="uId">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Depth (y)</span><span class="metric-num" id="uDepth">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Peak Channel</span><span class="metric-num" id="uCh">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Total Spikes</span><span class="metric-num" id="uSpk">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Firing Rate</span><span class="metric-num" id="uFr">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Estimated SNR</span><span class="metric-num" id="uSnr">-</span></div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Criteria Scorecard</div>
                <div style="font-size:0.84rem; display:flex; flex-direction:column; gap:6px;">
                    <div>• <strong>Allen Strict SUA:</strong> <span id="uPassAllen">-</span></div>
                    <div>• <strong>Continuous Respiratory:</strong> <span id="uPassResp">-</span></div>
                    <div>• <strong>Gasp / Apnea Recruited:</strong> <span id="uPassGasp">-</span></div>
                    <div>• <strong>De-Duplication Status:</strong> <span id="uDedup">-</span></div>
                </div>
                <div id="uReasons" class="reasons-box"></div>
            </div>

            <div class="card">
                <div class="card-title">Diving Reflex Dynamics (Apnea Window)</div>
                <div class="grid-2">
                    <div class="metric-cell"><span class="metric-lbl">Apnea Firing Rate</span><span class="metric-num" id="uApneaFr">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Recruitment Surge</span><span class="metric-num" id="uApneaFc">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Presence Ratio</span><span class="metric-num" id="uPr">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Refractory ISI Q</span><span class="metric-num" id="uIsi">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Phase Mod (R)</span><span class="metric-num" id="uPhaseR">-</span></div>
                    <div class="metric-cell"><span class="metric-lbl">Preferred Phase</span><span class="metric-num" id="uPhaseAng">-</span></div>
                </div>
            </div>
        </div>

        <div class="workspace">
            <div class="plot-box" id="plot-multimodal"></div>
            <div class="plots-grid">
                <div class="plot-box" id="plot-isi"></div>
                <div class="plot-box" id="plot-raster"></div>
            </div>
        </div>
    </div>

    <script>
        const allUnits = {json.dumps(units_manifest)};
        const t_an = {json.dumps(t_an_web)};
        const emg_tr = {json.dumps(emg_web)};
        const flow_tr = {json.dumps(flow_web)};

        let currentFilter = 'all';
        let filteredUnits = [...allUnits];

        function setFilter(filt) {{
            currentFilter = filt;
            document.querySelectorAll('.filter-group .btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            if (filt === 'all') filteredUnits = [...allUnits];
            else if (filt === 'pass_allen') filteredUnits = allUnits.filter(u => u.pass_allen);
            else if (filt === 'pass_gasp') filteredUnits = allUnits.filter(u => u.pass_gasp);
            else if (filt === 'would_dedup') filteredUnits = allUnits.filter(u => u.would_dedup);
            else if (filt === 'rostral') filteredUnits = allUnits.filter(u => u.is_rostral);
            else if (filt === 'fail') filteredUnits = allUnits.filter(u => !u.pass_allen && !u.pass_gasp);

            populateSelect();
            if (filteredUnits.length > 0) loadCluster(filteredUnits[0].cid);
        }}

        function populateSelect() {{
            const sel = document.getElementById('clusterSelect');
            sel.innerHTML = '';
            filteredUnits.forEach(u => {{
                const opt = document.createElement('option');
                opt.value = u.cid;
                opt.innerText = `C${{u.cid}} | ${{u.depth}}um | ${{u.verdict.split(':')[0]}}`;
                sel.appendChild(opt);
            }});
        }}

        function loadCluster(cid) {{
            const u = allUnits.find(x => x.cid === cid);
            if (!u) return;
            document.getElementById('clusterSelect').value = cid;

            document.getElementById('uId').innerText = u.cid;
            document.getElementById('uDepth').innerText = `${{u.depth}} um`;
            document.getElementById('uCh').innerText = u.ch;
            document.getElementById('uSpk').innerText = u.n_spk.toLocaleString();
            document.getElementById('uFr').innerText = `${{u.fr.toFixed(2)}} Hz`;
            document.getElementById('uSnr').innerText = u.snr.toFixed(2);

            const vTag = document.getElementById('uVerdict');
            vTag.innerText = u.verdict;
            vTag.className = 'verdict-tag ' + (u.verdict.includes('Strict') || u.verdict.includes('Continuous') ? 'verdict-pass' : (u.verdict.includes('Gasp') ? 'verdict-candidate' : 'verdict-fail'));

            document.getElementById('uPassAllen').innerText = u.pass_allen ? '✅ PASS' : '❌ FAIL';
            document.getElementById('uPassResp').innerText = u.pass_resp ? '✅ PASS' : '❌ FAIL';
            document.getElementById('uPassGasp').innerText = u.pass_gasp ? '✅ PASS (Recruited)' : '—';
            document.getElementById('uDedup').innerText = u.would_dedup ? `⚠️ Overlap ${{u.coinc_frac*100}}% with C${{u.coinc_partner}}` : '✅ Unique (<20% overlap)';

            document.getElementById('uReasons').innerText = u.fail_reasons;

            document.getElementById('uApneaFr').innerText = `${{u.apnea_fr.toFixed(2)}} Hz`;
            document.getElementById('uApneaFc').innerText = `${{u.apnea_fc.toFixed(1)}}x`;
            document.getElementById('uPr').innerText = u.pr.toFixed(2);
            document.getElementById('uIsi').innerText = u.isi_q.toFixed(3);
            document.getElementById('uPhaseR').innerText = u.mean_r.toFixed(3);
            document.getElementById('uPhaseAng').innerText = `${{u.pref_phase.toFixed(1)}}°`;

            renderPlots(u);
        }}

        function renderPlots(u) {{
            // 1. Multimodal trace with unit spikes overlay
            const tracesMm = [
                {{ x: t_an, y: flow_tr, type: 'scatter', mode: 'lines', name: 'Airflow (V)', line: {{ color: '#06b6d4', width: 1 }}, yaxis: 'y1' }},
                {{ x: t_an, y: emg_tr, type: 'scatter', mode: 'lines', name: 'Diaphragm (mV)', line: {{ color: '#10b981', width: 1 }}, yaxis: 'y2' }},
                {{ x: u.spikes, y: u.spikes.map(() => 1), type: 'scatter', mode: 'markers', name: `Spikes (C${{u.cid}})`, marker: {{ symbol: 'line-ns', size: 12, color: '#f43f5e', line: {{ width: 1.5, color: '#f43f5e' }} }}, yaxis: 'y3' }}
            ];
            const layoutMm = {{
                paper_bgcolor: '#1e293b', plot_bgcolor: '#0f172a',
                height: 230, margin: {{ l: 50, r: 20, t: 25, b: 30 }},
                grid: {{ rows: 3, columns: 1, subplots: [['xy1'], ['xy2'], ['xy3']], roworder: 'top to bottom' }},
                yaxis: {{ domain: [0.70, 1.0], title: 'Flow', gridcolor: '#334155' }},
                yaxis2: {{ domain: [0.35, 0.65], title: 'EMG', gridcolor: '#334155' }},
                yaxis3: {{ domain: [0.0, 0.30], title: 'Spikes', showticklabels: false, gridcolor: '#334155' }},
                xaxis: {{ title: 'Window Time (s)', gridcolor: '#334155' }},
                showlegend: false
            }};
            Plotly.newPlot('plot-multimodal', tracesMm, layoutMm, {{ responsive: true }});

            // 2. ISI distribution
            const traceIsi = [{{ x: u.isi_bins, y: u.isi_counts, type: 'bar', marker: {{ color: '#38bdf8' }} }}];
            const layoutIsi = {{
                paper_bgcolor: '#1e293b', plot_bgcolor: '#0f172a',
                title: {{ text: 'Inter-Spike Interval (0-100 ms) • Red Line: 1.0 ms Refractory', font: {{ size: 11, color: '#94a3b8' }} }},
                height: 280, margin: {{ l: 45, r: 20, t: 30, b: 35 }},
                xaxis: {{ title: 'ISI (ms)', gridcolor: '#334155' }},
                yaxis: {{ title: 'Spike Count', gridcolor: '#334155' }},
                shapes: [{{ type: 'line', x0: 1.0, x1: 1.0, y0: 0, y1: Math.max(...u.isi_counts, 10), line: {{ color: '#ef4444', width: 2, dash: 'dash' }} }}]
            }};
            Plotly.newPlot('plot-isi', traceIsi, layoutIsi, {{ responsive: true }});

            // 3. Spikes over time scatter
            const traceScatter = [{{ x: u.spikes, y: u.spikes.map((_, i) => i), type: 'scatter', mode: 'markers', marker: {{ size: 3, color: '#e11d48' }} }}];
            const layoutScatter = {{
                paper_bgcolor: '#1e293b', plot_bgcolor: '#0f172a',
                title: {{ text: 'Cumulative Spiking over 240s Window', font: {{ size: 11, color: '#94a3b8' }} }},
                height: 280, margin: {{ l: 45, r: 20, t: 30, b: 35 }},
                xaxis: {{ title: 'Time (s)', gridcolor: '#334155' }},
                yaxis: {{ title: 'Spike Number', gridcolor: '#334155' }}
            }};
            Plotly.newPlot('plot-raster', traceScatter, layoutScatter, {{ responsive: true }});
        }}

        // Init
        populateSelect();
        if (allUnits.length > 0) loadCluster(allUnits[0].cid);
    </script>
</body>
</html>
"""
with open(valid_dir / "cluster_validation_browser.html", "w", encoding="utf-8") as f:
    f.write(browser_html)

print(f"Interactive Multi-Cluster Browser generated: {valid_dir / 'cluster_validation_browser.html'}")
print("ALL MULTI-CRITERIA & DE-DUPLICATION DELIVERABLES READY!")
