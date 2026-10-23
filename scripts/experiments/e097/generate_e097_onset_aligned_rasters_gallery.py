#!/usr/bin/env python3
"""
Generate Inspiration-Onset Aligned Spike Rasters and PETHs (Sorted by Inspiration Duration T_I)
for All Master Consensus Single Units in e097 (226s continuous recording):
- Aligned to Inspiration Onset (I_on = 0 ms).
- Trials ordered in ascending order by Inspiration Duration T_I.
- Top Panel: Real-Time Spike Raster with Continuous Phase Background:
  * Warm Red = Inspiration (phi in [0, +pi])
  * Cool Blue = Expiration (phi in [-pi, 0])
  * Dark Red Curve = Inspiration Offset (I_off / T_I)
  * Black precision spike ticks
- Bottom Panel: Peri-Event Time Histogram (PETH):
  * Mean Firing Rate (solid blue curve) + (+/- 1 SD) standard deviation shadow
  * Secondary y-axis with synchronized Diaphragm EMG Envelope
- Responsive HTML Visual Gallery:
  * gallery_all_162_neurons.html (All 842 Breaths across 226s recording)
  * gallery_baseline_eupnea_198_breaths.html (198 Eupneic Breaths in 0-60s baseline)
  * Filter chips for Recruited Rostral, Baseline Archetypes, Caudal Motor, and Anatomical Depths.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Add repo root to sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

def compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s):
    """
    Precomputes the continuous respiratory phase phi(t) in [-pi, pi] for each trial.
    Inspiration [on, off] -> [0, pi]
    Expiration [off, next_on] -> [-pi, 0]
    Pre-inspiration -> [-pi, 0]
    """
    n_breaths = len(valid_onsets)
    n_bins = len(time_axis_s)
    phase_grid = np.zeros((n_breaths, n_bins), dtype=np.float32)

    for i in range(n_breaths):
        on = valid_onsets[i]
        off = valid_offsets[i]
        next_on = valid_onsets[i + 1] if i + 1 < n_breaths else on + (off - on) * 3.0
        prev_off = valid_offsets[i - 1] if i > 0 else on - (off - on) * 2.0
        t_eval = on + time_axis_s

        phi = np.full(n_bins, np.nan, dtype=np.float32)

        # Inspiration: [on, off] -> [0, pi]
        insp_m = (t_eval >= on) & (t_eval < off)
        if np.any(insp_m):
            phi[insp_m] = (t_eval[insp_m] - on) / (off - on + 1e-6) * np.pi

        # Expiration: [off, next_on] -> [-pi, 0]
        exp_m = (t_eval >= off) & (t_eval < next_on)
        if np.any(exp_m):
            phi[exp_m] = -np.pi + (t_eval[exp_m] - off) / (next_on - off + 1e-6) * np.pi

        # Pre-inspiration: [prev_off, on] -> [-pi, 0]
        pre_m = t_eval < on
        if np.any(pre_m):
            phi[pre_m] = -np.pi + (t_eval[pre_m] - prev_off) / (on - prev_off + 1e-6) * np.pi

        # Fill any remaining NaNs
        nan_m = np.isnan(phi)
        phi[nan_m] = -np.pi
        phase_grid[i, :] = phi

    return phase_grid

def plot_single_neuron_raster_peth(
    cid, depth, fr_hz, archetype_label,
    u_spks, valid_onsets, valid_offsets, sort_order, sorted_ti_ms,
    phase_grid_sorted, emg_trials_sorted,
    time_axis_s, time_axis_ms, dt_s,
    kernel, out_file
):
    n_breaths = len(sort_order)
    n_bins = len(time_axis_s)

    # Extract spikes for each sorted trial
    spike_scatter_x = []
    spike_scatter_y = []
    rate_trials = np.zeros((n_breaths, n_bins), dtype=np.float32)
    win_pre_s = -time_axis_s[0]
    win_post_s = time_axis_s[-1]

    for row_idx, orig_idx in enumerate(sort_order):
        on = valid_onsets[orig_idx]
        rel_spks = u_spks - on
        in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
        for s in in_win:
            spike_scatter_x.append(s * 1000.0) # ms
            spike_scatter_y.append(row_idx + 1)

        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
        smooth_fr = np.convolve(counts / dt_s, kernel, mode="same")
        rate_trials[row_idx, :] = smooth_fr

    mean_fr = np.mean(rate_trials, axis=0)
    std_fr = np.std(rate_trials, axis=0)

    mean_emg = np.mean(emg_trials_sorted, axis=0)
    emg_min, emg_max = np.min(mean_emg), np.max(mean_emg)
    norm_emg = (mean_emg - emg_min) / (emg_max - emg_min + 1e-12) if emg_max > emg_min else np.zeros_like(mean_emg)

    mean_ti_val_ms = np.mean(sorted_ti_ms)
    min_ti_val_ms = np.min(sorted_ti_ms)
    max_ti_val_ms = np.max(sorted_ti_ms)

    # Build figure
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.3, 1.0], hspace=0.18)

    # Colormap: Cool Blue (-pi) -> White (0) -> Warm Red (+pi)
    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_onset",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    # -------------------------------------------------------------------------
    # Top Panel: Raster with Phase Background & T_I Sorted
    # -------------------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0])
    im_phase = ax_raster.imshow(
        phase_grid_sorted,
        aspect="auto",
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
        origin="lower",
        alpha=0.60,
        interpolation="bilinear"
    )

    # Overlay Spikes
    if len(spike_scatter_x) > 0:
        ax_raster.scatter(
            spike_scatter_x,
            spike_scatter_y,
            marker='|',
            color='black',
            s=40 if n_breaths < 300 else 15,
            linewidth=1.2 if n_breaths < 300 else 0.8,
            alpha=0.92,
            zorder=3,
            label="Spike Times"
        )

    # Overlay Inspiration Offset boundary line (I_off) across sorted trials
    ioff_curve_x = sorted_ti_ms
    ioff_curve_y = np.arange(1, n_breaths + 1)
    ax_raster.plot(
        ioff_curve_x,
        ioff_curve_y,
        color="#b91c1c",
        linestyle="-",
        linewidth=2.2,
        alpha=0.90,
        zorder=4,
        label=r"Inspiration Offset ($I_{\mathrm{off}}$ / $T_I$)"
    )

    # Vertical line at Breath Onset (I_on: 0 ms)
    ax_raster.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)

    ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax_raster.set_ylim(0.5, n_breaths + 0.5)
    ax_raster.set_ylabel(f"Breath Trials (Sorted by $T_I$: {min_ti_val_ms:.0f} → {max_ti_val_ms:.0f} ms)", fontsize=11, fontweight="bold")

    title_str = f"Cluster #{cid} ({depth:.0f} µm | FR: {fr_hz:.1f} Hz)"
    if archetype_label:
        title_str += f" — {archetype_label}"
    title_str += " — Breath-Onset Aligned Spike Raster\n"
    title_str += r"[Trials Ordered by Inspiration Duration $T_I$ | Warm Red = Insp ($\phi \in [0, \pi]$) | Cool Blue = Exp ($\phi \in [-\pi, 0]$)]"
    ax_raster.set_title(title_str, fontsize=11, fontweight="bold")
    ax_raster.grid(False)

    cbar = plt.colorbar(im_phase, ax=ax_raster, fraction=0.03, pad=0.02)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$ (Exp)", r"$-\frac{\pi}{2}$", r"$0$ ($I_{\mathrm{on}}$)", r"$+\frac{\pi}{2}$", r"$+\pi$ ($I_{\mathrm{off}}$)"], fontsize=8)
    ax_raster.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    # -------------------------------------------------------------------------
    # Bottom Panel: PETH with Mean FR + SD Shadow & EMG Reference
    # -------------------------------------------------------------------------
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)
    ax_peth.plot(time_axis_ms, mean_fr, color="#0284c7", linewidth=2.5, label="Mean Firing Rate")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr - std_fr),
        mean_fr + std_fr,
        color="#0284c7",
        alpha=0.25,
        label=r"$\pm 1$ SD Shadow"
    )

    ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.85, label=r"Breath Onset ($I_{\mathrm{on}}$: 0 ms)")
    ax_peth.axvline(mean_ti_val_ms, color="#b91c1c", linestyle=":", linewidth=1.8, alpha=0.90, label=f"Mean $I_{{\\mathrm{{off}}}}$ ({mean_ti_val_ms:.1f} ms)")

    ax_peth.set_xlabel("Time from Breath Onset (ms)", fontsize=11, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (sp/s)", fontsize=11, fontweight="bold", color="#0284c7")
    ax_peth.tick_params(axis='y', labelcolor="#0284c7")
    ax_peth.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis for Diaphragm EMG
    ax_emg = ax_peth.twinx()
    ax_emg.plot(time_axis_ms, norm_emg, color="#475569", linestyle=":", linewidth=2.0, label="Dia. EMG Envelope")
    ax_emg.set_ylabel("Dia. EMG (norm)", fontsize=10, color="#475569")
    ax_emg.tick_params(axis='y', labelcolor="#475569")
    ax_emg.set_ylim(-0.05, 1.25)

    lines_peth, labels_peth = ax_peth.get_legend_handles_labels()
    lines_emg, labels_emg = ax_emg.get_legend_handles_labels()
    ax_peth.legend(lines_peth + lines_emg, labels_peth + labels_emg, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

def build_gallery_html(index_cards, n_breaths, ti_min, ti_max, ti_mean, out_html, title_suffix=""):
    """
    Builds the publication-grade visual gallery HTML matching LP21 gallery_all_37_neurons.html,
    enhanced with region/archetype filter buttons and real-time search!
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>e097 Breath-Onset Aligned Rasters (Ordered by Inspiration Duration) {title_suffix}</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --card-border: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-coral: #f43f5e;
            --accent-green: #22c55e;
            --badge-bg: #0369a1;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .header-container {{
            text-align: center;
            max-width: 1400px;
            margin: 0 auto 24px auto;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--card-border);
        }}
        h1 {{
            color: var(--accent-blue);
            font-size: 26px;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        p.subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
            margin: 0;
        }}
        .stats-badge-container {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 16px;
        }}
        .stat-pill {{
            background: #1e293b;
            border: 1px solid #334155;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            color: #cbd5e1;
        }}
        .stat-pill b {{
            color: var(--accent-blue);
        }}
        .filter-container {{
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
            margin: 20px auto;
            max-width: 1400px;
        }}
        .filter-btn {{
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: #38bdf8;
            color: #0b0f19;
            border-color: #38bdf8;
        }}
        .search-box {{
            background: #1e293b;
            color: #fff;
            border: 1px solid #334155;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            width: 220px;
            outline: none;
        }}
        .search-box:focus {{
            border-color: #38bdf8;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
            gap: 24px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid var(--card-border);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            border-color: #3b82f6;
        }}
        .card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: #000;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .unit-title {{
            font-weight: 700;
            font-size: 17px;
            color: #ffffff;
        }}
        .badge {{
            background: var(--badge-bg);
            color: #e0f2fe;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}
        .badge-recruited {{
            background: #e11d48;
            color: #fff;
        }}
        .badge-archetype {{
            background: #10b981;
            color: #fff;
        }}
    </style>
</head>
<body>
    <div class="header-container">
        <h1>Single-Neuron Breath-Onset Rasters (Sorted by Inspiration Duration) {title_suffix}</h1>
        <p class="subtitle">
            True Real-Time Spike Rasters (No Phase Warping) aligned to Breath Onset (I-on = 0 ms).<br>
            Trials are ordered in ascending order by Inspiration Duration (<i>T<sub>I</sub></i>) to reveal burst duration modulation & phase-locked termination.<br>
            Background: Warm Red = Inspiration (&phi; &in; [0, &pi;]) | Cool Blue = Expiration (&phi; &in; [-&pi;, 0]) | Red Curve = Inspiration Offset (<i>I</i><sub>off</sub>).
        </p>
        <div class="stats-badge-container">
            <div class="stat-pill"><b>{len(index_cards)}</b> Clean Single Units (SUA)</div>
            <div class="stat-pill"><b>{n_breaths}</b> Synchronized Breaths</div>
            <div class="stat-pill">Inspiration Duration (<i>T<sub>I</sub></i>): <b>{ti_min:.0f} ms → {ti_max:.0f} ms</b> (Mean: <b>{ti_mean:.1f} ms</b>)</div>
            <div class="stat-pill">Alignment: <b>Breath Onset (0 ms)</b></div>
        </div>
    </div>

    <div class="filter-container">
        <button class="filter-btn active" onclick="filterCards('all', this)">All Units ({len(index_cards)})</button>
        <button class="filter-btn" onclick="filterCards('recruited_rostral', this)">⭐ Recruited Rostral (BötC)</button>
        <button class="filter-btn" onclick="filterCards('baseline_archetypes', this)">⭐ Tracked Archetypes</button>
        <button class="filter-btn" onclick="filterCards('recruited_caudal', this)">⭐ Recruited Caudal (cVRG)</button>
        <button class="filter-btn" onclick="filterCards('botc', this)">BötC (&lt;750 µm)</button>
        <button class="filter-btn" onclick="filterCards('prebotc', this)">preBötC Core (750-1650 µm)</button>
        <button class="filter-btn" onclick="filterCards('caudal', this)">rVRG / cVRG (&gt;1650 µm)</button>
        <input type="text" id="search-input" class="search-box" placeholder="🔍 Search cluster ID or depth..." onkeyup="searchCards(this.value)">
    </div>

    <div class="grid" id="cards-grid">
"""
    for card in index_cards:
        b_class = "badge"
        if card["is_recruited"]:
            b_class += " badge-recruited"
        elif card["is_archetype"]:
            b_class += " badge-archetype"

        cat = card["category"]
        html += f"""
        <div class="card" data-category="{cat}" data-cid="{card['cid']}" data-depth="{card['depth_um']}">
            <div class="card-header">
                <span class="unit-title">{card['title_label']}</span>
                <span class="{b_class}">{card['depth_um']:.0f} µm | {card['firing_rate_hz']:.1f} Hz | {card['n_spikes']} Spikes</span>
            </div>
            <a href="{card['file_name']}" target="_blank" title="Click to open full resolution image">
                <img src="{card['file_name']}" alt="Cluster {card['cid']}">
            </a>
        </div>
"""
    html += """
    </div>

    <script>
        function filterCards(cat, btn) {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cards = document.querySelectorAll('.card');
            cards.forEach(c => {
                if (cat === 'all') {
                    c.style.display = 'block';
                } else if (cat === 'recruited_rostral') {
                    c.style.display = (c.dataset.category === 'recruited_rostral') ? 'block' : 'none';
                } else if (cat === 'baseline_archetypes') {
                    c.style.display = (c.dataset.category === 'baseline_archetypes') ? 'block' : 'none';
                } else if (cat === 'recruited_caudal') {
                    c.style.display = (c.dataset.category === 'recruited_caudal') ? 'block' : 'none';
                } else if (cat === 'botc') {
                    c.style.display = (parseFloat(c.dataset.depth) < 750) ? 'block' : 'none';
                } else if (cat === 'prebotc') {
                    const d = parseFloat(c.dataset.depth);
                    c.style.display = (d >= 750 && d < 1650) ? 'block' : 'none';
                } else if (cat === 'caudal') {
                    c.style.display = (parseFloat(c.dataset.depth) >= 1650) ? 'block' : 'none';
                }
            });
        }

        function searchCards(val) {
            const query = val.toLowerCase().trim();
            const cards = document.querySelectorAll('.card');
            cards.forEach(c => {
                const text = c.textContent.toLowerCase();
                c.style.display = (text.includes(query)) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Saved Visual Gallery HTML to: {out_html}")

def main():
    print("[*] Starting e097 Inspiration-Onset Aligned Rasters & Visual Gallery Generation...")

    cons_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    alf_dir = cons_dir / "alf"
    out_dir = cons_dir / "02_single_cell_phase_rasters" / "onset_aligned_sorted_by_insp_duration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Spikes & Master Consensus Units
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    pr_df = pd.read_csv(cons_dir / "detailed_presence_ratio_comparison.csv")

    target_cids = list(pr_df["cluster_id"].values)
    if 448 not in target_cids and 448 in spk_clusters:
        target_cids.append(448)

    # 2. Load NIDQ Analog Data for Breaths
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    emg_env = emg_proc.envelope

    # 3. Setup Time Axis
    win_pre_s = 0.15 # -150 ms
    win_post_s = 0.35 # +350 ms
    dt_s = 0.002 # 2 ms bins
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0

    # Gaussian kernel for smoothing firing rate (15 bins, std=3 -> ~6 ms std)
    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    # 4. Extract Breaths across Full 226s Recording
    onsets = emg_proc.breath_onsets
    offsets = emg_proc.breath_offsets
    valid_pairs = []
    for on, off in zip(onsets, offsets):
        if on > win_pre_s and on < t_vec[-1] - win_post_s and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    ti_arr = np.array([p[2] for p in valid_pairs])
    sort_order = np.argsort(ti_arr)
    sorted_ti_ms = [ti_arr[idx] * 1000.0 for idx in sort_order]

    n_breaths = len(sort_order)
    print(f"[+] Extracted {n_breaths} valid breaths across 226s recording.")
    print(f"    T_I range: {np.min(ti_arr)*1000:.1f} ms to {np.max(ti_arr)*1000:.1f} ms (Mean: {np.mean(ti_arr)*1000:.1f} ms)")

    # 5. Precompute Phase Grid and EMG Trials ONCE!
    print("[*] Precomputing respiratory phase grid and EMG envelopes...")
    raw_phase_grid = compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s)
    phase_grid_sorted = raw_phase_grid[sort_order, :]

    emg_trials = np.zeros((n_breaths, len(time_axis_s)), dtype=np.float32)
    for i in range(n_breaths):
        on = valid_onsets[i]
        t_eval = on + time_axis_s
        emg_trials[i, :] = np.interp(t_eval, t_vec, emg_env, left=0.0, right=0.0)
    emg_trials_sorted = emg_trials[sort_order, :]

    # Categorize units for badges
    recruited_rostral_ids = {315, 13, 343, 366, 61, 269, 176, 126}
    archetype_ids = {430, 298, 448, 18, 93, 82}
    recruited_caudal_ids = {316, 354, 10, 19, 387, 74}

    archetype_names = {
        430: "Unit 2 Pacemaker",
        298: "Unit 255 Ramping Follower",
        448: "Unit 163 Pre-I Pacemaker",
        18: "Unit 18 BötC Expiratory",
        93: "Unit 70 Vanguard",
        82: "Laser Surge Unit",
        315: "Recruited BötC Interneuron",
        13: "Recruited Far Rostral",
        343: "Recruited Far Rostral",
        366: "Recruited Far Rostral",
        61: "Recruited BötC",
        269: "Recruited rVRC",
        316: "Recruited cVRG Motor",
        354: "Recruited cVRG Motor",
        10: "Recruited cVRG Motor",
        19: "Recruited cVRG Motor"
    }

    # 6. Generate Figures for All Units
    print(f"[*] Generating {len(target_cids)} inspiration-onset aligned raster and PETH figures...")
    index_cards = []

    # Sort units by probe depth
    sorted_target_cids = sorted(target_cids, key=lambda c: pr_df[pr_df["cluster_id"] == c]["depth_um"].values[0] if len(pr_df[pr_df["cluster_id"] == c]) > 0 else 1380.0)

    for idx, cid in enumerate(sorted_target_cids, 1):
        row_match = pr_df[pr_df["cluster_id"] == cid]
        if len(row_match) > 0:
            row = row_match.iloc[0]
            depth = float(row["depth_um"])
            fr_hz = float(row["firing_rate_hz"])
            is_recruited = bool(row.get("is_newly_recruited", False))
        else:
            depth = 1380.0
            fr_hz = 18.8
            is_recruited = False

        u_spks = spk_times[spk_clusters == cid]
        out_png = out_dir / f"unit_{cid:03d}_depth_{int(depth)}um_raster_peth.png"

        label = archetype_names.get(cid, None)
        title_label = f"Cluster #{cid}"
        if label:
            title_label += f" [{label}]"

        # Determine category
        if cid in recruited_rostral_ids or (is_recruited and depth < 750.0):
            cat = "recruited_rostral"
        elif cid in archetype_ids:
            cat = "baseline_archetypes"
        elif cid in recruited_caudal_ids or (is_recruited and depth >= 1650.0):
            cat = "recruited_caudal"
        elif depth < 750.0:
            cat = "botc"
        elif depth < 1650.0:
            cat = "prebotc"
        else:
            cat = "caudal"

        plot_single_neuron_raster_peth(
            cid=cid, depth=depth, fr_hz=fr_hz, archetype_label=label,
            u_spks=u_spks, valid_onsets=valid_onsets, valid_offsets=valid_offsets,
            sort_order=sort_order, sorted_ti_ms=sorted_ti_ms,
            phase_grid_sorted=phase_grid_sorted, emg_trials_sorted=emg_trials_sorted,
            time_axis_s=time_axis_s, time_axis_ms=time_axis_ms, dt_s=dt_s,
            kernel=kernel, out_file=out_png
        )

        index_cards.append({
            "cid": int(cid),
            "title_label": title_label,
            "depth_um": float(depth),
            "firing_rate_hz": float(fr_hz),
            "n_spikes": len(u_spks),
            "file_name": out_png.name,
            "is_recruited": is_recruited,
            "is_archetype": (cid in archetype_ids),
            "category": cat
        })

        if idx % 25 == 0 or idx == len(sorted_target_cids):
            print(f"    Completed {idx}/{len(sorted_target_cids)} units...")

    # 7. Generate Responsive HTML Visual Gallery
    gallery_path = out_dir / "gallery_all_162_neurons.html"
    build_gallery_html(
        index_cards=index_cards,
        n_breaths=n_breaths,
        ti_min=np.min(ti_arr)*1000.0,
        ti_max=np.max(ti_arr)*1000.0,
        ti_mean=np.mean(ti_arr)*1000.0,
        out_html=gallery_path,
        title_suffix="(All 842 Breaths across 226s Recording)"
    )

    # 8. Copy key exemplar figures to brain artifact folder for embedding in chat
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\d29177e2-f1b1-4d22-9f16-357e065e0485")
    for key_cid in [430, 298, 448, 18, 315, 82]:
        match_files = list(out_dir.glob(f"unit_{key_cid:03d}_*.png"))
        if match_files:
            import shutil
            dest = artifact_dir / f"e097_breath_onset_raster_cluster_{key_cid}.png"
            shutil.copyfile(match_files[0], dest)

    print("\n" + "=" * 85)
    print(" [SUCCESS] ALL 162 ONSET-ALIGNED & T_I-SORTED NEURON RASTERS & GALLERY GENERATED!")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)

if __name__ == "__main__":
    main()
