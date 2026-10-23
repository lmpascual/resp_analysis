#!/usr/bin/env python3
"""
Generate Chronologically Ordered Breath-Onset Aligned Spike Rasters and PETHs
Showing Laser Trial Demarcations for All 162 Units in e097 (226s continuous recording):

Features:
1. Top Panel: Breath-Onset Aligned Spike Raster in Chronological Order (Breaths 1 to 842):
   - Aligned to Inspiration Onset (I_on = 0 ms, -150 ms to +350 ms).
   - Y-axis: Chronological Breath Number (1 to 842, from t = 0.2s to 226s).
   - Secondary Y-axis: Continuous Recording Time (seconds).
   - Clear Demarcation of Laser Trains:
     * Shaded orange horizontal bands with labels:
       ⚡ Laser Train 1 (60s, Breaths #199-#203)
       ⚡ Laser Train 2 (120s, Breaths #419-#423)
       ⚡ Laser Train 3 (164s, Breaths #588-#591)
   - Continuous phase background:
     * Warm Red = Inspiration (phi in [0, +pi])
     * Cool Blue = Expiration (phi in [-pi, 0])
   - Dark red line / points tracking breath-by-breath I_off.
   - Precision black spike ticks.

2. Bottom Panel: Dual-Condition PETH Comparison:
   - Blue curve: Pre-Laser Baseline PETH (Breaths 1-198, mean +/- 1 SD shadow).
   - Red curve: Post-Laser Recovery PETH (Breaths 591-842, mean +/- 1 SD shadow).
   - Secondary y-axis with synchronized Diaphragm EMG envelope.
   - Vertical landmark guides at I_on (0 ms) and mean I_off.

3. Interactive Visual Gallery HTML:
   - gallery_all_162_neurons_chronological.html
   - Filter chips for Recruited Rostral, Baseline Archetypes, Caudal Motor, and Anatomical Depths.
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
    Precomputes continuous respiratory phase phi(t) in [-pi, pi] for chronological breath trials.
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

        nan_m = np.isnan(phi)
        phi[nan_m] = -np.pi
        phase_grid[i, :] = phi

    return phase_grid

def plot_single_neuron_chronological_raster_peth(
    cid, depth, fr_hz, archetype_label,
    u_spks, valid_onsets, valid_offsets, ti_arr,
    phase_grid_chrono, emg_trials_chrono,
    laser_train_bands,
    time_axis_s, time_axis_ms, dt_s,
    kernel, out_file
):
    n_breaths = len(valid_onsets)
    n_bins = len(time_axis_s)
    win_pre_s = -time_axis_s[0]
    win_post_s = time_axis_s[-1]

    # Extract spikes and firing rates for each chronological breath trial
    spike_scatter_x = []
    spike_scatter_y = []
    rate_trials = np.zeros((n_breaths, n_bins), dtype=np.float32)

    for row_idx in range(n_breaths):
        on = valid_onsets[row_idx]
        rel_spks = u_spks - on
        in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
        for s in in_win:
            spike_scatter_x.append(s * 1000.0) # ms
            spike_scatter_y.append(row_idx + 1) # 1-indexed

        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
        smooth_fr = np.convolve(counts / dt_s, kernel, mode="same")
        rate_trials[row_idx, :] = smooth_fr

    # Pre-Laser baseline trials (Breaths 1-198)
    pre_indices = np.arange(0, min(198, n_breaths))
    mean_fr_pre = np.mean(rate_trials[pre_indices, :], axis=0)
    std_fr_pre = np.std(rate_trials[pre_indices, :], axis=0)

    # Post-Laser recovery trials (Breaths 591 to end)
    post_start_idx = min(590, n_breaths - 1)
    post_indices = np.arange(post_start_idx, n_breaths)
    mean_fr_post = np.mean(rate_trials[post_indices, :], axis=0)
    std_fr_post = np.std(rate_trials[post_indices, :], axis=0)

    # EMG
    mean_emg = np.mean(emg_trials_chrono[pre_indices, :], axis=0)
    emg_min, emg_max = np.min(mean_emg), np.max(mean_emg)
    norm_emg = (mean_emg - emg_min) / (emg_max - emg_min + 1e-12) if emg_max > emg_min else np.zeros_like(mean_emg)

    mean_ti_ms = np.mean(ti_arr[:198]) * 1000.0

    # Build Figure
    fig = plt.figure(figsize=(11.0, 8.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.4, 1.0], hspace=0.20)

    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_onset",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    # -------------------------------------------------------------------------
    # Top Panel: Chronological Raster with Laser Train Demarcations
    # -------------------------------------------------------------------------
    ax_raster = fig.add_subplot(gs[0])
    im_phase = ax_raster.imshow(
        phase_grid_chrono,
        aspect="auto",
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
        origin="lower",
        alpha=0.60,
        interpolation="bilinear"
    )

    # Overlay Laser Train Bands (Orange Shading)
    for tb in laser_train_bands:
        b_start = tb["start_row"]
        b_end = tb["end_row"]
        lbl = tb["label"]
        ax_raster.axhspan(b_start - 0.5, b_end + 0.5, color='#ff922b', alpha=0.35, zorder=2)
        ax_raster.text(
            time_axis_ms[-1] - 8, (b_start + b_end) / 2.0,
            f"⚡ {lbl}",
            color="#d9480f", fontsize=8.5, fontweight="bold",
            va="center", ha="right", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4e6", alpha=0.90, edgecolor="#ff922b")
        )

    # Overlay Spikes
    if len(spike_scatter_x) > 0:
        ax_raster.scatter(
            spike_scatter_x,
            spike_scatter_y,
            marker='|',
            color='black',
            s=16,
            linewidth=0.85,
            alpha=0.92,
            zorder=3,
            label="Spike Times"
        )

    # Overlay I_off for each breath
    ioff_curve_x = ti_arr * 1000.0
    ioff_curve_y = np.arange(1, n_breaths + 1)
    ax_raster.plot(
        ioff_curve_x,
        ioff_curve_y,
        color="#b91c1c",
        linestyle="",
        marker=".",
        markersize=2.0,
        alpha=0.75,
        zorder=4,
        label=r"Inspiration Offset ($I_{\mathrm{off}}$)"
    )

    # Breath Onset Guide (I_on: 0 ms)
    ax_raster.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)

    ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax_raster.set_ylim(0.5, n_breaths + 0.5)
    ax_raster.set_ylabel(f"Breath Trials (Chronological Order: 1 → {n_breaths})", fontsize=10.5, fontweight="bold")

    # Secondary Y-axis on the right showing Recording Time (s)
    ax_time = ax_raster.twinx()
    ax_time.set_ylim(valid_onsets[0], valid_onsets[-1])
    ax_time.set_ylabel("Continuous Recording Time (s)", fontsize=10, fontweight="bold", color="#475569")
    ax_time.tick_params(axis='y', labelcolor="#475569")

    title_str = f"Cluster #{cid} ({depth:.0f} µm | FR: {fr_hz:.1f} Hz)"
    if archetype_label:
        title_str += f" — {archetype_label}"
    title_str += " — Chronological Breath-Onset Aligned Raster\n"
    title_str += r"[Chronological Breaths 1 → 842 (0s → 226s) | Orange Bands = Laser Trains 1, 2, 3 | Red Dots = $I_{\mathrm{off}}$]"
    ax_raster.set_title(title_str, fontsize=11, fontweight="bold")
    ax_raster.grid(False)

    cbar = plt.colorbar(im_phase, ax=ax_raster, fraction=0.025, pad=0.09)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$", r"$+\pi$"], fontsize=7.5)

    ax_raster.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    # -------------------------------------------------------------------------
    # Bottom Panel: Dual-Condition PETH (Pre-Laser vs Post-Laser) & EMG
    # -------------------------------------------------------------------------
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)

    # Pre-Laser PETH (Blue)
    p1 = ax_peth.plot(time_axis_ms, mean_fr_pre, color="#0284c7", linewidth=2.2, label="Pre-Laser Eupnea FR (Breaths 1-198)")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr_pre - std_fr_pre),
        mean_fr_pre + std_fr_pre,
        color="#0284c7", alpha=0.18
    )

    # Post-Laser Recovery PETH (Red)
    p2 = ax_peth.plot(time_axis_ms, mean_fr_post, color="#e11d48", linewidth=2.2, linestyle="-", label="Post-Laser Recovery FR (Breaths 591-842)")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr_post - std_fr_post),
        mean_fr_post + std_fr_post,
        color="#e11d48", alpha=0.15
    )

    ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.85, label=r"Breath Onset ($I_{\mathrm{on}}$: 0 ms)")
    ax_peth.axvline(mean_ti_ms, color="#b91c1c", linestyle=":", linewidth=1.8, alpha=0.90, label=f"Mean $I_{{\\mathrm{{off}}}}$ ({mean_ti_ms:.1f} ms)")

    ax_peth.set_xlabel("Time from Breath Onset (ms)", fontsize=11, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=10.5, fontweight="bold")
    ax_peth.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis for Diaphragm EMG
    ax_emg = ax_peth.twinx()
    p_emg = ax_emg.plot(time_axis_ms, norm_emg, color="#64748b", linestyle=":", linewidth=1.8, label="Dia. EMG Envelope")
    ax_emg.set_ylabel("Dia. EMG (norm)", fontsize=9.5, color="#64748b")
    ax_emg.tick_params(axis='y', labelcolor="#64748b")
    ax_emg.set_ylim(-0.05, 1.25)

    # Combine legends
    lines_peth, labels_peth = ax_peth.get_legend_handles_labels()
    lines_emg, labels_emg = ax_emg.get_legend_handles_labels()
    ax_peth.legend(lines_peth + lines_emg, labels_peth + labels_emg, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=7.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)

def build_chronological_gallery_html(index_cards, n_breaths, out_html):
    """
    Builds the publication-grade visual gallery HTML for chronologically ordered rasters with laser markers.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>e097 Chronological Breath-Onset Aligned Rasters (with Laser Train Demarcations)</title>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --card-border: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-coral: #f43f5e;
            --accent-orange: #fb923c;
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
        .stat-pill.laser b {{
            color: var(--accent-orange);
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
            width: 240px;
            outline: none;
        }}
        .search-box:focus {{
            border-color: #38bdf8;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
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
        <h1>Chronological Breath-Onset Rasters with Laser Train Demarcations</h1>
        <p class="subtitle">
            True Real-Time Spike Rasters aligned to Breath Onset (I-on = 0 ms), ordered <b>chronologically from Breath #1 (t=0.2s) to Breath #{n_breaths} (t=226s)</b>.<br>
            Orange horizontal bands demarcate the exact trials where <b>Laser Trains 1 (60s), 2 (120s), and 3 (164s)</b> occurred.<br>
            Bottom Panel compares Pre-Laser Eupnea PETH (Blue) vs Post-Laser Recovery PETH (Red) with Diaphragm EMG Envelope overlay.
        </p>
        <div class="stats-badge-container">
            <div class="stat-pill"><b>{len(index_cards)}</b> Clean Single Units (SUA)</div>
            <div class="stat-pill"><b>{n_breaths}</b> Chronological Breaths (0s → 226s)</div>
            <div class="stat-pill laser">⚡ <b>3 Laser Photostimulation Trains</b> Demarcated</div>
            <div class="stat-pill">Ordering: <b>Chronological Timeline</b></div>
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
    print(f"[+] Saved Chronological Visual Gallery HTML to: {out_html}")

def main():
    print("[*] Starting e097 Chronological Breath-Onset Aligned Rasters with Laser Train Demarcations...")

    cons_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    alf_dir = cons_dir / "alf"
    out_dir = cons_dir / "02_single_cell_phase_rasters" / "onset_aligned_chronological"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Spikes & Master Consensus Units
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    pr_df = pd.read_csv(cons_dir / "detailed_presence_ratio_comparison.csv")

    target_cids = list(pr_df["cluster_id"].values)
    if 448 not in target_cids and 448 in spk_clusters:
        target_cids.append(448)

    # 2. Load NIDQ Analog Data for Breaths & Laser
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    emg_env = emg_proc.envelope

    # 3. Time Axis
    win_pre_s = 0.15 # -150 ms
    win_post_s = 0.35 # +350 ms
    dt_s = 0.002 # 2 ms bins
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0

    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    # 4. Extract Breaths across Full 226s Recording in Chronological Order
    onsets = emg_proc.breath_onsets
    offsets = emg_proc.breath_offsets
    valid_pairs = []
    for on, off in zip(onsets, offsets):
        if on > win_pre_s and on < t_vec[-1] - win_post_s and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    ti_arr = np.array([p[2] for p in valid_pairs])
    n_breaths = len(valid_onsets)

    print(f"[+] Extracted {n_breaths} chronological breath trials (0s to 226s).")

    # 5. Detect Laser Trains and Identify Overlapping Breath Rows
    diff = np.diff((laser_v > 0.5).astype(np.int32))
    pulse_starts = t_vec[np.where(diff == 1)[0] + 1]
    train_break_indices = np.where(np.diff(pulse_starts) > 1.0)[0]
    train_start_indices = [0] + [b + 1 for b in train_break_indices]
    train_starts = [pulse_starts[idx] for idx in train_start_indices]

    laser_train_bands = []
    for tr_i, ts in enumerate(train_starts, 1):
        te = ts + 1.955
        # Breaths overlapping with laser train window
        m = (valid_onsets + win_post_s >= ts) & (valid_onsets - win_pre_s <= te)
        ov_idx = np.where(m)[0]
        if len(ov_idx) > 0:
            start_row = ov_idx[0] + 1
            end_row = ov_idx[-1] + 1
        else:
            # Nearest breath if apnea caused gap
            nearest = np.argmin(np.abs(valid_onsets - ts))
            start_row = nearest + 1
            end_row = nearest + 2

        laser_train_bands.append({
            "train_num": tr_i,
            "start_time": ts,
            "end_time": te,
            "start_row": start_row,
            "end_row": end_row,
            "label": f"Train {tr_i} ({ts:.1f}s)"
        })
        print(f"    Train {tr_i} ({ts:.2f}s - {te:.2f}s) spans breath rows #{start_row} to #{end_row}")

    # 6. Precompute Phase Grid & EMG Envelopes
    print("[*] Precomputing chronological respiratory phase grid and EMG traces...")
    phase_grid_chrono = compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s)

    emg_trials_chrono = np.zeros((n_breaths, len(time_axis_s)), dtype=np.float32)
    for i in range(n_breaths):
        on = valid_onsets[i]
        t_eval = on + time_axis_s
        emg_trials_chrono[i, :] = np.interp(t_eval, t_vec, emg_env, left=0.0, right=0.0)

    # 7. Archetype Badges & Mapping
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

    # 8. Generate Chronological Raster & PETH Figures for All Units
    print(f"[*] Generating {len(target_cids)} chronological raster and PETH figures...")
    index_cards = []

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
        out_png = out_dir / f"unit_{cid:03d}_depth_{int(depth)}um_chronological_raster_peth.png"

        label = archetype_names.get(cid, None)
        title_label = f"Cluster #{cid}"
        if label:
            title_label += f" [{label}]"

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

        plot_single_neuron_chronological_raster_peth(
            cid=cid, depth=depth, fr_hz=fr_hz, archetype_label=label,
            u_spks=u_spks, valid_onsets=valid_onsets, valid_offsets=valid_offsets, ti_arr=ti_arr,
            phase_grid_chrono=phase_grid_chrono, emg_trials_chrono=emg_trials_chrono,
            laser_train_bands=laser_train_bands,
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

    # 9. Build Chronological Visual Gallery HTML
    gallery_path = out_dir / "gallery_all_162_neurons_chronological.html"
    build_chronological_gallery_html(
        index_cards=index_cards,
        n_breaths=n_breaths,
        out_html=gallery_path
    )

    # 10. Copy Key Exemplars to Artifacts Directory
    artifact_dir = Path(r"C:\Users\lpascu\.gemini\antigravity-ide\brain\d29177e2-f1b1-4d22-9f16-357e065e0485")
    for key_cid in [430, 298, 448, 18, 315, 82, 13, 316]:
        match_files = list(out_dir.glob(f"unit_{key_cid:03d}_*.png"))
        if match_files:
            import shutil
            dest = artifact_dir / f"e097_chronological_raster_cluster_{key_cid}.png"
            shutil.copyfile(match_files[0], dest)

    print("\n" + "=" * 85)
    print(" [SUCCESS] CHRONOLOGICAL RASTERS WITH LASER DEMARCATIONS & GALLERY GENERATED!")
    print("=" * 85)
    print(f" • Output Folder       : {out_dir.resolve()}")
    print(f" • Interactive Gallery : {gallery_path.resolve()}")
    print("=" * 85)

if __name__ == "__main__":
    main()
