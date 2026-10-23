#!/usr/bin/env python3
"""
Within-Epoch Ti-Sorted Breath-Aligned Spike Rasters & Interactive Galleries for e097 (226s Window).

Sorts breath trials by inspiration duration (T_I = I_off - I_on) within each physiological epoch
independently (Epoch 1, Epoch 2, Epoch 3, Epoch 4), preserving the chronological sequence of epochs
and laser stimulation trains while revealing duration-dependent firing dynamics within each epoch.

Datasets processed:
1. Motion-Corrected Consensus Dataset:
   output_e097_prelaser_through_laser3_226s_motion_corrected/consensus_master/
2. Uncorrected Baseline Consensus Dataset:
   output_e097_prelaser_through_laser3_226s/lupin/consensus_master/
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.pipeline.compat_patch
from src.analysis.epoch_spiking_statistics import (
    DEFAULT_EPOCHS,
    calculate_cluster_multi_epoch_profile
)
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.config import EMGConfig

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.0


def compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s):
    """Precomputes continuous respiratory phase phi(t) in [-pi, pi] for breath trials."""
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

        insp_m = (t_eval >= on) & (t_eval < off)
        if np.any(insp_m):
            phi[insp_m] = (t_eval[insp_m] - on) / (off - on + 1e-6) * np.pi

        exp_m = (t_eval >= off) & (t_eval < next_on)
        if np.any(exp_m):
            phi[exp_m] = -np.pi + (t_eval[exp_m] - off) / (next_on - off + 1e-6) * np.pi

        pre_m = t_eval < on
        if np.any(pre_m):
            phi[pre_m] = -np.pi + (t_eval[pre_m] - prev_off) / (on - prev_off + 1e-6) * np.pi

        nan_m = np.isnan(phi)
        phi[nan_m] = -np.pi
        phase_grid[i, :] = phi

    return phase_grid


def plot_single_neuron_within_epoch_ti_sorted_raster_peth(
    cid: int,
    depth: float,
    fr_hz: float,
    archetype_label: str,
    u_spks: np.ndarray,
    valid_onsets: np.ndarray,
    ordered_trial_indices: np.ndarray,
    ti_arr: np.ndarray,
    phase_grid_ordered: np.ndarray,
    laser_blocks: list,
    epoch_blocks: list,
    dividers: list,
    time_axis_s: np.ndarray,
    time_axis_ms: np.ndarray,
    dt_s: float,
    kernel: np.ndarray,
    mean_emg_pre: np.ndarray,
    out_file: Path,
    metrics_summary: str = ""
):
    n_breaths = len(ordered_trial_indices)
    n_bins = len(time_axis_s)
    win_pre_s = -time_axis_s[0]
    win_post_s = time_axis_s[-1]

    spike_scatter_x = []
    spike_scatter_y = []
    rate_trials_chrono = np.zeros((len(valid_onsets), n_bins), dtype=np.float32)

    for r, orig_i in enumerate(ordered_trial_indices):
        on = valid_onsets[orig_i]
        rel_spks = u_spks - on
        in_win = rel_spks[(rel_spks >= -win_pre_s) & (rel_spks <= win_post_s)]
        for s in in_win:
            spike_scatter_x.append(s * 1000.0)
            spike_scatter_y.append(r + 1)

        counts, _ = np.histogram(in_win, bins=np.append(time_axis_s, time_axis_s[-1] + dt_s))
        smooth_fr = np.convolve(counts / dt_s, kernel, mode="same")
        rate_trials_chrono[orig_i, :] = smooth_fr

    # Epoch 1 Pre vs Epoch 4 Post PETH
    ep1_indices = epoch_blocks[0]["indices"]
    mean_fr_pre = np.mean(rate_trials_chrono[ep1_indices, :], axis=0)
    std_fr_pre = np.std(rate_trials_chrono[ep1_indices, :], axis=0)

    ep4_indices = epoch_blocks[3]["indices"]
    mean_fr_post = np.mean(rate_trials_chrono[ep4_indices, :], axis=0)
    std_fr_post = np.std(rate_trials_chrono[ep4_indices, :], axis=0)

    norm_emg = (mean_emg_pre - np.min(mean_emg_pre)) / (np.ptp(mean_emg_pre) + 1e-12)

    fig = plt.figure(figsize=(12.0, 9.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.0], hspace=0.22)

    cmap_phase = LinearSegmentedColormap.from_list(
        "phase_cmap_onset",
        ["#93c5fd", "#dbeafe", "#ffffff", "#fed7aa", "#f87171"],
        N=256
    )

    # Top Panel: Within-Epoch Ti-Sorted Raster
    ax_raster = fig.add_subplot(gs[0])
    im_phase = ax_raster.imshow(
        phase_grid_ordered,
        aspect="auto",
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        extent=[time_axis_ms[0], time_axis_ms[-1], 0.5, n_breaths + 0.5],
        origin="lower",
        alpha=0.60,
        interpolation="bilinear"
    )

    # Laser Highlight Bands
    for lbl, y0, y1 in laser_blocks:
        ax_raster.fill_between(
            [time_axis_ms[0], time_axis_ms[-1]],
            [y0, y0],
            [y1, y1],
            color='#ff922b', alpha=0.40, zorder=2
        )
        ax_raster.text(
            time_axis_ms[-1] - 8, (y0 + y1) / 2.0,
            lbl,
            color="#d9480f", fontsize=8.5, fontweight="bold",
            va="center", ha="right", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4e6", alpha=0.92, edgecolor="#ff922b")
        )

    # Spike occurrences
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

    # Inspiration Offset Curve (Red dots)
    ordered_ti_ms = ti_arr[ordered_trial_indices] * 1000.0
    ax_raster.plot(
        ordered_ti_ms,
        np.arange(1, n_breaths + 1),
        color="#b91c1c",
        linestyle="",
        marker=".",
        markersize=2.0,
        alpha=0.75,
        zorder=4,
        label=r"Inspiration Offset ($I_{\mathrm{off}}$)"
    )

    # Divider lines between epochs
    for div in dividers:
        ax_raster.axhline(div, color="#64748b", linestyle=":", linewidth=0.8, alpha=0.6, zorder=2)

    ax_raster.axvline(0, color="#1e293b", linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)
    ax_raster.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax_raster.set_ylim(0.5, n_breaths + 0.5)
    ax_raster.set_ylabel(f"Breath Trials (Sorted by $T_I$ within Epochs: 1 → {n_breaths})", fontsize=10.5, fontweight="bold")

    # Right Y-axis for Epoch Labels & Ti Ranges
    ax_epoch = ax_raster.twinx()
    ax_epoch.set_ylim(0.5, n_breaths + 0.5)
    y_ticks = [eb["center_y"] for eb in epoch_blocks]
    y_labels = [eb["label"] for eb in epoch_blocks]
    ax_epoch.set_yticks(y_ticks)
    ax_epoch.set_yticklabels(y_labels, fontsize=8.5, fontweight="semibold", color="#334155")
    ax_epoch.tick_params(right=True, length=4, color="#64748b")

    title_str = f"Cluster #{cid} ({depth:.0f} µm | FR: {fr_hz:.1f} Hz)"
    if archetype_label:
        title_str += f" — {archetype_label}"
    title_str += "\n[Trials Sorted by Inspiration Duration ($T_I$) Within Each Epoch Independently | Red Dots = $I_{\mathrm{off}}$]"
    if metrics_summary:
        title_str += f"\n[{metrics_summary}]"
    ax_raster.set_title(title_str, fontsize=10.0, fontweight="bold")
    ax_raster.grid(False)

    # Inset Colorbar in upper-left corner
    ax_cbar = ax_raster.inset_axes([0.018, 0.89, 0.14, 0.035])
    cbar = plt.colorbar(im_phase, cax=ax_cbar, orientation="horizontal")
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$0$", r"$+\pi$"], fontsize=7)
    ax_cbar.set_title(r"Phase $\phi$", fontsize=7.5, pad=1.5)

    ax_raster.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.90, fontsize=8)

    # Bottom Panel: Dual-Condition PETH
    ax_peth = fig.add_subplot(gs[1], sharex=ax_raster)
    ti_mean_pre = np.mean(ti_arr[ep1_indices]) * 1000.0
    ti_mean_post = np.mean(ti_arr[ep4_indices]) * 1000.0

    p1 = ax_peth.plot(time_axis_ms, mean_fr_pre, color="#0284c7", linewidth=2.2, label=f"Pre-Laser Eupnea (Ep 1, mean $T_I$={ti_mean_pre:.1f} ms)")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr_pre - std_fr_pre),
        mean_fr_pre + std_fr_pre,
        color="#0284c7", alpha=0.18
    )

    p2 = ax_peth.plot(time_axis_ms, mean_fr_post, color="#e11d48", linewidth=2.2, linestyle="-", label=f"Post-Laser Recovery (Ep 4, mean $T_I$={ti_mean_post:.1f} ms)")
    ax_peth.fill_between(
        time_axis_ms,
        np.maximum(0, mean_fr_post - std_fr_post),
        mean_fr_post + std_fr_post,
        color="#e11d48", alpha=0.18
    )

    ax_emg = ax_peth.twinx()
    p3 = ax_emg.plot(time_axis_ms, norm_emg, color="#64748b", linestyle=":", linewidth=1.8, label="Diaphragm EMG Envelope (Norm)")
    ax_emg.set_ylabel("Norm EMG", fontsize=9, color="#64748b")
    ax_emg.set_ylim(-0.05, 1.25)

    ax_peth.axvline(0, color="#1e293b", linestyle="--", linewidth=1.4, alpha=0.85)
    ax_peth.axvline(ti_mean_pre, color="#0284c7", linestyle=":", linewidth=1.4, alpha=0.85)
    ax_peth.axvline(ti_mean_post, color="#e11d48", linestyle=":", linewidth=1.4, alpha=0.85)

    ax_peth.set_xlabel("Time from Diaphragm EMG Inspiration Onset (ms)", fontsize=10, fontweight="bold")
    ax_peth.set_ylabel("Firing Rate (Hz)", fontsize=10, fontweight="bold")
    ax_peth.grid(True, linestyle="--", alpha=0.4)

    lines = p1 + p2 + p3
    labels = [l.get_label() for l in lines]
    ax_peth.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", framealpha=0.85, fontsize=8)

    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_gallery_html(index_cards: list, out_html: Path, total_trials: int, dataset_title: str):
    """Generates a responsive visual gallery HTML."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_title} — Within-Epoch Ti-Sorted Breath-Aligned Rasters</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-sub: #475569;
            --border-color: #e2e8f0;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --badge-qc: #059669;
            --badge-warn: #d97706;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px 32px;
        }}
        header {{
            background: var(--card-bg);
            padding: 24px 32px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            font-weight: 700;
            color: #1e293b;
        }}
        .subtitle {{
            color: var(--text-sub);
            font-size: 14px;
            margin: 0 0 16px 0;
            line-height: 1.5;
        }}
        .controls {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-btn {{
            background: #f1f5f9;
            border: 1px solid var(--border-color);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .filter-btn:hover {{
            background: #e2e8f0;
        }}
        .filter-btn.active {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}
        .search-box {{
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-size: 13px;
            min-width: 220px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
            gap: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }}
        .card-header {{
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            background: #fafafa;
        }}
        .unit-title {{
            font-weight: 700;
            font-size: 14px;
            color: #1e293b;
        }}
        .badge {{
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 9999px;
            color: white;
        }}
        .badge-qc {{ background: var(--badge-qc); }}
        .badge-warn {{ background: var(--badge-warn); }}
        .card img {{
            width: 100%;
            height: auto;
            display: block;
            background: #fff;
        }}
        .metrics-row {{
            padding: 8px 16px;
            font-size: 12px;
            color: var(--text-sub);
            background: #fff;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <header>
        <h1>{dataset_title} — Within-Epoch $T_I$-Sorted Breath-Aligned Spike Rasters</h1>
        <p class="subtitle">
            Breath trials are <b>sorted by inspiration duration ($T_I$) independently within each physiological epoch</b> (Epoch 1, Epoch 2, Epoch 3, Epoch 4).
            Preserves epoch chronological sequence and laser train demarcations while revealing duration-dependent firing dynamics and phase stretch.<br>
            Total Breaths: <b>{total_trials}</b> | Red Dots: <b>Inspiration Offset ($I_\\mathrm{{off}}$)</b> | Orange Bands: <b>Laser Trains 1, 2, 3</b>.
        </p>
        <div class="controls">
            <button class="filter-btn active" onclick="filterCards('all', this)">All Units ({len(index_cards)})</button>
            <button class="filter-btn" onclick="filterCards('qc_passed', this)">QC Passed</button>
            <button class="filter-btn" onclick="filterCards('botc', this)">Bötzinger (&lt;750 µm)</button>
            <button class="filter-btn" onclick="filterCards('prebotc', this)">pre-Bötzinger (750-1650 µm)</button>
            <button class="filter-btn" onclick="filterCards('caudal', this)">Caudal (&ge;1650 µm)</button>
            <input type="text" class="search-box" placeholder="Search Cluster # or Archetype..." onkeyup="searchCards(this.value)">
        </div>
    </header>

    <div class="grid" id="cardGrid">
"""
    for card in index_cards:
        b_class = "badge badge-qc" if card["passed_qc"] else "badge badge-warn"
        cat = card["category"]
        m_summary = card.get("metrics_text", "")
        html += f"""
        <div class="card" data-category="{cat}" data-cid="{card['cid']}" data-depth="{card['depth_um']}">
            <div class="card-header">
                <span class="unit-title">{card['title_label']}</span>
                <span class="{b_class}">{card['depth_um']:.0f} µm | {card['firing_rate_hz']:.1f} Hz</span>
            </div>
            <a href="{card['file_name']}" target="_blank" title="Click to view high-resolution image">
                <img src="{card['file_name']}" alt="Cluster {card['cid']}" loading="lazy">
            </a>
            <div class="metrics-row">
                <span>{m_summary}</span>
            </div>
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
                } else if (cat === 'qc_passed') {
                    c.style.display = (c.dataset.category === 'qc_passed') ? 'block' : 'none';
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
    print(f"[+] Saved HTML Visual Gallery to: {out_html}")


def process_dataset(
    dataset_name: str,
    cons_dir: Path,
    valid_onsets: np.ndarray,
    ordered_trial_indices: np.ndarray,
    ti_arr: np.ndarray,
    phase_grid_ordered: np.ndarray,
    laser_blocks: list,
    epoch_blocks: list,
    dividers: list,
    time_axis_s: np.ndarray,
    time_axis_ms: np.ndarray,
    dt_s: float,
    kernel: np.ndarray,
    mean_emg_pre: np.ndarray
):
    print(f"\n{'='*75}\n[*] Processing Dataset: {dataset_name}\n  Directory: {cons_dir}\n{'='*75}")
    alf_dir = cons_dir / "alf"
    out_raster_dir = cons_dir / "02_single_cell_phase_rasters" / "ti_sorted_within_epochs"
    out_raster_dir.mkdir(parents=True, exist_ok=True)

    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")

    # Load multi-epoch spiking statistics if available
    stats_csv = cons_dir / "multi_epoch_spiking_statistics.csv"
    stats_lookup = {}
    if stats_csv.exists():
        df_stats = pd.read_csv(stats_csv)
        for cid, grp in df_stats.groupby("cluster_id"):
            ep1 = grp[grp["epoch_id"] == 1].iloc[0] if np.any(grp["epoch_id"] == 1) else None
            ep4 = grp[grp["epoch_id"] == 4].iloc[0] if np.any(grp["epoch_id"] == 4) else None
            archetype = ep1["archetype"] if ep1 is not None else "Unclassified"
            depth = grp["depth_um"].iloc[0]
            passed_qc = grp["passed_qc"].iloc[0]
            stats_lookup[cid] = {
                "archetype": archetype,
                "depth_um": depth,
                "passed_qc": passed_qc,
                "ep1": ep1,
                "ep4": ep4
            }

    qc_file = cons_dir / "master_consensus_qc_summary.csv"
    if qc_file.exists():
        qc_df = pd.read_csv(qc_file)
    else:
        qc_df = pd.read_csv(cons_dir / "detailed_presence_ratio_comparison.csv")

    cids_all = sorted(list(np.unique(spk_clusters)))
    print(f"[*] Found {len(cids_all)} clusters in dataset.")

    index_cards = []
    t0_start = time.time()

    for idx_num, cid in enumerate(cids_all, 1):
        u_spks = spk_times[spk_clusters == cid]
        if len(u_spks) < 5:
            continue

        meta = stats_lookup.get(cid, None)
        if meta is not None:
            depth = float(meta["depth_um"])
            archetype = meta["archetype"]
            passed_qc = bool(meta["passed_qc"])
            fr = float(meta["ep1"]["mean_fr_hz"]) if meta["ep1"] is not None else len(u_spks) / 226.24
            ep1 = meta["ep1"]
            ep4 = meta["ep4"]
            metrics_text = (
                f"Ep 1 FR: {ep1['mean_fr_hz']:.1f} Hz (R_rel={ep1['schreiber_reliability_10ms']:.2f}) | "
                f"Ep 4 FR: {ep4['mean_fr_hz']:.1f} Hz (R_rel={ep4['schreiber_reliability_10ms']:.2f}) | "
                f"ΔFR={ep1['rate_contrast_1_4']:+.2f} | r_sig={ep1['signal_correlation_1_4']:.2f}"
                if (ep1 is not None and ep4 is not None) else ""
            )
        else:
            depth_series = qc_df.loc[qc_df["cluster_id"] == cid, "depth_um"]
            depth = float(depth_series.iloc[0]) if not depth_series.empty else 0.0
            qc_series = qc_df.loc[qc_df["cluster_id"] == cid, "passed_qc"]
            passed_qc = bool(qc_series.iloc[0]) if not qc_series.empty else False
            fr = len(u_spks) / 226.24
            archetype = "Unclassified"
            metrics_text = f"Mean FR: {fr:.1f} Hz"

        category = "qc_passed" if passed_qc else "qc_failed"
        out_png = out_raster_dir / f"ti_sorted_raster_cluster_{cid:03d}.png"

        plot_single_neuron_within_epoch_ti_sorted_raster_peth(
            cid=cid,
            depth=depth,
            fr_hz=fr,
            archetype_label=archetype,
            u_spks=u_spks,
            valid_onsets=valid_onsets,
            ordered_trial_indices=ordered_trial_indices,
            ti_arr=ti_arr,
            phase_grid_ordered=phase_grid_ordered,
            laser_blocks=laser_blocks,
            epoch_blocks=epoch_blocks,
            dividers=dividers,
            time_axis_s=time_axis_s,
            time_axis_ms=time_axis_ms,
            dt_s=dt_s,
            kernel=kernel,
            mean_emg_pre=mean_emg_pre,
            out_file=out_png,
            metrics_summary=metrics_text
        )

        if idx_num % 15 == 0 or idx_num == len(cids_all):
            el = time.time() - t0_start
            print(f"  [{idx_num}/{len(cids_all)}] Rendered Cluster #{cid} ({el:.1f}s elapsed)")

        index_cards.append({
            "cid": cid,
            "depth_um": depth,
            "firing_rate_hz": fr,
            "passed_qc": passed_qc,
            "category": category,
            "title_label": f"Cluster #{cid} — {archetype}",
            "file_name": out_png.name,
            "metrics_text": metrics_text
        })

    gallery_html_file = out_raster_dir / "gallery_within_epoch_ti_sorted_rasters.html"
    build_gallery_html(index_cards, gallery_html_file, len(ordered_trial_indices), dataset_title=dataset_name)


def main():
    print("=" * 80)
    print("  WITHIN-EPOCH Ti-SORTED RASTERS & GALLERIES PIPELINE (e097)")
    print("=" * 80)

    # 1. Load NIDAQ Analog Data
    nidq_file = Path(r"Z:\ramirezlab\budde\npx\animal_files\e097_g2\e097_g2_t0.nidq.bin")
    fs = 10000.0
    n_ch = 9
    start_t = 796.827
    dur = 226.243
    start_samp = int(start_t * fs)
    n_samp = int(dur * fs)

    print(f"[*] Slicing NIDAQ raw analog file: {nidq_file}")
    raw = np.memmap(nidq_file, dtype="int16", mode="r")
    slice_data = raw[start_samp * n_ch : (start_samp + n_samp) * n_ch].reshape(-1, n_ch)
    laser_v = slice_data[:, 6].astype(np.float32) * (10.0 / 32768.0)
    emg_v = slice_data[:, 0].astype(np.float32) * (10.0 / 32768.0)
    t_vec = np.arange(len(emg_v)) / fs

    emg_cfg = EMGConfig(fs=fs, highpass_cutoff=300.0, lowpass_cutoff=3000.0, smoothing_window_ms=50.0, peak_threshold_sd=2.0)
    emg_proc = process_emg(emg_v, t_vec, emg_cfg)
    emg_env = emg_proc.envelope

    win_pre_s = 0.15
    win_post_s = 0.35
    dt_s = 0.002
    time_axis_s = np.arange(-win_pre_s, win_post_s + dt_s, dt_s)
    time_axis_ms = time_axis_s * 1000.0

    kernel = signal.windows.gaussian(15, std=3)
    kernel /= np.sum(kernel)

    # 2. Extract Valid Breaths
    valid_pairs = []
    for on, off in zip(emg_proc.breath_onsets, emg_proc.breath_offsets):
        if on > win_pre_s and on < t_vec[-1] - win_post_s and off > on:
            valid_pairs.append((on, off, off - on))

    valid_onsets = np.array([p[0] for p in valid_pairs])
    valid_offsets = np.array([p[1] for p in valid_pairs])
    ti_arr = np.array([p[2] for p in valid_pairs])
    n_breaths = len(valid_onsets)
    print(f"[+] Extracted {n_breaths} valid breaths (0s to 226s).")

    # 3. Partition into 4 Epochs and Laser Intervals
    idx_ep1 = np.where((valid_onsets >= 0.0) & (valid_onsets < 60.0))[0]
    idx_las1 = np.where((valid_onsets >= 60.0) & (valid_onsets < 62.0))[0]
    idx_ep2 = np.where((valid_onsets >= 62.0) & (valid_onsets < 119.7))[0]
    idx_las2 = np.where((valid_onsets >= 119.7) & (valid_onsets < 122.0))[0]
    idx_ep3 = np.where((valid_onsets >= 122.0) & (valid_onsets < 164.3))[0]
    idx_las3 = np.where((valid_onsets >= 164.3) & (valid_onsets < 166.5))[0]
    idx_ep4 = np.where((valid_onsets >= 166.5) & (valid_onsets <= 226.25))[0]

    # Sort each epoch independently by Ti ascending
    sorted_ep1 = idx_ep1[np.argsort(ti_arr[idx_ep1])]
    sorted_ep2 = idx_ep2[np.argsort(ti_arr[idx_ep2])]
    sorted_ep3 = idx_ep3[np.argsort(ti_arr[idx_ep3])]
    sorted_ep4 = idx_ep4[np.argsort(ti_arr[idx_ep4])]

    ordered_trial_indices = np.concatenate([
        sorted_ep1,
        idx_las1,
        sorted_ep2,
        idx_las2,
        sorted_ep3,
        idx_las3,
        sorted_ep4
    ])

    print(f"[+] Ordered trials length: {len(ordered_trial_indices)}")
    print(f"    - Epoch 1: {len(sorted_ep1)} trials, Ti: {np.min(ti_arr[sorted_ep1])*1000:.1f} - {np.max(ti_arr[sorted_ep1])*1000:.1f} ms")
    print(f"    - Laser 1: {len(idx_las1)} trials")
    print(f"    - Epoch 2: {len(sorted_ep2)} trials, Ti: {np.min(ti_arr[sorted_ep2])*1000:.1f} - {np.max(ti_arr[sorted_ep2])*1000:.1f} ms")
    print(f"    - Laser 2: {len(idx_las2)} trials")
    print(f"    - Epoch 3: {len(sorted_ep3)} trials, Ti: {np.min(ti_arr[sorted_ep3])*1000:.1f} - {np.max(ti_arr[sorted_ep3])*1000:.1f} ms")
    print(f"    - Laser 3: {len(idx_las3)} trials")
    print(f"    - Epoch 4: {len(sorted_ep4)} trials, Ti: {np.min(ti_arr[sorted_ep4])*1000:.1f} - {np.max(ti_arr[sorted_ep4])*1000:.1f} ms")

    # 4. Laser Blocks & Epoch Blocks Configuration
    y_ep1_start = 0.5
    y_ep1_end = len(sorted_ep1) + 0.5

    y_las1_start = y_ep1_end
    y_las1_end = y_las1_start + len(idx_las1)

    y_ep2_start = y_las1_end
    y_ep2_end = y_ep2_start + len(sorted_ep2)

    y_las2_start = y_ep2_end
    y_las2_end = y_las2_start + len(idx_las2)

    y_ep3_start = y_las2_end
    y_ep3_end = y_ep3_start + len(sorted_ep3)

    y_las3_start = y_ep3_end
    y_las3_end = y_las3_start + len(idx_las3)

    y_ep4_start = y_las3_end
    y_ep4_end = y_ep4_start + len(sorted_ep4)

    laser_blocks = [
        ("⚡ Laser Train 1 (60s)", y_las1_start, y_las1_end),
        ("⚡ Laser Train 2 (120s)", y_las2_start, y_las2_end),
        ("⚡ Laser Train 3 (164s)", y_las3_start, y_las3_end)
    ]

    dividers = [
        y_ep1_end, y_las1_end,
        y_ep2_end, y_las2_end,
        y_ep3_end, y_las3_end
    ]

    epoch_blocks = [
        {
            "id": 1,
            "indices": sorted_ep1,
            "center_y": (y_ep1_start + y_ep1_end) / 2.0,
            "label": f"Epoch 1: Pre-Laser (n={len(sorted_ep1)})\n$T_I$: {np.min(ti_arr[sorted_ep1])*1000:.0f}–{np.max(ti_arr[sorted_ep1])*1000:.0f} ms"
        },
        {
            "id": 2,
            "indices": sorted_ep2,
            "center_y": (y_ep2_start + y_ep2_end) / 2.0,
            "label": f"Epoch 2: Inter-Train 1 (n={len(sorted_ep2)})\n$T_I$: {np.min(ti_arr[sorted_ep2])*1000:.0f}–{np.max(ti_arr[sorted_ep2])*1000:.0f} ms"
        },
        {
            "id": 3,
            "indices": sorted_ep3,
            "center_y": (y_ep3_start + y_ep3_end) / 2.0,
            "label": f"Epoch 3: Inter-Train 2 (n={len(sorted_ep3)})\n$T_I$: {np.min(ti_arr[sorted_ep3])*1000:.0f}–{np.max(ti_arr[sorted_ep3])*1000:.0f} ms"
        },
        {
            "id": 4,
            "indices": sorted_ep4,
            "center_y": (y_ep4_start + y_ep4_end) / 2.0,
            "label": f"Epoch 4: Recovery (n={len(sorted_ep4)})\n$T_I$: {np.min(ti_arr[sorted_ep4])*1000:.0f}–{np.max(ti_arr[sorted_ep4])*1000:.0f} ms"
        }
    ]

    # 5. Precompute Phase Grid and EMG
    print("[*] Precomputing continuous respiratory phase grid...")
    phase_grid_chrono = compute_continuous_respiratory_phase_grid(valid_onsets, valid_offsets, time_axis_s)
    phase_grid_ordered = phase_grid_chrono[ordered_trial_indices, :]

    mean_emg_pre = np.zeros(len(time_axis_s), dtype=np.float32)
    for orig_i in sorted_ep1:
        mean_emg_pre += np.interp(valid_onsets[orig_i] + time_axis_s, t_vec, emg_env, left=0.0, right=0.0)
    mean_emg_pre /= len(sorted_ep1)

    # 6. Process Dataset A: Motion-Corrected Consensus
    mc_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected" / "consensus_master"
    if mc_dir.exists():
        process_dataset(
            dataset_name="Motion-Corrected Consensus",
            cons_dir=mc_dir,
            valid_onsets=valid_onsets,
            ordered_trial_indices=ordered_trial_indices,
            ti_arr=ti_arr,
            phase_grid_ordered=phase_grid_ordered,
            laser_blocks=laser_blocks,
            epoch_blocks=epoch_blocks,
            dividers=dividers,
            time_axis_s=time_axis_s,
            time_axis_ms=time_axis_ms,
            dt_s=dt_s,
            kernel=kernel,
            mean_emg_pre=mean_emg_pre
        )

    # 7. Process Dataset B: Uncorrected Baseline Consensus (Benchmark)
    uncorrected_dir = REPO_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    if uncorrected_dir.exists():
        process_dataset(
            dataset_name="Uncorrected Baseline (Lupin Consensus)",
            cons_dir=uncorrected_dir,
            valid_onsets=valid_onsets,
            ordered_trial_indices=ordered_trial_indices,
            ti_arr=ti_arr,
            phase_grid_ordered=phase_grid_ordered,
            laser_blocks=laser_blocks,
            epoch_blocks=epoch_blocks,
            dividers=dividers,
            time_axis_s=time_axis_s,
            time_axis_ms=time_axis_ms,
            dt_s=dt_s,
            kernel=kernel,
            mean_emg_pre=mean_emg_pre
        )

    print("\n" + "=" * 80)
    print("  WITHIN-EPOCH Ti-SORTED RASTERS & GALLERIES COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
