"""
Stage 5-6 Spike Sorting & Multi-Sorter Benchmark Inspector:
Generates publication-quality comparison dashboards and agreement matrices
across Kilosort 4, MountainSort 5, SpyKING-CIRCUS 2, Tridesclous 2, and Lupin.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger("SORTING_INSPECTOR")

def plot_multi_sorter_comparison(
    qc_dataframes: Dict[str, pd.DataFrame],
    out_path: Path,
    duration_s: float,
    agreement_matrix: Optional[pd.DataFrame] = None,
    dpi: int = 300
) -> Path:
    """
    Generates a 6-panel comprehensive benchmark dashboard across multiple sorters.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorters = list(qc_dataframes.keys())
    n_sorters = len(sorters)
    if n_sorters == 0:
        logger.warning("No QC dataframes provided for comparison plot.")
        return out_path

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(sorters)}

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Spike Sorting Benchmark & Cross-Sorter Comparison (Duration: {duration_s/60:.1f} min)", fontsize=16, fontweight='bold', y=0.98)

    # 1. Total and QC Unit Counts
    ax_yield = axes[0, 0]
    x = np.arange(n_sorters)
    w = 0.35
    total_units = [len(qc_dataframes[s]) for s in sorters]
    qc_units = [qc_dataframes[s]['passed_allen_qc'].sum() if len(qc_dataframes[s]) > 0 else 0 for s in sorters]

    ax_yield.bar(x - w/2, total_units, w, label='Total Sorter Units', color='#34495e', alpha=0.8)
    ax_yield.bar(x + w/2, qc_units, w, label='Allen QC Passed Units', color='#2ecc71', alpha=0.9)
    ax_yield.set_title("A. Unit Yield & Single-Unit QC Pass Count", fontsize=13, fontweight='bold')
    ax_yield.set_ylabel("Number of Units", fontsize=11)
    ax_yield.set_xticks(x)
    ax_yield.set_xticklabels(sorters, fontsize=10, rotation=15)
    ax_yield.legend(loc='upper right', fontsize=9)
    ax_yield.grid(axis='y', linestyle=':', alpha=0.6)
    for i in range(n_sorters):
        ax_yield.text(i - w/2, total_units[i] + max(total_units)*0.02, str(total_units[i]), ha='center', fontweight='bold', fontsize=9)
        ax_yield.text(i + w/2, qc_units[i] + max(total_units)*0.02, str(qc_units[i]), ha='center', fontweight='bold', fontsize=9, color='#27ae60')

    # 2. Firing Rate Distribution (Log scale)
    ax_fr = axes[0, 1]
    for s in sorters:
        df = qc_dataframes[s]
        if len(df) > 0:
            ax_fr.hist(
                np.log10(np.maximum(df['firing_rate_hz'], 0.01)),
                bins=20,
                alpha=0.5,
                color=color_map[s],
                label=f"{s} (mean={df['firing_rate_hz'].mean():.1f} Hz)"
            )
    ax_fr.set_title("B. Firing Rate Distributions (Log scale)", fontsize=13, fontweight='bold')
    ax_fr.set_xlabel("log10 Firing Rate (Hz)", fontsize=11)
    ax_fr.set_ylabel("Unit Count", fontsize=11)
    ax_fr.legend(loc='upper right', fontsize=9)
    ax_fr.grid(True, linestyle=':', alpha=0.6)

    # 3. ISI Violations vs Firing Rate
    ax_isi = axes[0, 2]
    for s in sorters:
        df = qc_dataframes[s]
        if len(df) > 0:
            ax_isi.scatter(
                df['firing_rate_hz'],
                df['isi_violations'],
                color=color_map[s],
                alpha=0.6,
                label=s,
                s=25
            )
    ax_isi.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='QC Cutoff (0.5)')
    ax_isi.set_title("C. Refractory Period Violations vs Firing Rate", fontsize=13, fontweight='bold')
    ax_isi.set_xlabel("Firing Rate (Hz)", fontsize=11)
    ax_isi.set_ylabel("ISI Violation Ratio", fontsize=11)
    ax_isi.set_ylim(-0.05, 1.05)
    ax_isi.legend(loc='upper right', fontsize=9)
    ax_isi.grid(True, linestyle=':', alpha=0.6)

    # 4. Estimated SNR Distribution
    ax_snr = axes[1, 0]
    for s in sorters:
        df = qc_dataframes[s]
        if len(df) > 0:
            ax_snr.hist(
                df['snr'],
                bins=20,
                alpha=0.5,
                color=color_map[s],
                label=f"{s} (mean={df['snr'].mean():.1f})"
            )
    ax_snr.set_title("D. Estimated Signal-to-Noise Ratio (SNR)", fontsize=13, fontweight='bold')
    ax_snr.set_xlabel("SNR", fontsize=11)
    ax_snr.set_ylabel("Unit Count", fontsize=11)
    ax_snr.legend(loc='upper right', fontsize=9)
    ax_snr.grid(True, linestyle=':', alpha=0.6)

    # 5. Presence Ratio Distribution
    ax_pres = axes[1, 1]
    for s in sorters:
        df = qc_dataframes[s]
        if len(df) > 0:
            ax_pres.hist(
                df['presence_ratio'],
                bins=20,
                alpha=0.5,
                color=color_map[s],
                label=s
            )
    ax_pres.axvline(0.90, color='black', linestyle='--', linewidth=1.5, label='QC Cutoff (0.90)')
    ax_pres.set_title("E. Temporal Presence Ratio", fontsize=13, fontweight='bold')
    ax_pres.set_xlabel("Presence Ratio", fontsize=11)
    ax_pres.set_ylabel("Unit Count", fontsize=11)
    ax_pres.legend(loc='upper left', fontsize=9)
    ax_pres.grid(True, linestyle=':', alpha=0.6)

    # 6. Cross-Sorter Agreement Heatmap
    ax_mat = axes[1, 2]
    if agreement_matrix is not None and not agreement_matrix.empty:
        im = ax_mat.imshow(agreement_matrix.values, cmap='YlGnBu', vmin=0.0, vmax=1.0)
        ax_mat.set_xticks(range(len(agreement_matrix.columns)))
        ax_mat.set_yticks(range(len(agreement_matrix.index)))
        ax_mat.set_xticklabels(agreement_matrix.columns, fontsize=9, rotation=20)
        ax_mat.set_yticklabels(agreement_matrix.index, fontsize=9)
        ax_mat.set_title("F. Inter-Sorter Pairwise Agreement (Jaccard)", fontsize=13, fontweight='bold')
        for r in range(len(agreement_matrix.index)):
            for c in range(len(agreement_matrix.columns)):
                val = agreement_matrix.values[r, c]
                ax_mat.text(c, r, f"{val:.2f}", ha='center', va='center', fontweight='bold', color='black' if val < 0.7 else 'white')
        fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    else:
        # Fallback: Depth Distribution of QC Units
        for s in sorters:
            df = qc_dataframes[s]
            if len(df) > 0:
                qc_passed = df[df['passed_allen_qc']]
                if len(qc_passed) > 0:
                    ax_mat.hist(qc_passed['depth_um'], bins=25, alpha=0.5, color=color_map[s], label=s)
        ax_mat.set_title("F. Depth Distribution of Allen QC Passed Units", fontsize=13, fontweight='bold')
        ax_mat.set_xlabel("Probe Depth (um)", fontsize=11)
        ax_mat.set_ylabel("QC Unit Count", fontsize=11)
        ax_mat.legend(loc='upper right', fontsize=9)
        ax_mat.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved multi-sorter benchmark dashboard to: {out_path}")
    return out_path
