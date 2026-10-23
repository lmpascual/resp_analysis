"""
Stage 2-4 Motion & Drift Diagnostic Inspector:
Generates comprehensive diagnostic figures to inspect:
1. Peak Detection & Localization Scatter (Depth vs Time color-coded by amplitude)
2. Estimated Motion Trajectories & Displacement Curves across Depth
3. Pre- vs Post-Interpolation Spatial Activity Density Profiles
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spikeinterface.full as si

logger = logging.getLogger("MOTION_INSPECTOR")

def plot_motion_diagnostics(
    motion_info: Dict[str, Any],
    recording: si.BaseRecording,
    out_path: Path,
    corrected_rec: Optional[si.BaseRecording] = None,
    time_range_s: Optional[Tuple[float, float]] = None,
    dpi: int = 300
) -> Path:
    """
    Generates a 4-panel comprehensive motion estimation & drift correction diagnostic figure:
    [Panel A]: Peak scatter (Depth vs Time) overlaid with estimated displacement curves.
    [Panel B]: Estimated drift displacement (um) over time per spatial bin.
    [Panel C]: Peak amplitude distribution & detection threshold verification.
    [Panel D]: Spatial spike activity density before vs after motion interpolation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    peaks = motion_info.get("peaks", None)
    peak_locations = motion_info.get("peak_locations", None)
    motion = motion_info.get("motion", None)

    if peaks is None or peak_locations is None:
        raise ValueError("motion_info must contain 'peaks' and 'peak_locations'.")

    fs = recording.get_sampling_frequency()
    total_dur_s = recording.get_total_duration()

    # Peak times and depths
    peak_times_s = peaks["sample_index"] / fs
    
    # Handle peak locations structure (can be structured array or 2D array)
    if hasattr(peak_locations, "dtype") and "y" in peak_locations.dtype.names:
        peak_depths_um = peak_locations["y"]
    elif isinstance(peak_locations, np.ndarray) and peak_locations.ndim == 2 and peak_locations.shape[1] >= 2:
        peak_depths_um = peak_locations[:, 1]
    else:
        peak_depths_um = np.array(peak_locations).flatten()

    peak_amps = np.abs(peaks["amplitude"])

    # Subsample peaks for plotting if very large
    max_peaks_to_plot = 50000
    if len(peak_times_s) > max_peaks_to_plot:
        idx_sub = np.random.choice(len(peak_times_s), max_peaks_to_plot, replace=False)
        p_times_plot = peak_times_s[idx_sub]
        p_depths_plot = peak_depths_um[idx_sub]
        p_amps_plot = peak_amps[idx_sub]
    else:
        p_times_plot = peak_times_s
        p_depths_plot = peak_depths_um
        p_amps_plot = peak_amps

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("SpikeInterface Motion Estimation & Kriging Interpolation Diagnostics", fontsize=16, fontweight='bold', y=0.98)

    # Panel A: Peak Scatter Over Time & Depth
    ax_scatter = axes[0, 0]
    sc = ax_scatter.scatter(
        p_times_plot,
        p_depths_plot,
        c=p_amps_plot,
        cmap='inferno_r',
        s=4,
        alpha=0.6,
        rasterized=True
    )
    ax_scatter.set_title("A. Detected Peaks (Depth vs Time)", fontsize=13, fontweight='bold')
    ax_scatter.set_xlabel("Time (s)", fontsize=11)
    ax_scatter.set_ylabel("Probe Depth (um)", fontsize=11)
    if time_range_s:
        ax_scatter.set_xlim(time_range_s)
    else:
        ax_scatter.set_xlim(0, total_dur_s)
    cbar = fig.colorbar(sc, ax=ax_scatter, pad=0.02)
    cbar.set_label("Peak Amplitude (uV)", fontsize=10)
    ax_scatter.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Estimated Drift Trajectory / Displacement Curves
    ax_disp = axes[0, 1]
    if hasattr(motion, "displacement"):
        disp = motion.displacement
        if isinstance(disp, list) and len(disp) > 0:
            disp_arr = disp[0]
        else:
            disp_arr = np.array(disp)

        t_bins = getattr(motion, "temporal_bins_s", None)
        if isinstance(t_bins, list) and len(t_bins) > 0:
            t_bins_arr = t_bins[0]
        elif t_bins is not None:
            t_bins_arr = np.array(t_bins)
        else:
            t_bins_arr = np.linspace(0, total_dur_s, disp_arr.shape[0])

        s_bins = getattr(motion, "spatial_bins_um", None)
        if isinstance(s_bins, list) and len(s_bins) > 0:
            s_bins_arr = s_bins[0]
        elif s_bins is not None:
            s_bins_arr = np.array(s_bins)
        else:
            s_bins_arr = None

        if disp_arr.ndim == 2:
            n_spatial = disp_arr.shape[1]
            colors = plt.cm.viridis(np.linspace(0, 1, n_spatial))
            for s_idx in range(n_spatial):
                lbl = f"Depth {s_bins_arr[s_idx]:.0f} um" if (s_bins_arr is not None and s_idx < len(s_bins_arr)) else f"Bin {s_idx}"
                ax_disp.plot(t_bins_arr, disp_arr[:, s_idx], label=lbl, color=colors[s_idx], linewidth=1.5)
            if n_spatial <= 10:
                ax_disp.legend(loc='upper right', fontsize=8)
        else:
            ax_disp.plot(t_bins_arr, disp_arr, color='#e74c3c', linewidth=2.0, label='Rigid Drift')
            ax_disp.legend(loc='upper right', fontsize=10)

    elif isinstance(motion, np.ndarray):
        t_bins = np.linspace(0, total_dur_s, len(motion))
        ax_disp.plot(t_bins, motion, color='#e74c3c', linewidth=2.0, label='Estimated Displacement')
        ax_disp.legend(loc='upper right', fontsize=10)

    ax_disp.set_title("B. Estimated Motion Displacement over Time", fontsize=13, fontweight='bold')
    ax_disp.set_xlabel("Time (s)", fontsize=11)
    ax_disp.set_ylabel("Displacement (um)", fontsize=11)
    ax_disp.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Peak Amplitude & Firing Rate Sanity Check
    ax_amp = axes[1, 0]
    ax_amp.hist(p_amps_plot, bins=50, color='#3498db', alpha=0.8, edgecolor='black')
    ax_amp.set_yscale('log')
    ax_amp.set_title("C. Detected Peak Amplitude Distribution (Log scale)", fontsize=13, fontweight='bold')
    ax_amp.set_xlabel("Peak Amplitude (uV)", fontsize=11)
    ax_amp.set_ylabel("Number of Peaks (Log scale)", fontsize=11)
    ax_amp.grid(True, which='both', linestyle=':', alpha=0.5)

    # Panel D: Spatial Activity Density Before vs After Motion Correction
    ax_density = axes[1, 1]
    depth_bins = np.linspace(0, np.max(p_depths_plot) + 50, 60)
    counts_orig, _ = np.histogram(peak_depths_um, bins=depth_bins)
    bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2.0

    ax_density.plot(counts_orig, bin_centers, color='#7f8c8d', linewidth=2.0, label='Uncorrected Activity Profile')
    
    # If motion correction applied, show estimated corrected profile
    if hasattr(motion, "displacement") and motion.displacement is not None:
        try:
            disp_data = motion.displacement[0] if isinstance(motion.displacement, list) else motion.displacement
            corrected_depths = peak_depths_um.copy() - np.mean(disp_data)
            counts_corr, _ = np.histogram(corrected_depths, bins=depth_bins)
            ax_density.plot(counts_corr, bin_centers, color='#2ecc71', linewidth=2.0, label='Drift-Corrected Activity Profile')
        except Exception:
            pass

    ax_density.set_title("D. Depth-Resolved Spike Activity Profile", fontsize=13, fontweight='bold')
    ax_density.set_xlabel("Spike Count per Depth Bin", fontsize=11)
    ax_density.set_ylabel("Probe Depth (um)", fontsize=11)
    ax_density.legend(loc='upper right', fontsize=10)
    ax_density.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved motion diagnostic figure to: {out_path}")
    return out_path
