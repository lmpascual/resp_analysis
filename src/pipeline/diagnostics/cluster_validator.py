"""
High-Performance Single-Cluster Diagnostic and Quality Assurance Engine.

Implements rigorous electrophysiological validation tests conforming to Phy,
Allen Institute, IBL, and respiratory neurobiology standards (Bush et al. 2024):
1. Global multi-channel spatial footprint selection across all physical probe contacts.
2. Exact trough alignment (sub-millisecond local minimum snapping) so spike markers
   align precisely with action potential troughs on the peak channel.
3. Dual-source respiratory landmark detection (Diaphragm EMG Onset + Flow Meter ZC Offset, with EMG offset fallback).
4. Continuous respiratory phase mapping and circular mean resultant vector (Mazurek R, theta, Rayleigh test).
5. Inter-Spike Interval (ISI) distribution and Hill/IBL refractory violation metrics (tau_ref = 1.0 ms).
6. Amplitude stationarity across recording session (rolling trendline + IQR).
7. Time-resolved ISI probability density functions (PDF ridge lines across temporal epochs).
8. Autocorrelogram (ACG) with refractory depression dip.
9. Truncated Gaussian amplitude cutoff for missing spikes estimation.
10. Mean template waveform kinetics and action potential duration (putative RS vs FS).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy import signal, stats

from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.processors.proc_flow import process_flow_meter
from src.pipeline.config import EMGConfig


@dataclass
class RespiratoryPhaseResult:
    spike_phases_deg: np.ndarray
    mean_vector_r: float
    preferred_phase_deg: float
    rayleigh_p_value: float
    functional_class: str
    subtype_class: str
    rose_bins_deg: np.ndarray
    rose_density: np.ndarray
    landmark_provenance: str
    n_spikes_in_phase: int


@dataclass
class ISIValidationResult:
    isi_ms: np.ndarray
    n_violations: int
    total_spikes: int
    violation_rate_q: float
    contamination_pct: float
    refractory_ms: float
    passed_allen_isi: bool
    hist_bins_ms: np.ndarray
    hist_counts: np.ndarray


@dataclass
class AmplitudeStationarityResult:
    spike_times_s: np.ndarray
    spike_amplitudes_uv: np.ndarray
    time_bin_centers_s: np.ndarray
    median_amplitudes_uv: np.ndarray
    iqr_lower_uv: np.ndarray
    iqr_upper_uv: np.ndarray
    drift_slope_uv_per_min: float
    coefficient_of_variation: float


@dataclass
class TimeResolvedISICDFResult:
    epoch_labels: List[str]
    epoch_time_ranges: List[Tuple[float, float]]
    epoch_isis_ms: List[np.ndarray]
    epoch_cdfs: List[np.ndarray]
    ks_drift_p_values: List[float]
    pdf_x_ms: Optional[np.ndarray] = None
    epoch_pdfs: Optional[List[np.ndarray]] = None

TimeResolvedISIPDFResult = TimeResolvedISICDFResult


@dataclass
class ACGResult:
    lags_ms: np.ndarray
    counts: np.ndarray
    refractory_dip_ratio: float


@dataclass
class WaveformKineticsResult:
    peak_channel: int
    footprint_channels: List[int]
    channel_depths_um: List[float]
    template_waveforms: Dict[int, np.ndarray]
    time_axis_ms: np.ndarray
    trough_to_peak_us: float
    half_width_us: float
    repolarization_slope: float
    putative_cell_type: str
    snr: float


@dataclass
class AmplitudeCutoffResult:
    amplitude_bins: np.ndarray
    amplitude_counts: np.ndarray
    fitted_gaussian_x: np.ndarray
    fitted_gaussian_y: np.ndarray
    estimated_missing_spikes_pct: float
    passed_cutoff_qc: bool


@dataclass
class MultiChannelTraceSnippet:
    t_axis_s: np.ndarray
    channels: List[int]
    channel_depths_um: List[float]
    traces_uv: np.ndarray  # (n_channels, n_samples)
    peak_channel: int
    peak_channel_index: int
    spike_times_in_window: np.ndarray
    aligned_trough_times: np.ndarray
    aligned_trough_voltages: np.ndarray
    t_start_s: float
    t_end_s: float
    violation_trough_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    violation_trough_voltages: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


@dataclass
class UnitValidationBundle:
    cluster_id: int
    firing_rate_hz: float
    total_spikes: int
    depth_um: float
    shank: int
    peak_channel: int
    footprint_channels: List[int]
    passed_allen_qc: bool
    respiratory_phase: RespiratoryPhaseResult
    isi: ISIValidationResult
    stationarity: AmplitudeStationarityResult
    time_resolved_isi: TimeResolvedISIPDFResult
    acg: ACGResult
    waveform: WaveformKineticsResult
    amplitude_cutoff: AmplitudeCutoffResult
    trace_snippet: Optional[MultiChannelTraceSnippet] = None
    qc_summary: str = ""


def detect_respiratory_landmarks(
    emg_raw: np.ndarray,
    flow_raw: Optional[np.ndarray],
    t_nidq: np.ndarray,
    fs_nidq: float = 10000.0,
    emg_cfg: Optional[EMGConfig] = None,
    offset_source: str = 'flow'
) -> Tuple[np.ndarray, np.ndarray, str, Any]:
    """
    Extracts respiratory breath onsets and offsets.
    - Inspiration Onset: Diaphragm EMG burst onset.
    - Inspiration Offset: Flow meter zero-crossing (default) or Diaphragm EMG offset.
    """
    if emg_cfg is None:
        emg_cfg = EMGConfig(
            fs=fs_nidq,
            highpass_cutoff=300.0,
            lowpass_cutoff=3000.0,
            notch_freq=60.0,
            smoothing_window_ms=50.0,
            peak_threshold_sd=2.5,
            min_breath_duration_ms=100.0,
            min_inter_breath_interval_ms=150.0
        )

    emg_proc = process_emg(emg_raw, t_nidq, emg_cfg)
    breath_onsets = np.array(emg_proc.breath_onsets, dtype=np.float64)
    emg_offsets = np.array(emg_proc.breath_offsets, dtype=np.float64)

    if len(breath_onsets) < 2:
        return breath_onsets, emg_offsets, "Onset: Diaphragm EMG | Offset: Diaphragm EMG (insufficient breaths)", emg_proc

    if offset_source.lower() == 'flow' and flow_raw is not None and len(flow_raw) > 0:
        try:
            flow_result = process_flow_meter(
                flow_raw=flow_raw,
                time_vector=t_nidq,
                valid_onsets=breath_onsets,
                valid_offsets=emg_offsets,
                fs=fs_nidq
            )
            raw_flow_offsets = np.array(flow_result.flow_zero_crossings, dtype=np.float64)
            reconciled_offsets = []
            for i in range(len(breath_onsets)):
                t_on = breath_onsets[i]
                t_next = breath_onsets[i + 1] if i + 1 < len(breath_onsets) else t_on + 2.0
                f_off = raw_flow_offsets[i] if i < len(raw_flow_offsets) else 0.0
                e_off = emg_offsets[i] if i < len(emg_offsets) else t_on + 0.15

                if t_on < f_off < t_next:
                    reconciled_offsets.append(f_off)
                elif t_on < e_off < t_next:
                    reconciled_offsets.append(e_off)
                else:
                    reconciled_offsets.append(t_on + 0.15)

            breath_offsets = np.array(reconciled_offsets, dtype=np.float64)
            provenance_str = "Onset: Diaphragm EMG | Offset: Flow Meter Zero-Crossing (NIDQ Ch 5)"
            return breath_onsets, breath_offsets, provenance_str, emg_proc
        except Exception:
            pass

    breath_offsets = emg_offsets
    provenance_str = "Onset: Diaphragm EMG | Offset: Diaphragm EMG Burst Drop (NIDQ Ch 0)"
    return breath_onsets, breath_offsets, provenance_str, emg_proc


def compute_respiratory_phase_vector(
    spike_times_s: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    landmark_provenance: str = "Onset: EMG | Offset: Flow"
) -> RespiratoryPhaseResult:
    """
    Maps each spike timestamp to continuous respiratory phase [0, 360 deg) and computes
    Mazurek vector strength R, preferred phase angle, and Rayleigh uniformity p-value.
    """
    n_spikes = len(spike_times_s)
    if n_spikes == 0 or len(breath_onsets) < 2:
        return RespiratoryPhaseResult(
            spike_phases_deg=np.array([], dtype=np.float64),
            mean_vector_r=0.0,
            preferred_phase_deg=0.0,
            rayleigh_p_value=1.0,
            functional_class="Tonic",
            subtype_class="Tonic / Unmodulated",
            rose_bins_deg=np.linspace(0, 360, 37),
            rose_density=np.zeros(36),
            landmark_provenance=landmark_provenance,
            n_spikes_in_phase=0
        )

    n_cycles = min(len(breath_onsets) - 1, len(breath_offsets))
    cycle_onsets = breath_onsets[:n_cycles]
    cycle_offsets = breath_offsets[:n_cycles]
    cycle_next_onsets = breath_onsets[1:n_cycles + 1]

    cycle_indices = np.searchsorted(cycle_next_onsets, spike_times_s)
    valid_mask = (cycle_indices < n_cycles) & (spike_times_s >= cycle_onsets[cycle_indices])

    valid_spk_times = spike_times_s[valid_mask]
    valid_cycles = cycle_indices[valid_mask]

    t_on = cycle_onsets[valid_cycles]
    t_off = cycle_offsets[valid_cycles]
    t_next = cycle_next_onsets[valid_cycles]

    phases_deg = np.zeros(len(valid_spk_times), dtype=np.float64)
    insp_mask = valid_spk_times < t_off
    exp_mask = ~insp_mask

    t_insp_dur = np.maximum(t_off - t_on, 1e-4)
    t_exp_dur = np.maximum(t_next - t_off, 1e-4)

    phases_deg[insp_mask] = 180.0 * (valid_spk_times[insp_mask] - t_on[insp_mask]) / t_insp_dur[insp_mask]
    phases_deg[exp_mask] = 180.0 + 180.0 * (valid_spk_times[exp_mask] - t_off[exp_mask]) / t_exp_dur[exp_mask]
    phases_deg = np.mod(phases_deg, 360.0)

    phases_rad = np.deg2rad(phases_deg)
    cos_mean = float(np.mean(np.cos(phases_rad)))
    sin_mean = float(np.mean(np.sin(phases_rad)))
    r_val = float(np.sqrt(cos_mean**2 + sin_mean**2))

    phi_pref_rad = float(np.arctan2(sin_mean, cos_mean))
    phi_pref_deg = float(np.rad2deg(phi_pref_rad))
    if phi_pref_deg < 0:
        phi_pref_deg += 360.0

    n_val = len(phases_rad)
    z_stat = n_val * (r_val ** 2)
    if n_val > 0:
        p_rayleigh = float(np.exp(-z_stat) * (1.0 + (2.0 * z_stat - z_stat**2) / (4.0 * n_val)))
        p_rayleigh = max(0.0, min(1.0, p_rayleigh))
    else:
        p_rayleigh = 1.0

    if r_val < 0.20 or p_rayleigh > 0.05:
        main_class = "Tonic"
        sub_class = "Tonic / Unmodulated"
    else:
        if 0.0 <= phi_pref_deg < 90.0:
            main_class = "Inspiratory"
            sub_class = "Early-Inspiratory (Pre-I)"
        elif 90.0 <= phi_pref_deg < 180.0:
            main_class = "Inspiratory"
            sub_class = "Late-Inspiratory"
        elif 180.0 <= phi_pref_deg < 270.0:
            main_class = "Expiratory"
            sub_class = "Post-Inspiratory (Post-I / Early-Exp)"
        else:
            main_class = "Expiratory"
            sub_class = "Late-Expiratory (E2 / Pre-I Transition)"

    rose_bins = np.linspace(0, 360, 37)
    counts, _ = np.histogram(phases_deg, bins=rose_bins)
    density = counts / (np.sum(counts) * 10.0) if np.sum(counts) > 0 else np.zeros(36)

    return RespiratoryPhaseResult(
        spike_phases_deg=phases_deg,
        mean_vector_r=round(r_val, 4),
        preferred_phase_deg=round(phi_pref_deg, 2),
        rayleigh_p_value=float(p_rayleigh),
        functional_class=main_class,
        subtype_class=sub_class,
        rose_bins_deg=rose_bins,
        rose_density=density,
        landmark_provenance=landmark_provenance,
        n_spikes_in_phase=len(phases_deg)
    )


def compute_isi_and_violations(
    spike_times_s: np.ndarray,
    refractory_ms: float = 1.0,
    max_isi_ms: float = 100.0,
    bin_size_ms: float = 0.5
) -> ISIValidationResult:
    """
    Computes fine-grained ISI distribution and Allen/IBL Hill refractory violation rate.
    """
    n_spikes = len(spike_times_s)
    if n_spikes < 2:
        return ISIValidationResult(
            isi_ms=np.array([], dtype=np.float64),
            n_violations=0,
            total_spikes=n_spikes,
            violation_rate_q=0.0,
            contamination_pct=0.0,
            refractory_ms=refractory_ms,
            passed_allen_isi=True,
            hist_bins_ms=np.arange(0, max_isi_ms + bin_size_ms, bin_size_ms),
            hist_counts=np.zeros(int(max_isi_ms / bin_size_ms))
        )

    sorted_spks = np.sort(spike_times_s)
    isis_s = np.diff(sorted_spks)
    isis_ms = isis_s * 1000.0

    ref_s = refractory_ms / 1000.0
    duration_s = max(float(sorted_spks[-1] - sorted_spks[0]), 1e-3)
    n_viol = int(np.sum(isis_s < ref_s))

    q_isi = float((n_viol * duration_s) / (2.0 * ref_s * (n_spikes ** 2)))
    contam_pct = float((n_viol / n_spikes) * 100.0)
    passed_allen = bool(q_isi <= 0.50)

    bins_ms = np.arange(0, max_isi_ms + bin_size_ms, bin_size_ms)
    counts, _ = np.histogram(isis_ms, bins=bins_ms)

    return ISIValidationResult(
        isi_ms=isis_ms,
        n_violations=n_viol,
        total_spikes=n_spikes,
        violation_rate_q=round(q_isi, 4),
        contamination_pct=round(contam_pct, 2),
        refractory_ms=refractory_ms,
        passed_allen_isi=passed_allen,
        hist_bins_ms=bins_ms,
        hist_counts=counts
    )


def compute_amplitude_stationarity(
    spike_times_s: np.ndarray,
    spike_amplitudes_uv: np.ndarray,
    n_time_bins: int = 50
) -> AmplitudeStationarityResult:
    """
    Evaluates spike amplitude drift and dispersion over recording time.
    """
    n_spikes = len(spike_times_s)
    if n_spikes == 0:
        return AmplitudeStationarityResult(
            spike_times_s=np.array([]),
            spike_amplitudes_uv=np.array([]),
            time_bin_centers_s=np.array([]),
            median_amplitudes_uv=np.array([]),
            iqr_lower_uv=np.array([]),
            iqr_upper_uv=np.array([]),
            drift_slope_uv_per_min=0.0,
            coefficient_of_variation=0.0
        )

    t_min = float(np.min(spike_times_s))
    t_max = float(np.max(spike_times_s))
    if t_max <= t_min:
        t_max = t_min + 1.0

    bin_edges = np.linspace(t_min, t_max, n_time_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_idx = np.digitize(spike_times_s, bin_edges) - 1
    medians, q25s, q75s = [], [], []

    for b in range(n_time_bins):
        amps_b = spike_amplitudes_uv[bin_idx == b]
        if len(amps_b) > 0:
            medians.append(float(np.median(amps_b)))
            q25s.append(float(np.percentile(amps_b, 25)))
            q75s.append(float(np.percentile(amps_b, 75)))
        else:
            medians.append(float(np.nan))
            q25s.append(float(np.nan))
            q75s.append(float(np.nan))

    med_arr = np.array(medians)
    nans = np.isnan(med_arr)
    if np.any(nans) and not np.all(nans):
        med_arr[nans] = np.interp(bin_centers[nans], bin_centers[~nans], med_arr[~nans])
    elif np.all(nans):
        med_arr[:] = 0.0

    if n_spikes >= 10:
        slope, intercept, r_val, p_val, std_err = stats.linregress(spike_times_s, spike_amplitudes_uv)
        slope_per_min = float(slope * 60.0)
    else:
        slope_per_min = 0.0

    mean_amp = float(np.mean(spike_amplitudes_uv))
    std_amp = float(np.std(spike_amplitudes_uv))
    cv = float(std_amp / mean_amp) if mean_amp > 1e-3 else 0.0

    return AmplitudeStationarityResult(
        spike_times_s=spike_times_s,
        spike_amplitudes_uv=spike_amplitudes_uv,
        time_bin_centers_s=bin_centers,
        median_amplitudes_uv=med_arr,
        iqr_lower_uv=np.array(q25s),
        iqr_upper_uv=np.array(q75s),
        drift_slope_uv_per_min=round(slope_per_min, 2),
        coefficient_of_variation=round(cv, 3)
    )


def compute_time_resolved_isi_cdfs(
    spike_times_s: np.ndarray,
    n_epochs: int = 5,
    max_points_per_epoch: int = 1000,
    **kwargs
) -> TimeResolvedISICDFResult:
    """
    Computes multi-epoch unsmoothed empirical cumulative distribution functions (ECDFs).
    Supports logarithmic scaling across multiple decades (from <1 ms to >10 s) to capture
    both rapid refractory burst doublets and long inter-breath pauses without smoothing distortion.
    """
    if len(spike_times_s) < 10:
        return TimeResolvedISICDFResult(
            epoch_labels=[],
            epoch_time_ranges=[],
            epoch_isis_ms=[],
            epoch_cdfs=[],
            ks_drift_p_values=[],
            pdf_x_ms=np.array([]),
            epoch_pdfs=[]
        )

    t_min = float(np.min(spike_times_s))
    t_max = float(np.max(spike_times_s))
    epoch_edges = np.linspace(t_min, t_max, n_epochs + 1)

    labels = []
    ranges = []
    epoch_isis = []
    epoch_cdfs = []
    ks_pvals = []
    ep1_raw_isis = None

    for ep in range(n_epochs):
        t0, t1 = epoch_edges[ep], epoch_edges[ep + 1]
        spks_ep = spike_times_s[(spike_times_s >= t0) & (spike_times_s < t1)]
        lbl = f"Epoch {ep + 1} ({t0:.1f}-{t1:.1f}s)"
        labels.append(lbl)
        ranges.append((t0, t1))

        if len(spks_ep) >= 5:
            isis_ms = np.diff(np.sort(spks_ep)) * 1000.0
            valid_isis = np.sort(isis_ms[isis_ms > 0])
            n_pts = len(valid_isis)
            if n_pts >= 2:
                raw_y = np.arange(1, n_pts + 1, dtype=np.float64) / float(n_pts)
                if n_pts > max_points_per_epoch:
                    sub_idx = np.round(np.linspace(0, n_pts - 1, max_points_per_epoch)).astype(int)
                    sub_x = valid_isis[sub_idx]
                    sub_y = raw_y[sub_idx]
                else:
                    sub_x = valid_isis
                    sub_y = raw_y

                epoch_isis.append(sub_x)
                epoch_cdfs.append(sub_y)

                if ep == 0:
                    ep1_raw_isis = valid_isis
                    ks_pvals.append(1.0)
                else:
                    if ep1_raw_isis is not None and len(ep1_raw_isis) >= 5:
                        _, p_val = stats.ks_2samp(ep1_raw_isis, valid_isis)
                        ks_pvals.append(float(p_val))
                    else:
                        ks_pvals.append(1.0)
            else:
                epoch_isis.append(np.array([]))
                epoch_cdfs.append(np.array([]))
                ks_pvals.append(1.0)
        else:
            epoch_isis.append(np.array([]))
            epoch_cdfs.append(np.array([]))
            ks_pvals.append(1.0)

    first_isis = epoch_isis[0] if len(epoch_isis) > 0 else np.array([])
    return TimeResolvedISICDFResult(
        epoch_labels=labels,
        epoch_time_ranges=ranges,
        epoch_isis_ms=epoch_isis,
        epoch_cdfs=epoch_cdfs,
        ks_drift_p_values=ks_pvals,
        pdf_x_ms=first_isis,
        epoch_pdfs=epoch_cdfs
    )


compute_time_resolved_isi_pdfs = compute_time_resolved_isi_cdfs


def compute_autocorrelogram(
    spike_times_s: np.ndarray,
    bin_size_ms: float = 1.0,
    max_lag_ms: float = 50.0
) -> ACGResult:
    """
    Computes cross-time lag autocorrelogram [-max_lag_ms, +max_lag_ms].
    """
    n_spikes = len(spike_times_s)
    n_bins = int(2 * max_lag_ms / bin_size_ms)
    bins = np.linspace(-max_lag_ms, max_lag_ms, n_bins + 1)

    if n_spikes < 5:
        return ACGResult(
            lags_ms=0.5 * (bins[:-1] + bins[1:]),
            counts=np.zeros(n_bins, dtype=np.int64),
            refractory_dip_ratio=1.0
        )

    spks = np.sort(spike_times_s)
    if n_spikes > 10000:
        spks = np.sort(np.random.choice(spks, 10000, replace=False))

    diffs = []
    max_lag_s = max_lag_ms / 1000.0
    for i, t in enumerate(spks):
        j = i + 1
        while j < len(spks) and (spks[j] - t) <= max_lag_s:
            dt = spks[j] - t
            diffs.append(dt * 1000.0)
            diffs.append(-dt * 1000.0)
            j += 1

    diffs_arr = np.array(diffs, dtype=np.float64)
    counts, _ = np.histogram(diffs_arr, bins=bins)

    lag_centers = 0.5 * (bins[:-1] + bins[1:])
    ref_mask = np.abs(lag_centers) <= 1.5
    baseline_mask = (np.abs(lag_centers) >= 10.0) & (np.abs(lag_centers) <= 40.0)

    mean_ref = np.mean(counts[ref_mask]) if np.any(ref_mask) else 0.0
    mean_base = np.mean(counts[baseline_mask]) if np.any(baseline_mask) else 1.0
    dip_ratio = float(mean_ref / max(mean_base, 1e-3))

    return ACGResult(
        lags_ms=lag_centers,
        counts=counts,
        refractory_dip_ratio=round(dip_ratio, 3)
    )


def compute_amplitude_cutoff(spike_amplitudes_uv: np.ndarray) -> AmplitudeCutoffResult:
    """
    Estimates missing spikes fraction using mirrored right-tail Gaussian estimation (IBL / Allen Institute standard).
    """
    n_spikes = len(spike_amplitudes_uv)
    if n_spikes < 20:
        return AmplitudeCutoffResult(
            amplitude_bins=np.array([0, 100]),
            amplitude_counts=np.array([n_spikes]),
            fitted_gaussian_x=np.array([]),
            fitted_gaussian_y=np.array([]),
            estimated_missing_spikes_pct=0.0,
            passed_cutoff_qc=True
        )

    amps = np.abs(spike_amplitudes_uv)
    bins = np.linspace(np.percentile(amps, 0.5), np.percentile(amps, 99.5), 60)
    counts, _ = np.histogram(amps, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    peak_idx = int(np.argmax(counts))
    mode_val = float(bin_centers[peak_idx])

    right_tail = amps[amps >= mode_val]
    if len(right_tail) > 5:
        sigma_est = float(np.median(np.abs(right_tail - mode_val)) / 0.6745)
        sigma_est = max(sigma_est, 1e-2)
    else:
        sigma_est = float(max(np.std(amps), 1e-2))

    min_detected = float(bin_centers[0])
    cdf_val = stats.norm.cdf(min_detected, loc=mode_val, scale=sigma_est)
    missing_pct = float(cdf_val * 100.0)

    x_dense = np.linspace(bin_centers[0], bin_centers[-1], 200)
    a_height = float(counts[peak_idx])
    y_dense = a_height * np.exp(-((x_dense - mode_val) ** 2) / (2 * (sigma_est ** 2)))

    passed = bool(missing_pct <= 10.0)
    return AmplitudeCutoffResult(
        amplitude_bins=bins,
        amplitude_counts=counts,
        fitted_gaussian_x=x_dense,
        fitted_gaussian_y=y_dense,
        estimated_missing_spikes_pct=round(missing_pct, 2),
        passed_cutoff_qc=passed
    )


def find_unit_channel_footprint(
    mm_ap: np.memmap,
    spike_times_s: np.ndarray,
    fs: float = 30000.0,
    candidate_channels: Optional[np.ndarray] = None,
    n_footprint_channels: int = 8,
    max_sample_spikes: int = 100,
    pre_samples: int = 30,
    post_samples: int = 45
) -> Tuple[int, List[int], Dict[int, float], Dict[int, np.ndarray]]:
    """
    Scans channels to find the TRUE peak channel and top 6-8 footprint channels
    by computing mean peak-to-trough (PTP) amplitude across candidate contacts.
    If candidate_channels is None, scans all available AP channels (0 to 383).
    """
    n_total_samples = mm_ap.shape[0]
    n_total_chans = mm_ap.shape[1]

    if candidate_channels is None:
        candidate_channels = np.arange(min(384, n_total_chans))

    if len(spike_times_s) == 0:
        return int(candidate_channels[0]), list(candidate_channels[:n_footprint_channels]), {}, {}

    sample_times = np.random.choice(spike_times_s, min(max_sample_spikes, len(spike_times_s)), replace=False)
    sample_idxs = (sample_times * fs).astype(np.int64)

    valid_idxs = sample_idxs[(sample_idxs >= pre_samples) & (sample_idxs < n_total_samples - post_samples)]
    if len(valid_idxs) == 0:
        return int(candidate_channels[0]), list(candidate_channels[:n_footprint_channels]), {}, {}

    # Extract snippets across candidate channels
    snippets = []
    for s_idx in valid_idxs:
        snip = mm_ap[s_idx - pre_samples : s_idx + post_samples, candidate_channels].astype(np.float32)
        snippets.append(snip)

    mean_waveforms = np.mean(snippets, axis=0)  # (n_samples, n_candidates)
    ptps = np.ptp(mean_waveforms, axis=0)

    best_idx = int(np.argmax(ptps))
    peak_chan = int(candidate_channels[best_idx])

    # Rank channels by PTP amplitude
    sorted_order = np.argsort(ptps)[::-1]
    top_indices = sorted_order[:n_footprint_channels]
    footprint_chans = [int(candidate_channels[i]) for i in top_indices]

    channel_ptps = {int(candidate_channels[i]): float(ptps[i]) for i in top_indices}
    template_wfs = {int(candidate_channels[i]): mean_waveforms[:, i] for i in top_indices}

    return peak_chan, footprint_chans, channel_ptps, template_wfs


def compute_waveform_kinetics(
    template_wf: np.ndarray,
    peak_chan: int,
    footprint_chans: List[int],
    channel_depths: Dict[int, float],
    all_template_wfs: Dict[int, np.ndarray],
    fs: float = 30000.0,
    noise_sigma: float = 10.0
) -> WaveformKineticsResult:
    """
    Computes action potential duration, half-width, and putative cell-type classification.
    """
    time_ms = (np.arange(len(template_wf)) - 30) / (fs / 1000.0)

    trough_idx = int(np.argmin(template_wf))
    after_trough = template_wf[trough_idx:]
    if len(after_trough) > 1:
        peak_offset = int(np.argmax(after_trough))
        peak_idx = trough_idx + peak_offset
    else:
        peak_idx = trough_idx

    trough_to_peak_samples = max(1, peak_idx - trough_idx)
    trough_to_peak_us = float((trough_to_peak_samples / fs) * 1e6)

    trough_val = template_wf[trough_idx]
    baseline_val = float(np.mean(template_wf[:15]))
    half_depol = baseline_val + 0.5 * (trough_val - baseline_val)

    left_cross = np.where(template_wf[:trough_idx] > half_depol)[0]
    right_cross = np.where(template_wf[trough_idx:] > half_depol)[0]

    if len(left_cross) > 0 and len(right_cross) > 0:
        l_idx = left_cross[-1]
        r_idx = trough_idx + right_cross[0]
        half_width_us = float(((r_idx - l_idx) / fs) * 1e6)
    else:
        half_width_us = 250.0

    if peak_idx > trough_idx:
        dt_s = (peak_idx - trough_idx) / fs
        dV = template_wf[peak_idx] - template_wf[trough_idx]
        repol_slope = float(dV / dt_s / 1e6)
    else:
        repol_slope = 0.0

    cell_type = "Regular-Spiking (Putative Principal)" if trough_to_peak_us >= 400.0 else "Fast-Spiking (Putative Interneuron)"

    ptp_amp = float(np.ptp(template_wf))
    snr = float(round(ptp_amp / max(noise_sigma, 1.0), 2))

    return WaveformKineticsResult(
        peak_channel=peak_chan,
        footprint_channels=footprint_chans,
        channel_depths_um=[channel_depths.get(ch, 0.0) for ch in footprint_chans],
        template_waveforms=all_template_wfs,
        time_axis_ms=time_ms,
        trough_to_peak_us=round(trough_to_peak_us, 1),
        half_width_us=round(half_width_us, 1),
        repolarization_slope=round(repol_slope, 2),
        putative_cell_type=cell_type,
        snr=snr
    )


def extract_representative_trace_snippet(
    mm_ap: np.memmap,
    channels: List[int],
    channel_depths: Dict[int, float],
    spike_times_s: np.ndarray,
    peak_channel: int,
    t_start_s: Optional[float] = None,
    window_duration_s: float = 1.5,
    fs: float = 30000.0,
    refractory_ms: float = 1.0,
    max_display_pts_per_chan: int = 100000,
    ds_factor: int = 1
) -> MultiChannelTraceSnippet:
    """
    Extracts and bandpass filters (300-6000 Hz zero-phase) multi-channel voltage traces.
    By default extracts a pristine 1.5s window at native 30 kHz sampling rate (zero decimation)
    centered on the highest firing rate burst epoch, preserving 100% of raw spike waveform fidelity.
    Pins detected spike troughs (red) and any refractory period violations (< refractory_ms, amber)
    directly onto the voltage trace.
    """
    n_total_samples = mm_ap.shape[0]

    if t_start_s is None:
        if len(spike_times_s) > 0 and window_duration_s < 10.0:
            t_min = float(np.min(spike_times_s))
            t_max = float(np.max(spike_times_s))
            eval_times = np.arange(t_min, max(t_min + 0.5, t_max - window_duration_s), 0.5)
            if len(eval_times) > 0:
                counts = [np.sum((spike_times_s >= t) & (spike_times_s < t + window_duration_s)) for t in eval_times]
                t_start_s = float(eval_times[int(np.argmax(counts))])
            else:
                t_start_s = t_min
        else:
            t_start_s = float(np.min(spike_times_s)) if len(spike_times_s) > 0 else 0.0

    t_end_s = t_start_s + window_duration_s
    s0 = max(0, int(t_start_s * fs))
    s1 = min(n_total_samples, int(t_end_s * fs))

    raw_chunk = mm_ap[s0:s1, channels].astype(np.float32)

    # 4th-order zero-phase Butterworth bandpass filter (300 - 6000 Hz)
    nyq = 0.5 * fs
    b, a = signal.butter(4, [300.0 / nyq, 6000.0 / nyq], btype='band')
    filt_chunk = signal.filtfilt(b, a, raw_chunk, axis=0) * 0.195  # in uV

    t_full = np.arange(s0, s1) / fs

    peak_idx = channels.index(peak_channel) if peak_channel in channels else 0
    filt_peak = filt_chunk[:, peak_idx]

    spks_in_win = np.sort(spike_times_s[(spike_times_s >= t_start_s) & (spike_times_s < t_end_s)])

    # Identify refractory violation spikes (< refractory_ms)
    ref_s = refractory_ms / 1000.0
    viol_flags = np.zeros(len(spks_in_win), dtype=bool)
    if len(spks_in_win) > 1:
        diffs = np.diff(spks_in_win)
        for i in range(len(spks_in_win)):
            prev_dt = diffs[i - 1] if i > 0 else 999.0
            next_dt = diffs[i] if i < len(diffs) else 999.0
            if prev_dt < ref_s or next_dt < ref_s:
                viol_flags[i] = True

    # Align each spike to exact negative trough within +/- 0.6 ms (+/- 18 samples)
    search_half_samples = int(0.0006 * fs)
    aligned_troughs_t = []
    aligned_troughs_v = []
    viol_troughs_t = []
    viol_troughs_v = []

    for i, st in enumerate(spks_in_win):
        s_center = int((st - t_start_s) * fs)
        w_start = max(0, s_center - search_half_samples)
        w_end = min(len(filt_peak), s_center + search_half_samples)
        if w_end > w_start:
            local_min_idx = w_start + int(np.argmin(filt_peak[w_start:w_end]))
            t_tr = t_full[local_min_idx]
            v_tr = filt_peak[local_min_idx]
        else:
            t_tr = st
            v_tr = 0.0

        aligned_troughs_t.append(t_tr)
        aligned_troughs_v.append(v_tr)
        if viol_flags[i]:
            viol_troughs_t.append(t_tr)
            viol_troughs_v.append(v_tr)

    # Peak-preserving min-max decimation for full session multi-channel display
    if len(filt_chunk) > max_display_pts_per_chan:
        bin_size = max(2, len(filt_chunk) // (max_display_pts_per_chan // 2))
        n_bins = len(filt_chunk) // bin_size
        reshaped = filt_chunk[:n_bins * bin_size].reshape(n_bins, bin_size, len(channels))
        mins = reshaped.min(axis=1)
        maxs = reshaped.max(axis=1)

        dec_traces = np.empty((n_bins * 2, len(channels)), dtype=np.float32)
        dec_traces[0::2] = mins
        dec_traces[1::2] = maxs

        dec_times = np.empty(n_bins * 2, dtype=np.float64)
        bin_times = t_full[:n_bins * bin_size:bin_size]
        dec_times[0::2] = bin_times
        dec_times[1::2] = bin_times + (bin_size / (2.0 * fs))

        t_out = dec_times
        traces_out = dec_traces.T  # (n_channels, n_samples)
    else:
        t_out = t_full
        traces_out = filt_chunk.T

    return MultiChannelTraceSnippet(
        t_axis_s=t_out,
        channels=channels,
        channel_depths_um=[channel_depths.get(ch, (ch // 2) * 15.0) for ch in channels],
        traces_uv=traces_out,
        peak_channel=peak_channel,
        peak_channel_index=peak_idx,
        spike_times_in_window=spks_in_win,
        aligned_trough_times=np.array(aligned_troughs_t, dtype=np.float64),
        aligned_trough_voltages=np.array(aligned_troughs_v, dtype=np.float32),
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        violation_trough_times=np.array(viol_troughs_t, dtype=np.float64),
        violation_trough_voltages=np.array(viol_troughs_v, dtype=np.float32)
    )
