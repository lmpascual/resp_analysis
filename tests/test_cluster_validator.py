"""
Unit tests for the Phy-like single-cluster validation suite.
"""

import numpy as np
import pytest

from src.pipeline.diagnostics.cluster_validator import (
    compute_isi_and_violations,
    compute_respiratory_phase_vector,
    detect_respiratory_landmarks,
    compute_amplitude_stationarity,
    compute_time_resolved_isi_pdfs,
    compute_autocorrelogram,
    compute_amplitude_cutoff,
    compute_waveform_kinetics
)


def test_isi_and_violations_known_contamination():
    # Construct spike train with exactly 3 refractory violations (< 1.5 ms)
    # Spikes at 10.000, 10.001 (1ms viol), 10.050, 10.0512 (1.2ms viol), 10.100, 10.101 (1ms viol), 10.200
    spikes = np.array([10.000, 10.001, 10.050, 10.0512, 10.100, 10.101, 10.200], dtype=np.float64)
    res = compute_isi_and_violations(spikes, refractory_ms=1.5)

    assert res.n_violations == 3
    assert res.total_spikes == 7
    assert res.contamination_pct == pytest.approx(3 / 7 * 100.0, rel=1e-2)
    assert res.violation_rate_q > 0.0


def test_respiratory_phase_vector_inspiratory():
    # 10 synthetic breaths: onset at 0, 1, 2, ... offset at 0.3, 1.3, 2.3, ...
    onsets = np.arange(0.0, 10.0, 1.0)
    offsets = onsets + 0.3

    # Generate spikes strictly during mid-inspiration (t_on + 0.15s -> ~90 deg)
    spikes = []
    for on in onsets[:-1]:
        spikes.extend([on + 0.14, on + 0.15, on + 0.16])
    spikes = np.array(spikes)

    res = compute_respiratory_phase_vector(spikes, onsets, offsets)

    assert res.functional_class == "Inspiratory"
    # Preferred phase should be near 90 deg
    assert 60.0 <= res.preferred_phase_deg <= 120.0
    assert res.mean_vector_r > 0.80
    assert res.rayleigh_p_value < 0.001


def test_respiratory_phase_vector_tonic_uniform():
    # Dense uniform Poisson-like spikes across breath cycles
    np.random.seed(42)
    onsets = np.arange(0.0, 30.0, 1.0)
    offsets = onsets + 0.4
    spikes = np.sort(np.random.uniform(0.0, 29.0, 3000))

    res = compute_respiratory_phase_vector(spikes, onsets, offsets)
    # Uniform spikes should have very low vector strength R
    assert res.mean_vector_r < 0.15
    assert res.functional_class == "Tonic"


def test_detect_respiratory_landmarks_dual_mode():
    fs = 10000.0
    duration = 5.0
    t = np.arange(0, duration, 1.0 / fs)

    # Synthetic diaphragm EMG: burst of 300 Hz noise every 1.0s
    emg = np.zeros_like(t)
    for b in range(4):
        t0, t1 = b * 1.0 + 0.1, b * 1.0 + 0.4
        idx = (t >= t0) & (t < t1)
        emg[idx] = np.random.normal(0, 1.0, np.sum(idx))

    # Synthetic flow: inward negative dip during inspiration, zero-crossing at t1
    flow = -np.sin(2 * np.pi * 1.0 * t)

    # 1. Test flow mode
    onsets, offsets, prov_flow, _ = detect_respiratory_landmarks(
        emg_raw=emg, flow_raw=flow, t_nidq=t, fs_nidq=fs, offset_source='flow'
    )
    assert len(onsets) >= 2
    assert "Onset: Diaphragm EMG" in prov_flow

    # 2. Test EMG fallback mode
    onsets_e, offsets_e, prov_emg, _ = detect_respiratory_landmarks(
        emg_raw=emg, flow_raw=None, t_nidq=t, fs_nidq=fs, offset_source='emg'
    )
    assert len(onsets_e) >= 2
    assert "Offset: Diaphragm EMG" in prov_emg


def test_amplitude_stationarity_and_pdfs():
    times = np.linspace(0, 60.0, 1000)
    # Constant 100 uV amplitude with noise
    amps = 100.0 + np.random.normal(0, 5.0, 1000)

    st = compute_amplitude_stationarity(times, amps, n_time_bins=10)
    assert len(st.time_bin_centers_s) == 10
    assert abs(st.drift_slope_uv_per_min) < 15.0
    assert 90.0 < np.mean(st.median_amplitudes_uv) < 110.0

    cdfs = compute_time_resolved_isi_pdfs(times, n_epochs=5)
    assert len(cdfs.epoch_labels) == 5
    assert len(cdfs.epoch_cdfs) == 5
    assert len(cdfs.epoch_isis_ms) == 5
    assert len(cdfs.ks_drift_p_values) == 5
    assert cdfs.epoch_cdfs[0][-1] == 1.0
    assert cdfs.ks_drift_p_values[0] == 1.0


def test_autocorrelogram_and_cutoff():
    times = np.cumsum(np.random.exponential(0.02, 1000))
    acg = compute_autocorrelogram(times, bin_size_ms=1.0, max_lag_ms=50.0)
    assert len(acg.lags_ms) == 100

    np.random.seed(42)
    # Well-separated amplitudes (150 +/- 10 uV, far from 0)
    amps = np.random.normal(150, 10, 1000)
    cut = compute_amplitude_cutoff(amps)
    assert cut.estimated_missing_spikes_pct >= 0.0
    assert cut.passed_cutoff_qc is True
