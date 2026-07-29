"""
Optogenetics Respiratory Phase Response & Phase Perturbation Processor.
Quantifies laser pulse timing relative to respiratory phase and measures phase advances or delays of next inspiration.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..models import EMGProcessed, PulseEvents

@dataclass
class OptoPhaseTrial:
    opto_onset: float           # Laser pulse start time (s)
    opto_duration: float        # Laser pulse duration (s)
    prev_breath_time: float     # Inspiration onset immediately before opto pulse (s)
    next_breath_time: float     # Inspiration onset immediately after opto pulse (s)
    control_cycle_t0: float     # Baseline control inter-breath interval (s)
    perturbed_cycle_t1: float   # Cycle duration containing opto pulse (s)
    opto_phase_phi: float       # Phase of stimulus onset within cycle (0 <= phi < 1)
    phase_shift_delta: float    # Normalized phase shift: (T1 - T0) / T0 (>0 = delay, <0 = advance)

@dataclass
class OptoPhaseAnalysis:
    trials: List[OptoPhaseTrial] = field(default_factory=list)
    mean_phase_shift_by_bin: np.ndarray = field(default_factory=lambda: np.array([]))
    phase_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    num_opto_pulses: int = 0
    phase_advances_count: int = 0
    phase_delays_count: int = 0

def analyze_opto_respiratory_phase(emg: EMGProcessed, opto: PulseEvents) -> OptoPhaseAnalysis:
    """
    Analyze optogenetic stimulus phase & perturbation on diaphragm breath timing.
    For each laser pulse:
    1. Find preceding breath onset T_prev and next breath onset T_next.
    2. Calculate baseline control cycle T0 from pre-stimulus inter-breath interval.
    3. Compute stimulus phase phi = (T_opto - T_prev) / T0.
    4. Compute perturbed cycle T1 = T_next - T_prev.
    5. Compute normalized phase shift delta_phi = (T1 - T0) / T0.
    """
    breaths = emg.breath_onsets
    onsets = opto.onsets
    durations = opto.durations

    if len(breaths) < 3 or len(onsets) == 0:
        return OptoPhaseAnalysis()

    trials = []
    for i, opto_t in enumerate(onsets):
        opto_dur = durations[i] if i < len(durations) else 0.1

        # Find preceding breath onset
        past_idx = np.where(breaths <= opto_t)[0]
        if len(past_idx) < 2:
            continue
        prev_idx = past_idx[-1]
        prev_t = breaths[prev_idx]

        # Baseline control cycle T0 from prior breath cycle
        t0 = prev_t - breaths[prev_idx - 1]
        if t0 <= 0.05 or t0 >= 3.0:  # Ignore unrealistic baseline cycles
            continue

        # Next breath onset following opto pulse
        future_idx = np.where(breaths > opto_t)[0]
        if len(future_idx) == 0:
            continue
        next_idx = future_idx[0]
        next_t = breaths[next_idx]

        # Perturbed cycle T1
        t1 = next_t - prev_t
        if t1 <= 0.05 or t1 >= 5.0:
            continue

        # Phase phi in [0, 1) and phase shift delta_phi
        phi = (opto_t - prev_t) / t0
        if phi < 0.0 or phi >= 1.5:  # Filter out of bound anomalies
            continue

        delta_phi = (t1 - t0) / t0

        trials.append(OptoPhaseTrial(
            opto_onset=float(opto_t),
            opto_duration=float(opto_dur),
            prev_breath_time=float(prev_t),
            next_breath_time=float(next_t),
            control_cycle_t0=float(t0),
            perturbed_cycle_t1=float(t1),
            opto_phase_phi=float(phi),
            phase_shift_delta=float(delta_phi)
        ))

    # Calculate Phase Response Curve (PRC) binned averages across [0, 1] phase
    bins = np.linspace(0.0, 1.0, 11)
    mean_shifts = np.zeros(10)
    if len(trials) > 0:
        phis = np.array([tr.opto_phase_phi for tr in trials])
        shifts = np.array([tr.phase_shift_delta for tr in trials])
        for b in range(10):
            mask = (phis >= bins[b]) & (phis < bins[b+1])
            if np.any(mask):
                mean_shifts[b] = np.mean(shifts[mask])

    advances = sum(1 for tr in trials if tr.phase_shift_delta < -0.05)
    delays = sum(1 for tr in trials if tr.phase_shift_delta > 0.05)

    return OptoPhaseAnalysis(
        trials=trials,
        mean_phase_shift_by_bin=mean_shifts,
        phase_bins=(bins[:-1] + bins[1:]) / 2.0,
        num_opto_pulses=len(trials),
        phase_advances_count=advances,
        phase_delays_count=delays
    )
