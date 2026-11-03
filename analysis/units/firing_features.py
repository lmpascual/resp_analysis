"""Firing features of a unit in a set of breath cycles, and how alike two units' features are.

Used to propose candidate unit matches between windows (never to accept them: the firing changes are what the analyses
measure) and to benchmark how much a known neuron's firing drifts between windows and across epoch transitions.

Per unit, over the breath cycles of a condition (onset to next onset):
  rate_hz       spikes / time
  curve         rate in each of 18 phase bins (cibrrig phi: 0 = inspiration onset, pi = offset), Hz
  L, pref       vector-mean length and angle of the curve (2025 classes: tonic if L < 0.2)
  peth          rate around inspiration onset, -400 to +600 ms in 20 ms bins, Hz
  acg           autocorrelogram 1-50 ms (1 ms bins), as spikes / s at each lag (rate-normalised shape compared)
  cv2           median CV2 of consecutive inter-spike intervals (local irregularity, burstiness)
Similarity of two feature sets: log2 rate ratio, Pearson r of the phase curves and of the PETHs, angular difference of
the preferred phases, difference of L, Pearson r of the ACGs, difference of CV2.
"""
from __future__ import annotations

import numpy as np

PHASE_BINS = 18
PETH_EDGES = np.arange(-0.4, 0.6 + 1e-9, 0.02)
ACG_EDGES = np.arange(0.001, 0.051, 0.001)
TONIC_L = 0.2


def phase_from_breaths(on, off, dt=1e-3):
    """cibrrig's phi (physiology.compute_dia_phase) rebuilt from onsets / offsets: 0 at inspiration onset, rising
    linearly to pi at offset, -pi just after offset, rising linearly to 0 at the next onset; NaN outside the breaths.
    Lets the npxsort env (no cibrrig) use a population run's breaths.tsv."""
    t = np.arange(on[0], on[-1], dt)
    phi = np.full(t.size, np.nan)
    k = np.searchsorted(on, t, side="right") - 1
    ok = (k >= 0) & (k < on.size - 1)
    k = k[ok]
    tt = t[ok]
    insp = tt < off[k]
    v = np.empty(tt.size)
    v[insp] = np.pi * (tt[insp] - on[k[insp]]) / np.maximum(off[k[insp]] - on[k[insp]], 1e-9)
    e = ~insp
    v[e] = -np.pi + np.pi * (tt[e] - off[k[e]]) / np.maximum(on[k[e] + 1] - off[k[e]], 1e-9)
    phi[ok] = v
    return t, phi


def cycle_spans(onsets, mask):
    """(onset, next onset) of the cycles whose onset index is selected by mask (length = len(onsets))."""
    idx = np.flatnonzero(np.asarray(mask)[:-1])
    return np.column_stack([onsets[idx], onsets[idx + 1]])


def in_spans(t, spans):
    k = np.searchsorted(spans[:, 0], t, side="right") - 1
    ok = k >= 0
    ok[ok] = t[ok] < spans[k[ok], 1]
    return ok


def features(spk, spans, onsets, phi_t, phi):
    """Feature dict of one unit's spike times (s) within spans (n x 2), with the phase trace phi(phi_t)."""
    spans = np.asarray(spans, float)
    dur = float(np.sum(spans[:, 1] - spans[:, 0]))
    s = spk[in_spans(spk, spans)] if spk.size else spk
    edges = np.linspace(-np.pi, np.pi, PHASE_BINS + 1)
    m = in_spans(phi_t, spans) & np.isfinite(phi)
    occ = np.histogram(phi[m], edges)[0] * float(np.median(np.diff(phi_t)))
    p = phi[np.clip(np.searchsorted(phi_t, s), 0, phi.size - 1)]
    curve = np.histogram(p[np.isfinite(p)], edges)[0] / np.maximum(occ, 1e-9)
    centres = (edges[:-1] + edges[1:]) / 2
    z = np.sum(curve * np.exp(1j * centres)) / max(curve.sum(), 1e-12)
    on = onsets[(onsets >= spans[0, 0]) & (onsets < spans[-1, 1])]
    on = on[in_spans(on, spans)]
    peth = np.zeros(PETH_EDGES.size - 1)
    if on.size and s.size:
        k0 = np.searchsorted(s, on + PETH_EDGES[0])
        k1 = np.searchsorted(s, on + PETH_EDGES[-1])
        d = np.concatenate([s[a:b] - t for a, b, t in zip(k0, k1, on)]) if (k1 > k0).any() else np.array([])
        peth = np.histogram(d, PETH_EDGES)[0] / (on.size * np.diff(PETH_EDGES))
    acg = np.zeros(ACG_EDGES.size - 1)
    if s.size > 1:
        k1 = np.searchsorted(s, s + ACG_EDGES[-1])
        parts = [s[i + 1:j] - s[i] for i, j in enumerate(k1) if j > i + 1]
        d = np.concatenate(parts) if parts else np.array([])
        acg = np.histogram(d, ACG_EDGES)[0] / max(s.size, 1) / np.diff(ACG_EDGES)
    isi = np.diff(s)
    cv2 = float(np.median(2 * np.abs(np.diff(isi)) / (isi[1:] + isi[:-1]))) if isi.size > 2 else np.nan
    return {"rate_hz": s.size / max(dur, 1e-9), "n_spikes": int(s.size), "curve": curve, "L": float(abs(z)),
            "pref": float(np.angle(z)), "peth": peth, "acg": acg, "cv2": cv2}


def _r(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def combined_score(s, La, Lb):
    """One number per pair (higher = more alike), as benchmarked on e097 (firing_benchmark.py): phase-curve r (only
    when both units are phase-locked), onset PETH r, ACG r, minus |log2 rate ratio|, |CV2 difference|, |L difference|."""
    c = (s["curve_r"] if np.isfinite(s["curve_r"]) else 0.0) if (La >= TONIC_L and Lb >= TONIC_L) else 0.0
    f = lambda x, default: x if np.isfinite(x) else default  # noqa: E731
    return (c + f(s["peth_r"], 0.0) + f(s["acg_r"], 0.0) - abs(s["log2_rate_ratio"]) - abs(f(s["cv2_diff"], 1.0))
            - abs(s["L_diff"]))


def similarity(fa, fb):
    d = {"log2_rate_ratio": float(np.log2((fb["rate_hz"] + 0.05) / (fa["rate_hz"] + 0.05))),
         "curve_r": _r(fa["curve"], fb["curve"]),
         "pref_diff_rad": float(abs(np.angle(np.exp(1j * (fb["pref"] - fa["pref"]))))),
         "L_diff": fb["L"] - fa["L"],
         "peth_r": _r(fa["peth"], fb["peth"]),
         "acg_r": _r(fa["acg"], fb["acg"]),
         "cv2_diff": fb["cv2"] - fa["cv2"],
         "both_tonic": bool(fa["L"] < TONIC_L and fb["L"] < TONIC_L)}
    return d
