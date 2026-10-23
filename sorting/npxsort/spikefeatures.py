"""Per-spike features read from the preprocessed binary in one sequential pass.

Used by the review browser (footprint-mismatch flags) and by the epoch-stability stage (does a
unit's waveform change between named epochs, e.g. eupnea vs gasping?).

Footprint: for each unit, up to FOOTPRINT_CHANNELS channels (largest template peak-to-peak first)
that hold at least FOOTPRINT_MIN_FRACTION of the largest channel's peak-to-peak. For every spike the
signed amplitude on each is the extremum of the template's sign within ±AMP_HALF_WINDOW samples of
the template's extremum time on that channel.
"""
from __future__ import annotations

import numpy as np

FOOTPRINT_CHANNELS = 5
FOOTPRINT_MIN_FRACTION = 0.3
AMP_HALF_WINDOW = 6              # samples (±0.2 ms)
REVERSAL_HALF_WINDOW = 15        # samples (±0.5 ms) searched for an opposite-sign peak
WAVEFORM_CHANNELS = 8            # channels kept for per-epoch mean waveforms
WAVEFORM_BEFORE, WAVEFORM_AFTER = 30, 60   # samples (1 ms before, 2 ms after at 30 kHz)


def footprint_setup(dense: np.ndarray, ptp: np.ndarray, nbefore: int):
    """Per unit: footprint channels, extremum offsets (samples from the spike time), signed template
    extremum values (0 = unused slot), and the template's own opposite-to-same peak ratio on the
    largest channel (used to judge polarity reversals)."""
    n = dense.shape[0]
    ch = np.zeros((n, FOOTPRINT_CHANNELS), dtype=np.int64)
    off = np.zeros((n, FOOTPRINT_CHANNELS), dtype=np.int64)
    tv = np.zeros((n, FOOTPRINT_CHANNELS), dtype=np.float32)
    opp_ratio = np.zeros(n, dtype=np.float32)
    for i in range(n):
        best = np.argsort(-ptp[i])[:FOOTPRINT_CHANNELS]
        keep = best[ptp[i, best] >= FOOTPRINT_MIN_FRACTION * ptp[i, best[0]]]
        for j, c in enumerate(keep):
            w = dense[i, :, c]
            k = int(np.argmax(np.abs(w)))
            ch[i, j], off[i, j], tv[i, j] = c, k - nbefore, w[k]
            if j == 0:
                seg = w[max(0, k - REVERSAL_HALF_WINDOW):k + REVERSAL_HALF_WINDOW + 1]
                opp = seg.min() if w[k] > 0 else seg.max()
                opp_ratio[i] = abs(opp) / max(abs(w[k]), 1e-6)
        ch[i, len(keep):] = keep[0]
    return ch, off, tv, opp_ratio


def polarity_reversed(opp_ratio: np.ndarray, opp_sigma: np.ndarray, template_opp_ratio: np.ndarray,
                      min_ratio: float = 2.0, template_factor: float = 2.0, min_sigma: float = 10.0) -> np.ndarray:
    """Spikes whose largest-channel deflection has the opposite sign to the template's: the
    opposite peak is > min_ratio x the same-sign one, > template_factor x the template's own
    ratio, and > min_sigma noise SDs. Defaults calibrated on e097: they catch all 26 spikes of the
    positive gasping unit absorbed into KS4 #361 (ratio 4-9, 15-19 SD) and flag 0.14% of all spikes."""
    with np.errstate(invalid="ignore"):
        return ((opp_ratio > min_ratio) & (opp_ratio > template_factor * template_opp_ratio)
                & (opp_sigma > min_sigma))


def noise_levels(traces, gain, n_samples: int, fs: float, n_chunks: int = 10, seed: int = 0) -> np.ndarray:
    """Per-channel noise SD (µV): MAD of n_chunks random 1-s chunks."""
    rng = np.random.default_rng(seed)
    n = int(fs)
    x = np.concatenate([np.asarray(traces[s0:s0 + n], dtype=np.float32)
                        for s0 in rng.integers(0, n_samples - n, n_chunks)])
    return (np.median(np.abs(x - np.median(x, axis=0)), axis=0) / 0.6745 * gain).astype(np.float32)


def spike_pass(traces, gain, offset, noise, n_samples: int, fs: float, setups: dict, spike_vectors: dict,
               epochs: list | None = None, waveform_channels: dict | None = None) -> dict:
    """One sequential pass over the binary.

    setups[name] = footprint_setup(...); spike_vectors[name] = (sample, unit) of a time-sorted spike
    vector. Returns per sorter a dict with, per spike:
      amp      (n, K) signed footprint amplitudes (µV; 0 in unused slots)
      alpha    best overall scale against the template (a.t / t.t)
      rel      |a - alpha t| / |alpha t|   (NaN with < 2 footprint channels; inf if alpha <= 0)
      sig      sqrt(sum(((a - alpha t) / noise)^2))
      opp_ratio opposite-sign peak / same-sign peak on the largest channel (within ±0.5 ms); compare
               with the unit's template value (footprint_setup's opp_ratio) to find polarity reversals
      opp_sigma that opposite-sign peak in noise SDs
    and, if epochs [(start_sample, end_sample), ...] and waveform_channels[name] (n_units, C) are
    given, per-unit per-epoch sums and counts of the waveform on those channels ("wf_sum", "wf_n").
    """
    win = np.arange(-AMP_HALF_WINDOW, AMP_HALF_WINDOW + 1)
    rwin = np.arange(-REVERSAL_HALF_WINDOW, REVERSAL_HALF_WINDOW + 1)
    wwin = np.arange(-WAVEFORM_BEFORE, WAVEFORM_AFTER)
    margin = 256
    out = {}
    for name, (sample, unit) in spike_vectors.items():
        k = setups[name][0].shape[1]
        out[name] = {"amp": np.zeros((sample.size, k), dtype=np.float32),
                     "alpha": np.full(sample.size, np.nan, dtype=np.float32),
                     "rel": np.full(sample.size, np.nan, dtype=np.float32),
                     "sig": np.full(sample.size, np.nan, dtype=np.float32),
                     "opp_ratio": np.full(sample.size, np.nan, dtype=np.float32),
                     "opp_sigma": np.full(sample.size, np.nan, dtype=np.float32)}
        if epochs and waveform_channels is not None:
            nu, c = waveform_channels[name].shape
            out[name]["wf_sum"] = np.zeros((nu * len(epochs), wwin.size, c), dtype=np.float64)
            out[name]["wf_n"] = np.zeros(nu * len(epochs), dtype=np.int64)
    ep = np.array(epochs, dtype=np.int64).reshape(-1, 2) if epochs else None
    step = int(fs)
    for c0 in range(0, n_samples, step):
        c1 = min(n_samples, c0 + step)
        if not any(np.searchsorted(s, c0) < np.searchsorted(s, c1) for s, _ in spike_vectors.values()):
            continue  # no spikes in this chunk: don't read it
        a0, a1 = max(0, c0 - margin), min(n_samples, c1 + margin)
        block = np.asarray(traces[a0:a1])
        last = block.shape[0] - 1
        for name, (sample, unit) in spike_vectors.items():
            i0, i1 = np.searchsorted(sample, [c0, c1])
            if i1 == i0:
                continue
            ch, off, tv, opp_ratio = setups[name]
            o = out[name]
            u = unit[i0:i1]
            rel_t = sample[i0:i1] - a0
            idx = np.clip(rel_t[:, None, None] + off[u][:, :, None] + win[None, None, :], 0, last)
            chu = ch[u]
            seg = block[idx, chu[:, :, None]].astype(np.float32)
            t = tv[u]
            amp = np.where(t > 0, seg.max(axis=2), seg.min(axis=2)) * gain[chu] + offset[chu]
            used = t != 0
            amp = np.where(used, amp, 0.0)
            tt = (t * t).sum(axis=1)
            alpha = (amp * t).sum(axis=1) / tt
            d = np.where(used, amp - alpha[:, None] * t, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.sqrt((d ** 2).sum(axis=1)) / (np.abs(alpha) * np.sqrt(tt))
            sig = np.sqrt(((d / noise[chu]) ** 2).sum(axis=1))
            rel[alpha <= 0] = np.inf
            few = used.sum(axis=1) < 2
            rel[few] = np.nan
            sig[few] = np.nan
            # polarity reversal on the largest channel
            ridx = np.clip(rel_t[:, None] + off[u][:, 0:1] + rwin[None, :], 0, last)
            rseg = block[ridx, chu[:, 0:1]].astype(np.float32) * gain[chu[:, 0:1]] + offset[chu[:, 0:1]]
            same = np.abs(amp[:, 0])
            opp = np.where(t[:, 0] > 0, -rseg.min(axis=1), rseg.max(axis=1))
            o["amp"][i0:i1] = amp
            o["alpha"][i0:i1] = alpha
            o["rel"][i0:i1] = rel
            o["sig"][i0:i1] = sig
            o["opp_ratio"][i0:i1] = opp / np.maximum(same, 1e-3)
            o["opp_sigma"][i0:i1] = opp / noise[chu[:, 0]]
            if "wf_sum" in o:
                s = sample[i0:i1]
                e = np.full(s.size, -1, dtype=np.int64)
                for j, (e0, e1) in enumerate(ep):
                    e[(s >= e0) & (s < e1)] = j
                keep = e >= 0
                if keep.any():
                    key = u[keep] * len(ep) + e[keep]
                    srt = np.argsort(key, kind="stable")
                    key = key[srt]
                    wch = waveform_channels[name][u[keep][srt]]
                    widx = np.clip(rel_t[keep][srt][:, None] + wwin[None, :], 0, last)
                    wf = block[widx[:, :, None], wch[:, None, :]].astype(np.float32) * gain[wch][:, None, :]
                    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
                    o["wf_sum"][key[starts]] += np.add.reduceat(wf, starts, axis=0)
                    o["wf_n"][key[starts]] += np.diff(np.r_[starts, key.size])
    return out
