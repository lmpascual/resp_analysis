"""Diaphragm motor-unit spikes outside the inspiratory bursts ("tonic" activity) and ECG R peaks, on the imec clock.

The integrated diaphragm (cibrrig) shows the bursts; tonic activity is single motor-unit spikes between them. EMG
300-3000 Hz, |EMG| peaks above 6 x the recording's median noise (MAD), at least 1 ms apart. The ECG leaks into the
EMG at a lag that differs between animals, so for each 10-minute block the artifact window is measured: the median
|EMG| around the R peaks (ECG 10-200 Hz, the larger-polarity deflection, >= 50 ms apart), where it is > 2 x its floor
around a maximum > 3 x the floor, +-1 ms (default -3 to +12 ms if there is no clear artifact). Spikes in it are
dropped. Expiratory activity per breath = spikes from inspiration offset + 40 ms to the next onset - 30 ms.

    events = emg_ecg_events(nidq_bin, n_channels, emg_ch, ecg_ch, sync_map_json, start_imec_s, duration_s)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 1e4


def _read(raw, rows, ch, rng):
    return np.asarray(raw[rows[0]:rows[-1] + 1, ch], dtype=np.float64)[rows - rows[0]] * (rng / 32768)


def emg_ecg_events(nidq_bin, n_channels, emg_ch, ecg_ch, sync_map, start_s, duration_s, block_s=600.0,
                   thr_sd=6.0):
    """EMG spike times (all, and with ECG-artifact spikes removed) and R-peak times, in s from start_s (imec)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nidq_sync import nidq_rows
    meta = Path(nidq_bin).with_suffix(".meta")
    rng = next((float(x.split("=", 1)[1]) for x in meta.read_text().splitlines() if x.startswith("niAiRangeMax=")), 10.0)
    raw = np.memmap(nidq_bin, dtype="int16", mode="r").reshape(-1, n_channels)
    smap = json.loads(Path(sync_map).read_text()) if sync_map else None
    n_tot = int(duration_s * FS)
    sos_emg = butter(4, [300, 3000], btype="band", fs=FS, output="sos")
    sos_ecg = butter(2, [10, 200], btype="band", fs=FS, output="sos")
    blk, pad = int(block_s * FS), int(FS)
    pieces = []
    for s in range(0, n_tot, blk):
        a0, a1 = max(0, s - pad), min(n_tot, s + blk + pad)
        rows = nidq_rows(smap, start_s + a0 / FS, a1 - a0, FS) if smap else int(start_s * FS) + np.arange(a0, a1)
        rows = rows[rows < raw.shape[0]]
        pieces.append((s, a0, sosfiltfilt(sos_emg, _read(raw, rows, emg_ch, rng)),
                       sosfiltfilt(sos_ecg, _read(raw, rows, ecg_ch, rng))))
    noise = float(np.median([np.median(np.abs(e)) / 0.6745 for _, _, e, _ in pieces]))
    spk, amp, rpk, art, windows = [], [], [], [], []
    lag = np.arange(int(-0.03 * FS), int(0.04 * FS))
    for s, a0, e, k in pieces:
        sgn = -1.0 if np.percentile(k, 0.5) + np.percentile(k, 99.5) < 0 else 1.0
        pk, _ = find_peaks(sgn * k, distance=int(0.05 * FS), height=np.percentile(sgn * k, 99) * 0.4)
        # where the ECG sits in the EMG: mean |EMG| around R peaks (median over beats, so bursts don't dominate);
        # the artifact window is where it exceeds the noise floor by 3 x, around its maximum
        ok = pk[(pk + lag[0] >= 0) & (pk + lag[-1] < e.size)]
        prof = np.median(np.abs(e[ok[:, None] + lag[None, :]]), axis=0) if ok.size > 20 else np.zeros(lag.size)
        floor = np.median(prof)
        i = int(np.argmax(prof))
        if prof[i] > 3 * floor:
            lo, hi = i, i
            while lo > 0 and prof[lo - 1] > 2 * floor:
                lo -= 1
            while hi < prof.size - 1 and prof[hi + 1] > 2 * floor:
                hi += 1
            w0, w1 = (lag[lo] - 0.001 * FS) / FS, (lag[hi] + 0.001 * FS) / FS
        else:
            w0, w1 = -0.003, 0.012
        windows.append((w0, w1))
        sp, pr = find_peaks(np.abs(e), height=thr_sd * noise, distance=int(0.001 * FS))
        keep = (sp + a0 >= s) & (sp + a0 < s + blk)
        sp, h = sp[keep], pr["peak_heights"][keep]
        r_blk = pk[(pk + a0 >= s) & (pk + a0 < s + blk)]
        j = np.searchsorted(pk, sp - w1 * FS)                    # first R peak that could own the spike
        a_ = np.zeros(sp.size, bool)
        for dj in (0, 1, 2):
            jj = np.clip(j + dj, 0, max(pk.size - 1, 0))
            if pk.size:
                dd = (sp - pk[jj]) / FS
                a_ |= (dd >= w0) & (dd <= w1)
        spk.append(sp + a0)
        amp.append(h)
        art.append(a_)
        rpk.append(r_blk + a0)
    rpk = np.concatenate(rpk) / FS
    spk = np.concatenate(spk) / FS
    amp = np.concatenate(amp)
    art = np.concatenate(art)
    return {"emg_spikes_s": spk[~art], "emg_amp_v": amp[~art], "emg_spikes_all_s": spk, "r_peaks_s": rpk,
            "emg_noise_v": noise, "ecg_artifact_windows_s": np.array(windows)}


def expiratory_counts(spike_t, on_s, off_s, lead=0.04, trail=0.03):
    """Spikes and duration of each breath's expiratory window (offset + lead to next onset - trail)."""
    e0 = off_s + lead
    e1 = np.r_[on_s[1:], np.nan] - trail
    ok = e1 > e0
    n = np.zeros(on_s.size)
    n[ok] = np.searchsorted(spike_t, e1[ok]) - np.searchsorted(spike_t, e0[ok])
    return np.where(ok, n, np.nan), np.where(ok, e1 - e0, np.nan)
