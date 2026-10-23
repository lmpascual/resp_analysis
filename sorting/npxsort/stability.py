"""Does a unit's waveform change between named epochs (config: `epochs`)?

Built for state changes such as eupnea -> anoxia -> gasping -> recovery. When a unit's firing
pattern changes in an epoch (e.g. expiratory during eupnea, inspiratory during gasping), this tells
apart two explanations:
  - the same neuron changed its firing: its spatial footprint (relative amplitude across channels)
    and waveform stay the same, even if its amplitude scales with firing rate;
  - a different neuron with a similar spike took over: its footprint differs.

For every unit and every pair of epochs with at least MIN_SPIKES spikes each:
  footprint_cosine  cosine of the two epochs' mean normalised footprints (1 = identical shape)
  neighbour_cosine  the same measure between this unit and its most similar other unit: how different
                    a genuinely different neuron nearby looks (e097 median 0.96-0.975)
  waveform_corr     correlation of the two epochs' mean waveforms on the unit's top 8 channels
  amp_ratio         median spike amplitude (template scale) in epoch b / epoch a
  location_shift_um change in the footprint's centre of mass along the probe (b - a) ...
  shift_vs_probe_um ... minus the median shift of all units for that epoch pair (tissue movement
                    relative to the probe, e.g. ~0.3-0.7 µm during e097's gasping, moves every unit)
  auc, auc_null     how well a 2-fold cross-validated linear discriminant tells the two epochs' spikes
                    apart from normalised footprints (0.5 = not at all), and the same between the two
                    halves of one epoch. With thousands of spikes it detects even sub-micrometre
                    movement, so it shows that something changed, not that the neuron did.
  verdict           decided from effect size (footprint_cosine, waveform_corr), not from auc:
                      "same footprint"            cosine >= SAME_MIN_COS and waveform_corr >= SAME_MIN_WAVEFORM_CORR
                      "same footprint; a neighbour looks alike"  the above, but neighbour_cosine >=
                                                  SAME_MIN_COS, so a neighbour taking over would look the same
                      "footprint differs"         cosine < DIFF_MAX_COS (about the median distinct-neighbour level)
                      "unclear"                   in between

Outputs per sorter: epoch_stability.tsv (one row per unit and epoch pair), epoch_activity.tsv (spike
count and rate per unit and epoch) and epoch_templates.npz (mean waveform per unit and epoch).
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import spikeinterface.core as si

from .spikefeatures import WAVEFORM_BEFORE, WAVEFORM_CHANNELS, footprint_setup, noise_levels, spike_pass

log = logging.getLogger(__name__)

MIN_SPIKES = 30            # per epoch, for a comparison
MAX_SPIKES = 2000          # per epoch, subsampled for the discriminant
SAME_MIN_COS = 0.99        # e097: same unit, baseline vs recovery: median 0.9998, 5th percentile 0.995
SAME_MIN_WAVEFORM_CORR = 0.90
DIFF_MAX_COS = 0.97        # e097: nearest distinct neighbour: median 0.96 (KS4) / 0.975 (Lupin)


def binary_memmap(folder: Path, n_channels: int) -> np.memmap:
    info = json.loads((Path(folder) / "binary.json").read_text())["kwargs"]
    return np.memmap(Path(folder) / info["file_paths"][0], dtype=info["dtype"], mode="r",
                     offset=int(info.get("file_offset", 0) or 0)).reshape(-1, n_channels)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = rankdata(np.r_[pos, neg])
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def lda_auc(xa: np.ndarray, xb: np.ndarray, rng: np.random.Generator) -> float:
    """2-fold cross-validated AUC of a regularised Fisher discriminant between two spike sets."""
    if xa.shape[0] > MAX_SPIKES:
        xa = xa[rng.choice(xa.shape[0], MAX_SPIKES, replace=False)]
    if xb.shape[0] > MAX_SPIKES:
        xb = xb[rng.choice(xb.shape[0], MAX_SPIKES, replace=False)]
    fa, fb = rng.permutation(xa.shape[0]) % 2, rng.permutation(xb.shape[0]) % 2
    aucs = []
    for k in (0, 1):
        ta, tb = xa[fa != k], xb[fb != k]
        cov = (np.cov(ta, rowvar=False) * (len(ta) - 1) + np.cov(tb, rowvar=False) * (len(tb) - 1)) / (len(ta) + len(tb) - 2)
        cov = np.atleast_2d(cov) + 1e-3 * np.eye(xa.shape[1]) * max(np.trace(np.atleast_2d(cov)) / xa.shape[1], 1e-9)
        w = np.linalg.solve(cov, tb.mean(0) - ta.mean(0))
        aucs.append(_auc(xb[fb == k] @ w, xa[fa == k] @ w))
    a = float(np.mean(aucs))
    return max(a, 1 - a)   # direction learned on the other fold can flip; report separability


def verdict(cos: float, wcorr: float, neighbour_cos: float) -> str:
    if np.isnan(cos):
        return ""
    if cos < DIFF_MAX_COS:
        return "footprint differs"
    if cos >= SAME_MIN_COS and (np.isnan(wcorr) or wcorr >= SAME_MIN_WAVEFORM_CORR):
        if not np.isnan(neighbour_cos) and neighbour_cos >= SAME_MIN_COS:
            return "same footprint; a neighbour looks alike"
        return "same footprint"
    return "unclear"


def neighbour_cosines(dense: np.ndarray, ch: np.ndarray, off: np.ndarray, tv: np.ndarray, nbefore: int) -> np.ndarray:
    """For each unit: the highest footprint cosine with any other unit that is present (>= 30% of
    this unit's amplitude) on this unit's footprint channels, sampled at the same channels and times."""
    out = np.full(dense.shape[0], np.nan)
    for i in range(dense.shape[0]):
        used = tv[i] != 0
        if used.sum() < 2:
            continue
        t = tv[i][used]
        vals = dense[:, off[i][used] + nbefore, ch[i][used]]
        cos = (vals @ t) / (np.linalg.norm(vals, axis=1) * np.linalg.norm(t) + 1e-9)
        cos[i] = -np.inf
        cos[np.abs(vals).max(axis=1) < 0.3 * np.abs(t).max()] = -np.inf
        best = cos.max()
        if np.isfinite(best):
            out[i] = best
    return out


def run_stability(analyzers: dict, cfg: dict, rec: si.BaseRecording) -> dict:
    epochs = cfg["epochs"]
    if not epochs:
        return {}
    out_dir = Path(cfg["paths"]["output_dir"])
    fs, n = rec.get_sampling_frequency(), rec.get_num_samples()
    traces = binary_memmap(Path(cfg["paths"]["scratch_dir"]) / "preprocessed", rec.get_num_channels())
    gain = rec.get_channel_gains().astype("float32")
    offset = rec.get_channel_offsets().astype("float32")
    locs = rec.get_channel_locations()
    ep = [(int(round(e["start_s"] * fs)), min(n, int(round(e["end_s"] * fs)))) for e in epochs]
    names = [e["name"] for e in epochs]

    setups, vectors, wch, dense_all, nbefore = {}, {}, {}, {}, {}
    for name, an in analyzers.items():
        tm = an.get_extension("templates")
        dense = tm.get_data(operator="average").astype("float32")
        ptp = dense.max(axis=1) - dense.min(axis=1)
        setups[name] = footprint_setup(dense, ptp, tm.nbefore)
        sv = an.sorting.to_spike_vector()
        vectors[name] = (sv["sample_index"].astype(np.int64), sv["unit_index"].astype(np.int64))
        wch[name] = np.argsort(-ptp, axis=1)[:, :WAVEFORM_CHANNELS]
        dense_all[name], nbefore[name] = dense, tm.nbefore
    noise = noise_levels(traces, gain, n, fs)
    res = spike_pass(traces, gain, offset, noise, n, fs, setups, vectors, epochs=ep, waveform_channels=wch)

    summary = {}
    rng = np.random.default_rng(cfg["postprocessing"]["seed"])
    for name, an in analyzers.items():
        sample, unit = vectors[name]
        r = res[name]
        ch, off, tv, _ = setups[name]
        uids = list(an.unit_ids)
        nu, ne = len(uids), len(ep)
        e_of = np.full(sample.size, -1, dtype=np.int64)
        for j, (e0, e1) in enumerate(ep):
            e_of[(sample >= e0) & (sample < e1)] = j
        amp = r["amp"]
        norm = np.linalg.norm(amp, axis=1, keepdims=True)
        feat = amp / np.maximum(norm, 1e-6)
        ycom = (np.abs(amp) * locs[ch[unit], 1]).sum(axis=1) / np.maximum(np.abs(amp).sum(axis=1), 1e-6)
        wf = (r["wf_sum"] / np.maximum(r["wf_n"], 1)[:, None, None]).reshape(nu, ne, *r["wf_sum"].shape[1:])
        counts = r["wf_n"].reshape(nu, ne)

        order = np.argsort(unit, kind="stable")
        bounds = np.searchsorted(unit[order], np.arange(nu + 1))
        nn_cos = neighbour_cosines(dense_all[name], ch, off, tv, nbefore[name])
        rows, act = [], []
        for i, uid in enumerate(uids):
            pos = order[bounds[i]:bounds[i + 1]]
            used = tv[i] != 0
            for j, (e0, e1) in enumerate(ep):
                k = int((e_of[pos] == j).sum())
                act.append({"unit_id": uid, "epoch": names[j], "n_spikes": k, "rate_hz": k / max((e1 - e0) / fs, 1e-9)})
            if used.sum() < 2:
                continue
            for ja, jb in itertools.combinations(range(ne), 2):
                pa, pb = pos[e_of[pos] == ja], pos[e_of[pos] == jb]
                if pa.size < MIN_SPIKES or pb.size < MIN_SPIKES:
                    continue
                xa, xb = feat[pa][:, used], feat[pb][:, used]
                ma, mb = xa.mean(0), xb.mean(0)
                cos = float(ma @ mb / max(np.linalg.norm(ma) * np.linalg.norm(mb), 1e-9))
                auc = lda_auc(xa, xb, rng)
                big = pa if pa.size >= pb.size else pb
                half = big.size // 2
                auc_null = lda_auc(feat[big[:half]][:, used], feat[big[half:]][:, used], rng) if half >= MIN_SPIKES else np.nan
                wa, wb = wf[i, ja].ravel(), wf[i, jb].ravel()
                wcorr = float(np.corrcoef(wa, wb)[0, 1]) if counts[i, ja] and counts[i, jb] else np.nan
                rows.append({
                    "unit_id": uid, "epoch_a": names[ja], "epoch_b": names[jb], "n_a": int(pa.size), "n_b": int(pb.size),
                    "rate_a_hz": pa.size / ((ep[ja][1] - ep[ja][0]) / fs), "rate_b_hz": pb.size / ((ep[jb][1] - ep[jb][0]) / fs),
                    "footprint_cosine": cos, "neighbour_cosine": nn_cos[i], "waveform_corr": wcorr,
                    "amp_ratio": float(np.nanmedian(r["alpha"][pb]) / np.nanmedian(r["alpha"][pa])),
                    "location_shift_um": float(np.median(ycom[pb]) - np.median(ycom[pa])),
                    "auc": auc, "auc_null": auc_null, "verdict": verdict(cos, wcorr, nn_cos[i]),
                })
        folder = out_dir / name
        table = pd.DataFrame(rows)
        if len(table):
            table.insert(table.columns.get_loc("location_shift_um") + 1, "shift_vs_probe_um",
                         table["location_shift_um"] - table.groupby(["epoch_a", "epoch_b"])["location_shift_um"].transform("median"))
        table.to_csv(folder / "epoch_stability.tsv", sep="\t", index=False, float_format="%.4g")
        pd.DataFrame(act).to_csv(folder / "epoch_activity.tsv", sep="\t", index=False, float_format="%.4g")
        np.savez_compressed(folder / "epoch_templates.npz", unit_ids=np.asarray(uids), epochs=np.asarray(names),
                            channels=wch[name], waveforms=wf.astype(np.float32), counts=counts,
                            nbefore=np.int64(WAVEFORM_BEFORE), sampling_frequency=np.float64(fs))
        summary[name] = table["verdict"].value_counts().to_dict() if len(table) else {}
        log.info("%s epoch stability: %d unit-epoch comparisons; verdicts %s", name, len(table), summary[name])
    return summary
