"""Screen every unit for a second population of spikes that appears only in some epochs.

Built for e097, where neurons recruited only during gasping were absorbed by other units. KS4 #293 held
30 spikes of a much larger cell during gasping (87.8-92.1 s): troughs of 248-312 uV on ch 53 against
~64 uV for #293's own spikes, ~2.3x on ch 55. The review browser's footprint-mismatch flag missed all
30, because it compares the pattern across channels after scaling, and the intruder's pattern was close
to a scaled copy of #293's. This screen compares unscaled amplitudes with the unit's own baseline.

It only lists units and spikes to review; it never changes the sorting. Gasping can legitimately change
a neuron's amplitude, and a gasp burst makes collisions with neighbours more likely, so every candidate
needs a look in the review browser.

    python epoch_outliers.py C:/Users/lpascu/resp_analysis/outputs/e097/intact_window_v2 ^
        --time-offset-s 1740 --out C:/Users/lpascu/resp_analysis/outputs/e097/epoch_outliers/intact_window_v2_kilosort4

Spikes come from <output_dir>/<sorter>/alf (units.tsv ids), or from a Phy folder with --phy-folder
(spike_times in samples; Phy cluster ids mapped to units.tsv ids with cluster_si_unit_id.tsv, so a Phy
split shows as its original unit). Epochs, the scratch folder and the preprocessed binary come from the
run's config_used.yaml. All times are file time; --time-offset-s only adds raw-run times to the outputs.

Definitions (the output README points here):
- channels: the unit's top K (--channels, default 4) channels by template peak-to-peak, among those within
  RADIUS_UM of its largest channel. The template is the mean of up to MAX_TEMPLATE_SPIKES spikes evenly
  spaced through the unit's train, read from the preprocessed binary. Everything is deterministic per
  unit, so a unit's result doesn't change when a rule changes other units.
- amplitude a (uV, unscaled): the extremum of the template's sign within +-0.2 ms of the template's
  extremum time on that channel (npxsort.spikefeatures.spike_pass, as the review browser measures it).
  m = a x sign(template there), so a bigger spike in the template's direction has a bigger m.
- reference: the epoch named --reference (default baseline). A unit with fewer than MIN_REF_SPIKES
  spikes there uses its epoch with the most spikes (reference column).
- robust z per spike and channel: (m - median_ref) / max(1.4826 MAD_ref, 0.5 noise SD).
- outlier: |z| > --z (default 5) on any of the K channels. "high" if the most extreme z is positive (a
  bigger spike than the unit's own, as with #293), "low" if negative (smaller, or a different shape).
- per unit and epoch (unit_epochs.tsv):
    n_outliers, expected (reference outlier fraction x n_spikes), ratio, p_excess (Poisson upper tail)
    shift_z            median z on each channel, largest in absolute value: the whole distribution moved
                       (the neuron's own amplitude changing), unlike a separate population
    dbic               BIC(1 component) - BIC(2 components) of a Gaussian mixture on the epoch's z vectors
    new_*              the mixture component farther from the reference median: weight, number of spikes,
                       its mean z (largest absolute channel), separation (Mahalanobis distance between the
                       two component means under the pooled covariance) and ref_frac, the fraction of the
                       reference spikes the same mixture assigns to it
    second_mode        dbic > MIN_DBIC, separation >= MIN_SEPARATION, new_n >= MIN_OUTLIERS and ref_frac <=
                       new_weight / 5 (much rarer in the reference). The new component may be the epoch's
                       majority: then most of the unit's spikes in that epoch are another population.
    excess             n_outliers >= MIN_OUTLIERS, ratio >= MIN_RATIO and p_excess < MAX_P (or
                       < MAX_P_SECOND_MODE when the epoch has a second mode)
- burst: the BURST_WINDOW_S window (anywhere outside the reference epoch) holding the most outliers, tested
  like an epoch (excess against the reference fraction, and the same mixture on its spikes). It catches
  brief recruitment inside a long epoch, e.g. 6 spikes in 0.3 s of the 200-s recovery epoch.
- candidate: a unit with excess outliers in an epoch other than the reference, or in its burst window.
  tier "separate population": one of those windows also has a second mode, and outlier_coherence >=
  MIN_COHERENCE; else "excess outliers".
  evidence = the largest -log10 p_excess of those windows (capped at 300); separation = the largest
  separation among the windows with a second mode. Ranked by tier, then separation (separate population;
  the cleanest splits first) or evidence (excess outliers).
- shared_with: other candidates whose flagged outliers coincide with this unit's (within SHARED_MS; at least
  MIN_SHARED and SHARED_MIN_FRACTION of the smaller set), as unit:count. The same intruding neuron often
  lands in several units (#293 and #292 shared all 30 events).
- outlier_coherence: norm of the flagged outliers' median z / their median distance from it (high = the
  outliers sit together, as another neuron's spikes would; low = scattered, as collisions would).
- outlier_spikes.tsv lists every outlier of a candidate unit outside the reference epoch (in_flagged_window
  marks those inside the flagged epochs or burst window) with its z and amplitude per channel and the
  review browser's footprint-mismatch measures (rel, sig, footprint_mismatch) for comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import poisson
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spikeinterface.core as si  # noqa: E402

from npxsort.review import FOOTPRINT_MIN_SIGMA, FOOTPRINT_MISMATCH  # noqa: E402
from npxsort.spikefeatures import (AMP_HALF_WINDOW, WAVEFORM_AFTER, WAVEFORM_BEFORE, noise_levels,  # noqa: E402
                                   spike_pass)
from npxsort.stability import binary_memmap  # noqa: E402

log = logging.getLogger("epoch_outliers")

MAX_TEMPLATE_SPIKES = 300    # spikes averaged per unit for its template (channel choice only)
RADIUS_UM = 100.0            # channels within this distance of the largest one (the analyzer's sparsity radius)
MIN_REF_SPIKES = 30          # in the reference epoch, else the unit's busiest epoch is its reference
MIN_EPOCH_SPIKES = 10        # an epoch with fewer spikes gets counts but no mixture
MAD_FLOOR_NOISE = 0.5        # robust SD >= this x the channel's noise SD (amplitudes are quantised at ~3 uV)
MIN_OUTLIERS = 5
MIN_RATIO = 3.0              # outliers / expected
MAX_P = 1e-6                 # Poisson upper tail of the outlier count ...
MAX_P_SECOND_MODE = 1e-4     # ... or this, when the same window has a second mode (e097 KS4 #520: 17 outliers
                             # vs 5.4 expected in gasping, p 5e-5, separation 11)
MIN_DBIC = 10.0
MIN_SEPARATION = 3.0
MAX_MIXTURE_SPIKES = 5000    # seeded random subset per epoch for the mixture fit
BURST_WINDOW_S = 10.0
SCORE_CAP = 300.0
MIN_COHERENCE = 1.5          # tier "separate population" also needs outliers that sit together (e097: scattered
                             # collision-like outliers 0.6-0.8, separate populations 1.8-10)
SHARED_MS = 0.5              # flagged outliers of two candidates this close count as one event ...
MIN_SHARED = 3               # ... listed in shared_with from this many on, and from this fraction of the smaller
SHARED_MIN_FRACTION = 0.25   # set of flagged outliers (gasp volleys give most units a few coincidences)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="npxsort run output_dir (config_used.yaml, <sorter>/units.tsv, review/)")
    p.add_argument("--sorter", default="kilosort4")
    p.add_argument("--phy-folder", default=None, help="read spikes from this Phy folder instead of <sorter>/alf")
    p.add_argument("--units-tsv", default=None, help="units.tsv for labels (default <output_dir>/<sorter>/units.tsv)")
    p.add_argument("--review-tsv", default=None, help="review verdicts (default <output_dir>/review/<sorter>_unit_review.tsv)")
    p.add_argument("--reference", default="baseline", help="epoch every other epoch is compared with")
    p.add_argument("--channels", type=int, default=4, help="top template channels per unit")
    p.add_argument("--z", type=float, default=5.0, help="|robust z| above this on any channel = outlier")
    p.add_argument("--figures", type=int, default=30, help="figures for this many top candidates")
    p.add_argument("--extra-figure-units", default="", help="comma list of unit ids that get a figure anyway")
    p.add_argument("--time-offset-s", type=float, default=0.0, help="added to file time for raw-run time columns")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


class Run:
    """Spikes, epochs and the preprocessed binary of one sorted window."""

    def __init__(self, a):
        self.out = Path(a.output_dir)
        self.cfg = yaml.safe_load((self.out / "config_used.yaml").read_text())
        folder = Path(self.cfg["paths"]["scratch_dir"]) / "preprocessed"
        rec = si.load(folder)
        self.fs = float(rec.get_sampling_frequency())
        self.n_samples = int(rec.get_num_samples())
        self.n_channels = int(rec.get_num_channels())
        self.gain = rec.get_channel_gains().astype(np.float32)
        self.offset = rec.get_channel_offsets().astype(np.float32)
        self.locations = rec.get_channel_locations()
        self.traces = binary_memmap(folder, self.n_channels)
        self.binary = folder

        if a.phy_folder:
            phy = Path(a.phy_folder)
            sample = np.load(phy / "spike_times.npy").ravel().astype(np.int64)
            clusters = np.load(phy / "spike_clusters.npy").ravel().astype(np.int64)
            lut_file = next((phy / f for f in ("cluster_si_unit_id.tsv", "cluster_si_unit_ids.tsv") if (phy / f).exists()), None)
            if lut_file is not None:
                m = pd.read_csv(lut_file, sep="\t")
                lut = dict(zip(m["cluster_id"].astype(int), m["si_unit_id"].astype(int)))
                missing = set(np.unique(clusters)) - set(lut)
                if missing:
                    raise ValueError(f"{lut_file} has no unit id for Phy clusters {sorted(missing)[:10]}")
                ids = np.vectorize(lut.get, otypes=[np.int64])(clusters)
            else:
                ids = clusters
            self.source = str(phy)
        else:
            alf = self.out / a.sorter / "alf"
            sample = np.round(np.load(alf / "spike_times.npy").ravel() * self.fs).astype(np.int64)
            ids = np.load(alf / "spike_clusters.npy").ravel().astype(np.int64)
            self.source = str(alf)
        o = np.argsort(sample, kind="stable")
        sample, ids = sample[o], ids[o]
        self.unit_ids = np.unique(ids)
        self.unit = np.searchsorted(self.unit_ids, ids)
        self.sample = sample

        units_tsv = Path(a.units_tsv) if a.units_tsv else self.out / a.sorter / "units.tsv"
        units = pd.read_csv(units_tsv, sep="\t").set_index("unit_id")
        self.label = units["label"].reindex(self.unit_ids).fillna("").to_numpy()
        self.extracted_as = (units["extracted_as"].reindex(self.unit_ids).fillna("").astype(str).to_numpy()
                             if "extracted_as" in units else np.full(self.unit_ids.size, ""))
        review = Path(a.review_tsv) if a.review_tsv else self.out / "review" / f"{a.sorter}_unit_review.tsv"
        verdicts = pd.read_csv(review, sep="\t").set_index("unit_id")["verdict"] if review.exists() else pd.Series(dtype=object)
        self.verdict = verdicts.reindex(self.unit_ids).fillna("").to_numpy()

        self.epochs = self.cfg.get("epochs") or []
        if a.reference not in [e["name"] for e in self.epochs]:
            raise ValueError(f"No epoch named {a.reference!r} in {self.out / 'config_used.yaml'}")
        self.epoch_names = [e["name"] for e in self.epochs]
        self.epoch_of = np.full(sample.size, -1, dtype=np.int64)
        for j, e in enumerate(self.epochs):
            s0, s1 = int(round(e["start_s"] * self.fs)), int(round(e["end_s"] * self.fs))
            self.epoch_of[(sample >= s0) & (sample < s1)] = j


def templates(run: Run) -> np.ndarray:
    """Mean waveform (uV) on all channels over up to MAX_TEMPLATE_SPIKES spikes per unit, evenly spaced
    through its spike train (so a unit's template doesn't change when other units do)."""
    n, nch, nu = run.n_samples, run.n_channels, run.unit_ids.size
    s, u = run.sample, run.unit
    sel = np.flatnonzero((s >= WAVEFORM_BEFORE) & (s + WAVEFORM_AFTER <= n))
    sel = sel[np.argsort(u[sel], kind="stable")]
    bounds = np.searchsorted(u[sel], np.arange(nu + 1))
    chosen = np.sort(np.concatenate([sel[b0 + np.unique(np.arange(m) * (b1 - b0) // max(m, 1))]
                                     for b0, b1 in zip(bounds[:-1], bounds[1:])
                                     for m in [min(MAX_TEMPLATE_SPIKES, b1 - b0)]]))
    samp, un = s[chosen], u[chosen]
    wwin = np.arange(-WAVEFORM_BEFORE, WAVEFORM_AFTER)
    sums = np.zeros((nu, wwin.size, nch), dtype=np.float64)
    cnt = np.zeros(nu, dtype=np.int64)
    step, batch = int(run.fs), 256
    for c0 in range(0, n, step):
        i0, i1 = np.searchsorted(samp, [c0, c0 + step])
        if i0 == i1:
            continue
        a0, a1 = max(0, c0 - WAVEFORM_BEFORE), min(n, c0 + step + WAVEFORM_AFTER)
        block = np.asarray(run.traces[a0:a1], dtype=np.float32)
        for k0 in range(i0, i1, batch):
            k1 = min(i1, k0 + batch)
            uu = un[k0:k1]
            o = np.argsort(uu, kind="stable")
            uu = uu[o]
            wf = block[samp[k0:k1][o][:, None] - a0 + wwin[None, :]]
            starts = np.r_[0, np.flatnonzero(np.diff(uu)) + 1]
            sums[uu[starts]] += np.add.reduceat(wf, starts, axis=0)
            cnt[uu[starts]] += np.diff(np.r_[starts, uu.size])
    return (sums / np.maximum(cnt, 1)[:, None, None] * run.gain[None, None, :]).astype(np.float32)


def channel_setup(dense: np.ndarray, k: int, locations: np.ndarray):
    """spike_pass setup for the top-k channels by peak-to-peak within RADIUS_UM of the largest one (no
    minimum fraction, unlike the footprint; the radius keeps distant noise-level channels out)."""
    nbefore = WAVEFORM_BEFORE
    ptp = dense.max(axis=1) - dense.min(axis=1)
    dist = np.linalg.norm(locations[ptp.argmax(axis=1)][:, None, :] - locations[None, :, :], axis=2)
    ch = np.argsort(-np.where(dist <= RADIUS_UM, ptp, -np.inf), axis=1)[:, :k]
    w = np.take_along_axis(dense, ch[:, None, :], axis=2)            # (units, samples, k)
    kk = np.abs(w).argmax(axis=1)                                      # (units, k)
    tv = np.take_along_axis(w, kk[:, None, :], axis=1)[:, 0, :]
    tv = np.where(tv == 0, 1e-3, tv).astype(np.float32)
    off = (kk - nbefore).astype(np.int64)
    return ch.astype(np.int64), off, tv, np.zeros(dense.shape[0], dtype=np.float32)


def measure(run: Run, k: int, cache: Path):
    """Templates, top-k channels and every spike's amplitudes on them (cached: the two passes over the
    binary take most of the run time). Returns ch, tv, templates on those channels, noise, spike_pass result."""
    key = (f"{run.source}|{run.binary}|{k}|{run.sample.size}|{int(run.sample.sum())}|{MAX_TEMPLATE_SPIKES}|"
           f"{AMP_HALF_WINDOW}|{RADIUS_UM}|v2")
    if cache.exists():
        d = np.load(cache)
        if str(d["key"]) == key and np.array_equal(d["unit_ids"], run.unit_ids):
            log.info("amplitudes from cache %s", cache)
            return d["ch"], d["tv"], d["dense"], d["noise"], {f: d[f] for f in ("amp", "alpha", "rel", "sig")}
    dense = templates(run)
    setup = channel_setup(dense, k, run.locations)
    ch, _, tv, _ = setup
    noise = noise_levels(run.traces, run.gain, run.n_samples, run.fs)
    log.info("templates done; measuring %d spikes on %d channels per unit", run.sample.size, k)
    res = spike_pass(run.traces, run.gain, run.offset, noise, run.n_samples, run.fs, {"s": setup},
                     {"s": (run.sample, run.unit)})["s"]
    dense_k = np.take_along_axis(dense, ch[:, None, :], axis=2)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=key, unit_ids=run.unit_ids, ch=ch, tv=tv, dense=dense_k, noise=noise,
             **{f: res[f] for f in ("amp", "alpha", "rel", "sig")})
    return ch, tv, dense_k, noise, res


def poisson_p(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """P(X >= k) for X ~ Poisson(mu); mu floored so a reference without outliers still gives a finite p."""
    return poisson.sf(np.asarray(k) - 1, np.maximum(mu, 0.5))


def mixture(x: np.ndarray, outlier: np.ndarray, xref: np.ndarray) -> dict:
    """2- vs 1-component Gaussian mixture on one window's z vectors; the component farther from the
    reference median (z = 0) is the "new" one. With >= MIN_OUTLIERS outliers the two components start at
    the medians of the inliers and the outliers, else at k-means (fixed seed): deterministic per unit."""
    xs, os_ = x, outlier
    if x.shape[0] > MAX_MIXTURE_SPIKES:   # evenly spaced subset for the fit
        k = np.linspace(0, x.shape[0] - 1, MAX_MIXTURE_SPIKES).astype(int)
        xs, os_ = x[k], outlier[k]
    g1 = GaussianMixture(1, covariance_type="full", reg_covar=1e-3).fit(xs)
    if MIN_OUTLIERS <= os_.sum() < os_.size:
        init = np.stack([np.median(xs[~os_], axis=0), np.median(xs[os_], axis=0)])
        g2 = GaussianMixture(2, covariance_type="full", reg_covar=1e-3, means_init=init, random_state=0).fit(xs)
    else:
        g2 = GaussianMixture(2, covariance_type="full", reg_covar=1e-3, n_init=3, random_state=0).fit(xs)
    j = int(np.argmax(np.linalg.norm(g2.means_, axis=1)))
    pooled = (g2.weights_[:, None, None] * g2.covariances_).sum(axis=0)
    d = g2.means_[j] - g2.means_[1 - j]
    mz = g2.means_[j]
    new_n = int((g2.predict(x) == j).sum())
    return {"dbic": float(g1.bic(xs) - g2.bic(xs)), "new_weight": new_n / x.shape[0],
            "new_n": new_n, "new_mean_z": float(mz[np.argmax(np.abs(mz))]),
            "separation": float(np.sqrt(d @ np.linalg.solve(pooled, d))),
            "ref_frac": float((g2.predict(xref) == j).mean()) if xref.shape[0] else np.nan}


def second_mode(mix: dict) -> bool:
    """A separate component: clearly better than one Gaussian, well apart, big enough, and much rarer
    in the reference (it may be the epoch's majority: the unit's spikes there are mostly another population)."""
    return bool(mix["dbic"] > MIN_DBIC and mix["separation"] >= MIN_SEPARATION and mix["new_n"] >= MIN_OUTLIERS
                and mix["ref_frac"] <= mix["new_weight"] / 5)


def best_window(t: np.ndarray, w: float) -> tuple[float, int]:
    """Start time and count of the length-w window holding the most of the sorted times t."""
    if t.size == 0:
        return np.nan, 0
    hi = np.searchsorted(t, t + w, side="left")
    k = hi - np.arange(t.size)
    i = int(np.argmax(k))
    return float(t[i]), int(k[i])


def main():
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Run(a)
    fs, nu, K = run.fs, run.unit_ids.size, a.channels
    log.info("%d units, %d spikes from %s; binary %s", nu, run.sample.size, run.source, run.binary)

    ch, tv, dense, noise, res = measure(run, K, out / "cache" / "amplitudes.npz")
    amp = res["amp"]                                    # (spikes, K) signed uV
    m = amp * np.sign(tv[run.unit])                     # polarity-aligned
    flagged_mismatch = (res["rel"] > FOOTPRINT_MISMATCH) & (res["sig"] > FOOTPRINT_MIN_SIGMA)

    ne = len(run.epochs)
    ref_j = run.epoch_names.index(a.reference)
    order = np.argsort(run.unit, kind="stable")
    bounds = np.searchsorted(run.unit[order], np.arange(nu + 1))
    z = np.full(m.shape, np.nan, dtype=np.float32)
    rows, cand_rows, spike_rows, fig_info, flag_times = [], [], [], {}, {}
    dur = np.array([e["end_s"] - e["start_s"] for e in run.epochs], dtype=float)
    for i in range(nu):
        pos = order[bounds[i]:bounds[i + 1]]
        ep = run.epoch_of[pos]
        counts = np.bincount(ep[ep >= 0], minlength=ne)
        rj = ref_j if counts[ref_j] >= MIN_REF_SPIKES else (int(np.argmax(counts)) if counts.max() >= MIN_REF_SPIKES else -1)
        base = dict(unit_id=int(run.unit_ids[i]), label=run.label[i], verdict=run.verdict[i],
                    extracted_as=run.extracted_as[i], reference=run.epoch_names[rj] if rj >= 0 else "",
                    channels=",".join(str(int(c)) for c in ch[i]))
        if rj < 0:
            for j in range(ne):
                rows.append({**base, "epoch": run.epoch_names[j], "n_spikes": int(counts[j]), "rate_hz": counts[j] / dur[j]})
            continue
        mi = m[pos]
        ref = mi[ep == rj]
        med = np.median(ref, axis=0)
        sd = np.maximum(1.4826 * np.median(np.abs(ref - med), axis=0), MAD_FLOOR_NOISE * noise[ch[i]])
        zi = (mi - med) / sd
        z[pos] = zi
        ext = np.take_along_axis(zi, np.abs(zi).argmax(axis=1)[:, None], axis=1)[:, 0]
        out_mask = np.abs(ext) > a.z
        ref_frac = float(out_mask[ep == rj].mean())
        zref = zi[ep == rj]

        unit_rows, flagged_epochs = [], []
        for j in range(ne):
            sel = ep == j
            n_j = int(sel.sum())
            r = {**base, "epoch": run.epoch_names[j], "n_spikes": n_j, "rate_hz": n_j / dur[j]}
            if n_j:
                zj = zi[sel]
                k = int(out_mask[sel].sum())
                mu = ref_frac * n_j
                shift = np.median(zj, axis=0)
                p = float(poisson_p(k, mu)) if j != rj else np.nan
                r.update(n_outliers=k, n_high=int((out_mask[sel] & (ext[sel] > 0)).sum()),
                         n_low=int((out_mask[sel] & (ext[sel] < 0)).sum()), frac_outliers=k / n_j,
                         expected=mu, ratio=k / max(mu, 0.5), p_excess=p,
                         neglog10_p=min(SCORE_CAP, -np.log10(max(p, 1e-300))) if j != rj else np.nan,
                         shift_z=float(shift[np.argmax(np.abs(shift))]),
                         median_amp_uv_ch1=float(np.median(amp[pos][sel][:, 0])))
                if n_j >= MIN_EPOCH_SPIKES:
                    mix = mixture(zj, out_mask[sel], zref if j != rj else zj[:0])
                    r.update(mix, second_mode=j != rj and second_mode(mix))
                else:
                    r["second_mode"] = False
                more = j != rj and k >= MIN_OUTLIERS and r["ratio"] >= MIN_RATIO
                r["excess"] = bool(more and (p < MAX_P or (r["second_mode"] and p < MAX_P_SECOND_MODE)))
                if r["excess"]:
                    flagged_epochs.append(j)
            unit_rows.append(r)
        rows.extend(unit_rows)

        # densest BURST_WINDOW_S of outliers anywhere, outside the reference epoch
        tpos = run.sample[pos] / fs
        tout = tpos[out_mask & (ep != rj)]
        w0, wk = best_window(tout, BURST_WINDOW_S)
        in_w = (tpos >= w0) & (tpos < w0 + BURST_WINDOW_S) & (ep != rj) if wk else np.zeros(pos.size, dtype=bool)
        mu_w = ref_frac * int(in_w.sum())
        p_w = float(poisson_p(wk, mu_w)) if wk else 1.0
        more = wk >= MIN_OUTLIERS and wk / max(mu_w, 0.5) >= MIN_RATIO and p_w < MAX_P_SECOND_MODE
        burst_mix = mixture(zi[in_w], out_mask[in_w], zref) if more and in_w.sum() >= MIN_EPOCH_SPIKES else None
        burst = more and (p_w < MAX_P or (burst_mix is not None and second_mode(burst_mix)))
        if not flagged_epochs and not burst:
            continue
        if not burst:
            burst_mix = None

        lps = [r["neglog10_p"] for r in unit_rows if r.get("excess")] + ([min(SCORE_CAP, -np.log10(max(p_w, 1e-300)))] if burst else [])
        seps = [unit_rows[j]["separation"] for j in flagged_epochs if unit_rows[j].get("second_mode")]
        if burst_mix is not None and second_mode(burst_mix):
            seps.append(burst_mix["separation"])
        second = bool(seps)
        top = max((unit_rows[j] for j in flagged_epochs), key=lambda r: r.get("neglog10_p", 0.0), default=None)
        win = in_w.copy() if burst else np.zeros(pos.size, dtype=bool)
        for j in flagged_epochs:
            win |= ep == j
        sel_out = out_mask & win
        zo = zi[sel_out]
        centre = np.median(zo, axis=0) if zo.size else np.zeros(K)
        spread = float(np.median(np.linalg.norm(zo - centre, axis=1))) if zo.size else np.nan
        coherence = float(np.linalg.norm(centre) / max(spread, 1.0)) if zo.size else np.nan
        cand_rows.append({
            **base, "tier": "separate population" if second and coherence >= MIN_COHERENCE else "excess outliers",
            "evidence": max(lps), "separation": max(seps) if seps else np.nan,
            "flagged_epochs": ",".join(run.epoch_names[j] for j in flagged_epochs),
            "outlier_coherence": coherence, "outlier_centre_z": float(np.linalg.norm(centre)),
            "top_epoch": top["epoch"] if top else "",
            "top_n_outliers": top.get("n_outliers") if top else None, "top_expected": top.get("expected") if top else None,
            "top_n_spikes": top.get("n_spikes") if top else None, "top_shift_z": top.get("shift_z") if top else None,
            "top_new_weight": top.get("new_weight") if top else None, "top_separation": top.get("separation") if top else None,
            "burst_start_s": w0 if burst else np.nan, "burst_n_outliers": wk if burst else 0,
            "burst_expected": mu_w if burst else np.nan, "burst_p": p_w if burst else np.nan,
            "burst_separation": burst_mix["separation"] if burst_mix else np.nan,
            "burst_second_mode": bool(burst_mix and second_mode(burst_mix)),
            "n_outliers_flagged": int(sel_out.sum()), "n_high": int((sel_out & (ext > 0)).sum()),
            "n_low": int((sel_out & (ext < 0)).sum()),
            "n_footprint_mismatch": int((sel_out & flagged_mismatch[pos]).sum()),
            "outlier_rel_median": float(np.nanmedian(res["rel"][pos][sel_out])) if sel_out.any() else np.nan,
            "ref_rel_median": float(np.nanmedian(res["rel"][pos][ep == rj])),
            "ref_outlier_frac": ref_frac, "n_spikes": int(pos.size),
        })
        for q in np.flatnonzero(out_mask & (ep != rj)):
            sp = pos[q]
            row = {"unit_id": int(run.unit_ids[i]), "time_s": round(run.sample[sp] / fs, 5),
                   "time_raw_s": round(run.sample[sp] / fs + a.time_offset_s, 5),
                   "epoch": run.epoch_names[ep[q]] if ep[q] >= 0 else "", "in_flagged_window": bool(win[q]),
                   "direction": "high" if ext[q] > 0 else "low", "max_abs_z": round(float(abs(ext[q])), 2)}
            for c in range(K):
                row[f"ch{c + 1}"] = int(ch[i, c])
                row[f"amp_uv_ch{c + 1}"] = round(float(amp[sp, c]), 1)
                row[f"z_ch{c + 1}"] = round(float(zi[q, c]), 2)
            row.update(rel=round(float(res["rel"][sp]), 3), sig=round(float(res["sig"][sp]), 1),
                       footprint_mismatch=bool(flagged_mismatch[sp]))
            spike_rows.append(row)
        fig_info[int(run.unit_ids[i])] = (i, pos, rj, out_mask, sel_out, med, sd)
        flag_times[int(run.unit_ids[i])] = np.sort(run.sample[pos[sel_out]])

    # the same intruding events absorbed by several units (e.g. #293 and #292 shared all 30)
    tol = int(round(SHARED_MS / 1000 * fs))
    for r in cand_rows:
        x = flag_times[r["unit_id"]]
        shared = []
        for v, y in flag_times.items():
            if v == r["unit_id"] or not x.size or not y.size:
                continue
            j = np.searchsorted(y, x)
            d = np.minimum(np.abs(y[np.minimum(j, y.size - 1)] - x), np.abs(y[np.maximum(j - 1, 0)] - x))
            k = int((d <= tol).sum())
            if k >= MIN_SHARED and k >= SHARED_MIN_FRACTION * min(x.size, y.size):
                shared.append((k, v))
        r["shared_with"] = ",".join(f"{v}:{k}" for k, v in sorted(shared, reverse=True))

    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("epoch") + 1, "start_s", table["epoch"].map({e["name"]: e["start_s"] for e in run.epochs}))
    table.to_csv(out / "unit_epochs.tsv", sep="\t", index=False, float_format="%.4g")
    cands = pd.DataFrame(cand_rows)
    if len(cands):
        sep = cands["tier"] == "separate population"
        cands = (cands.assign(_t=~sep, _k=np.where(sep, -cands["separation"], -cands["evidence"]))
                 .sort_values(["_t", "_k"]).drop(columns=["_t", "_k"]).reset_index(drop=True))
        cands.insert(0, "rank", np.arange(1, len(cands) + 1))
        if a.time_offset_s:
            cands.insert(cands.columns.get_loc("burst_start_s") + 1, "burst_start_raw_s", cands["burst_start_s"] + a.time_offset_s)
    cands.to_csv(out / "candidates.tsv", sep="\t", index=False, float_format="%.4g")
    spikes = pd.DataFrame(spike_rows)
    spikes.to_csv(out / "outlier_spikes.tsv", sep="\t", index=False)
    log.info("%d of %d units are candidates; %d outlier spikes listed", len(cands), nu, len(spikes))

    extra = [int(x) for x in a.extra_figure_units.split(",") if x.strip()]
    fig_units = list(cands["unit_id"][:a.figures]) if len(cands) else []
    fig_units += [u for u in extra if u not in fig_units]
    figures(run, a, out / "figures", fig_units, cands, fig_info, amp, z, ch, tv, dense, noise)
    summary = write_readme(run, a, out, table, cands, spikes, nu)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


EPOCH_COLOURS = ["#4c78a8", "#f58518", "#e45756", "#54a24b", "#b279a2", "#9d755d", "#72b7b2", "#eeca3b"]


def figures(run, a, folder, fig_units, cands, fig_info, amp, z, ch, tv, dense, noise):
    """Per unit: amplitude vs time on its K channels coloured by epoch (outliers ringed), the reference
    band, an amplitude scatter of its two top channels, and the mean waveform of the flagged outliers
    against a sample of the unit's reference spikes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fs, K = run.fs, a.channels
    order = np.argsort(run.unit, kind="stable")
    bounds = np.searchsorted(run.unit[order], np.arange(run.unit_ids.size + 1))
    rank_of = dict(zip(cands["unit_id"], cands["rank"])) if len(cands) else {}
    rng = np.random.default_rng(a.seed)
    wwin = np.arange(-WAVEFORM_BEFORE, WAVEFORM_AFTER)
    for u in fig_units:
        i = int(np.searchsorted(run.unit_ids, u))
        if i >= run.unit_ids.size or run.unit_ids[i] != u:
            log.warning("unit %s not in this sorting; no figure", u)
            continue
        pos = order[bounds[i]:bounds[i + 1]]
        if u in fig_info:
            _, _, rj, out_mask, sel_out, med, sd = fig_info[u]
        else:
            ep = run.epoch_of[pos]
            zi = z[pos]
            ext = np.take_along_axis(zi, np.nan_to_num(np.abs(zi)).argmax(axis=1)[:, None], axis=1)[:, 0]
            out_mask = np.abs(ext) > a.z
            sel_out = np.zeros(pos.size, dtype=bool)
            rj = run.epoch_names.index(a.reference)
            ref = amp[pos][ep == rj] * np.sign(tv[i])
            med = np.median(ref, axis=0) if len(ref) else np.zeros(K)
            sd = np.maximum(1.4826 * np.median(np.abs(ref - med), axis=0), MAD_FLOOR_NOISE * noise[ch[i]]) if len(ref) else np.ones(K)
        t = run.sample[pos] / fs
        ep = run.epoch_of[pos]
        ai = amp[pos]
        fig = plt.figure(figsize=(16, 2.1 * K + 1.2))
        gs = fig.add_gridspec(K, 3, width_ratios=[4.2, 1.4, 1.4], hspace=0.35, wspace=0.28)
        sgn = np.sign(tv[i])
        for c in range(K):
            ax = fig.add_subplot(gs[c, 0])
            for e in run.epochs:
                ax.axvspan(e["start_s"], e["end_s"], color="0.93", zorder=0)
            outside = ep < 0
            ax.scatter(t[outside], ai[outside, c], s=2, c="0.6", lw=0, rasterized=True)
            for j, name in enumerate(run.epoch_names):
                s = ep == j
                ax.scatter(t[s], ai[s, c], s=2, c=EPOCH_COLOURS[j % len(EPOCH_COLOURS)], lw=0, rasterized=True,
                           label=f"{name} ({s.sum()})" if c == 0 else None)
            ax.scatter(t[out_mask & ~sel_out], ai[out_mask & ~sel_out, c], s=14, facecolors="none", edgecolors="0.35", lw=0.5)
            ax.scatter(t[sel_out], ai[sel_out, c], s=22, facecolors="none", edgecolors="k", lw=0.8,
                       label=f"outliers in flagged epochs ({sel_out.sum()})" if c == 0 else None)
            lo, hi = sgn[c] * (med[c] - a.z * sd[c]), sgn[c] * (med[c] + a.z * sd[c])
            ax.axhline(sgn[c] * med[c], color="k", lw=0.6)
            ax.axhspan(min(lo, hi), max(lo, hi), color="k", alpha=0.06, zorder=0)
            ax.set_ylabel(f"ch {ch[i, c]} (uV)")
            ax.set_xlim(0, run.n_samples / fs)
            if c == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=4, markerscale=3, frameon=False, bbox_to_anchor=(0, 1.32))
            if c == K - 1:
                ax.set_xlabel("file time (s)" + (f"; raw = file + {a.time_offset_s:g} s" if a.time_offset_s else ""))
        # amplitude scatter, top two channels
        ax = fig.add_subplot(gs[:max(1, K // 2), 1])
        for j in range(len(run.epochs)):
            s = ep == j
            ax.scatter(ai[s, 0], ai[s, 1], s=2, c=EPOCH_COLOURS[j % len(EPOCH_COLOURS)], lw=0, rasterized=True)
        ax.scatter(ai[sel_out, 0], ai[sel_out, 1], s=18, facecolors="none", edgecolors="k", lw=0.7)
        ax.set_xlabel(f"ch {ch[i, 0]} (uV)")
        ax.set_ylabel(f"ch {ch[i, 1]} (uV)")
        if K > 2:
            ax = fig.add_subplot(gs[max(1, K // 2):, 1])
            for j in range(len(run.epochs)):
                s = ep == j
                ax.scatter(ai[s, 0], ai[s, 2], s=2, c=EPOCH_COLOURS[j % len(EPOCH_COLOURS)], lw=0, rasterized=True)
            ax.scatter(ai[sel_out, 0], ai[sel_out, 2], s=18, facecolors="none", edgecolors="k", lw=0.7)
            ax.set_xlabel(f"ch {ch[i, 0]} (uV)")
            ax.set_ylabel(f"ch {ch[i, 2]} (uV)")
        # mean waveforms: flagged outliers vs reference sample
        refpos = pos[ep == rj]
        refpick = np.sort(rng.choice(refpos, min(200, refpos.size), replace=False)) if refpos.size else refpos
        outpick = pos[sel_out][:200]
        groups = []
        for p_, col, lab in ((refpick, "0.4", f"{run.epoch_names[rj]} sample"), (outpick, "k", "flagged outliers")):
            s = run.sample[p_]
            s = s[(s >= WAVEFORM_BEFORE) & (s + WAVEFORM_AFTER <= run.n_samples)]
            w = (np.stack([run.traces[t - WAVEFORM_BEFORE:t + WAVEFORM_AFTER][:, ch[i]] for t in s]).astype(np.float32)
                 * run.gain[ch[i]] if s.size else np.zeros((0, wwin.size, K), dtype=np.float32))
            groups.append((w, col, f"{lab} ({s.size})"))
        tms = wwin / fs * 1000
        for c in range(K):
            ax = fig.add_subplot(gs[c, 2])
            cc = int(ch[i, c])
            for w, col, lab in groups:
                if len(w):
                    for row in w[:40, :, c]:
                        ax.plot(tms, row, color=col, lw=0.3, alpha=0.25)
                    ax.plot(tms, w[:, :, c].mean(0), color=col if col == "k" else "tab:blue", lw=1.5, label=lab if c == 0 else None)
            ax.plot(tms, dense[i, :, c], color="tab:red", lw=0.8, ls="--", label="template" if c == 0 else None)
            ax.set_title(f"ch {cc}", fontsize=8, pad=2)
            if c == 0:
                ax.legend(fontsize=6, frameon=False, loc="lower right")
            if c == K - 1:
                ax.set_xlabel("ms")
        rk = rank_of.get(u)
        cand = cands[cands["unit_id"] == u].iloc[0] if rk else None
        title = f"{a.sorter} #{u}  ({run.label[i]}{', review: ' + run.verdict[i] if run.verdict[i] else ''})"
        if cand is not None:
            title += (f"   rank {rk} ({cand['tier']}{', separation ' + format(cand['separation'], '.1f') if cand['separation'] == cand['separation'] else ''}),"
                      f" evidence {cand['evidence']:.0f}; flagged: {cand['flagged_epochs'] or '-'}"
                      f"{'; burst at ' + format(cand['burst_start_s'], '.1f') + ' s' if cand['burst_n_outliers'] else ''}"
                      f"; {cand['n_outliers_flagged']} outliers ({cand['n_high']} high, {cand['n_low']} low)")
        else:
            title += "   not a candidate"
        fig.suptitle(title + f"\nreference {run.epoch_names[rj]}: grey band = median +- {a.z:g} robust SD; rings = outliers",
                     fontsize=9, y=0.995)
        name = f"rank{rk:03d}_unit{u}.png" if rk else f"unit{u}.png"
        fig.savefig(folder / name, dpi=110, bbox_inches="tight")
        plt.close(fig)
    log.info("%d figures in %s", len(fig_units), folder)


def write_readme(run, a, out, table, cands, spikes, nu) -> dict:
    n_c = len(cands)
    single = run.verdict == "single"
    single_ids = set(run.unit_ids[single].tolist())
    single_c = sorted(set(cands["unit_id"]) & single_ids) if n_c else []
    by_label = cands["label"].value_counts().to_dict() if n_c else {}
    tiers = ("separate population", "excess outliers")
    by_tier = {t: int((cands["tier"] == t).sum()) if n_c else 0 for t in tiers}
    single_by_tier = {t: sorted(int(u) for u in cands["unit_id"][cands["tier"] == t] if u in single_ids) if n_c else []
                      for t in tiers}
    summary = {
        "source": run.source, "binary": str(run.binary), "n_units": int(nu), "n_candidates": int(n_c),
        "candidates_by_tier": by_tier, "candidates_by_label": {str(k): int(v) for k, v in by_label.items()},
        "n_reviewed_single": int(single.sum()), "reviewed_single_candidates": [int(x) for x in single_c],
        "reviewed_single_by_tier": single_by_tier,
        "n_outlier_spikes_listed": int(len(spikes)),
        "settings": {"channels": a.channels, "z": a.z, "reference": a.reference, "min_outliers": MIN_OUTLIERS,
                     "min_ratio": MIN_RATIO, "max_p": MAX_P, "min_dbic": MIN_DBIC, "min_separation": MIN_SEPARATION,
                     "burst_window_s": BURST_WINDOW_S, "amp_half_window_samples": AMP_HALF_WINDOW},
    }
    lines = [f"# Epoch outlier screen: {a.sorter}", "",
             f"Spikes: `{run.source}`. Binary: `{run.binary}`. Command:", "",
             "```", "python epoch_outliers.py " + " ".join(sys.argv[1:]), "```", "",
             f"{n_c} of {nu} units are candidates: {by_tier['separate population']} separate population, "
             f"{by_tier['excess outliers']} excess outliers (by label: {by_label}).",
             f"Reviewed `single`: {len(single_c)} of {int(single.sum())} are candidates; separate population "
             f"{single_by_tier['separate population']}, excess outliers {single_by_tier['excess outliers']}.", "",
             "Files: `candidates.tsv` (ranked), `unit_epochs.tsv` (every unit x epoch), `outlier_spikes.tsv` "
             "(outliers of candidate units), `figures/` (top candidates). Definitions: see the docstring of "
             "`sorting_v2/epoch_outliers.py`; settings in `summary.json`.", "",
             "Candidates are for review only; nothing here changes the sorting.", ""]
    if n_c:
        cols = ["rank", "unit_id", "label", "verdict", "tier", "separation", "evidence", "flagged_epochs", "top_epoch",
                "top_n_outliers", "top_expected", "n_high", "n_low", "burst_start_s", "burst_n_outliers"]
        lines += ["## Top candidates", "", cands[cols].head(40).to_markdown(index=False, floatfmt=".3g"), ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
