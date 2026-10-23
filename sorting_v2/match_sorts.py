"""Match units between two sorts of one recording (two time windows) by their templates.

Built for e097, where each anoxia challenge was sorted in its own window, so unit ids differ between
the sorts. Both windows come from the same probe, channels and CatGT preprocessing, and the tissue
didn't move between them (DREDGE step < 0.05 um), so templates are compared without correction.

    python match_sorts.py C:/Users/lpascu/resp_analysis/outputs/e097/intact_window_v2 ^
        C:/Users/lpascu/resp_analysis/outputs/e097/anoxia2_window_v2 ^
        --offset-a-s 1740 --offset-b-s 2250 --switch-a-s 60 --switch-b-s 60 ^
        --recovery-a C:/Users/lpascu/resp_analysis/outputs/e097/anoxia_recovery/intact_window_kilosort4 ^
        --recovery-b C:/Users/lpascu/resp_analysis/outputs/e097/anoxia_recovery/anoxia2_window_kilosort4 ^
        --out C:/Users/lpascu/resp_analysis/outputs/e097/anoxia_recovery/matched_units

Templates: each unit's mean waveform on all channels over its spikes in the `baseline` epoch (config
`epochs`), read from the preprocessed binary the same way as epoch_templates.npz (which keeps only 8
channels per unit), so state changes during the challenge don't drive the match. Units with fewer than
MIN_SPIKES baseline spikes use the analyzer's whole-window template instead (template_a/b = "window").
Cached in <out>/cache.

Units: label != noise, plus units reviewed `single`; units reviewed `noise` are left out (as in
anoxia_recovery.py). Noise-labelled units are still checked as possible partners of unmatched units.

For every pair of units whose template centres (ptp-weighted over the top 8 channels) lie within
MAX_DIST_UM, the measures of npxsort/stability.py, computed on templates:
  footprint_cosine  cosine of the signed footprint amplitudes on the union of both units' footprint
                    channels (up to 5 each, >= 30% of the largest; spikefeatures.footprint_setup). Each
                    amplitude is the extremum of the template's sign within +-0.2 ms of its extremum
                    time on that channel, as spike_pass measures spikes
  waveform_r        correlation of the two templates on the union of both units' top 8 channels, best
                    over lags of +-MAX_LAG samples (lag_samples > 0: A's waveform is that many samples
                    later than B's relative to the spike time)
  amp_ratio         peak-to-peak on the larger unit's peak channel, B / A
  dist_um           distance between the template centres
Assignment: one-to-one, Hungarian on 1 - footprint_cosine, in which a unit may stay unmatched. Without
that option the Hungarian gives every unit some partner and breaks up clear matches to do it (e097: 107
units with a >= 0.99 partner were left unmatched). Two passes:
  1. pairs passing the match thresholds (cosine >= MATCH_MIN_COS, waveform_r >= MATCH_MIN_WAVEFORM_R);
     leaving a unit unmatched costs (1 - MATCH_MIN_COS) / 2, so any passing pair beats two unmatched units
       match                           its runner-ups on both sides are < MATCH_MIN_COS
       match; a neighbour looks alike  a runner-up on either side is also >= MATCH_MIN_COS (a duplicate
                                       or split unit, or two similar neurons: which partner is right is
                                       uncertain)
  2. the remaining units, pairs with cosine >= DIFF_MAX_COS: "weak" (reported, not used)
Thresholds are D11's, checked against the nearest non-matching neighbour (calibration.png): the same unit
in two epochs of one sort versus the most similar other unit of the same sort and the runner-up candidate.
margin = cosine - the best other candidate's cosine (the smaller of the A-side and B-side margins).

With --recovery-a/-b (anoxia_recovery.py output folders) the accepted pairs are joined with both
unit_recovery.tsv tables.
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
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spikeinterface.core as si  # noqa: E402

from npxsort.spikefeatures import (AMP_HALF_WINDOW, WAVEFORM_AFTER, WAVEFORM_BEFORE,  # noqa: E402
                                   WAVEFORM_CHANNELS, footprint_setup)
from npxsort.stability import MIN_SPIKES, binary_memmap  # noqa: E402

log = logging.getLogger("match_sorts")

MAX_DIST_UM = 50.0
MAX_LAG = 5                  # samples (0.17 ms)
MAX_TEMPLATE_SPIKES = 2000   # baseline spikes averaged per unit (seeded random subset above this)
MATCH_MIN_COS = 0.99         # set from calibration.png; see the e097 numbers in matched_units/README.md
MATCH_MIN_WAVEFORM_R = 0.90
DIFF_MAX_COS = 0.97          # D11: about how different the nearest other neuron looks
END_WINDOW_S = 60.0          # "end of window" rate: the last 60 s of each sort
PLOT_RADIUS_UM = 100.0       # example figures: channels around the peak
ACCEPTED = ("match", "match; a neighbour looks alike")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_a", help="npxsort output_dir of sort A")
    p.add_argument("output_b", help="npxsort output_dir of sort B")
    p.add_argument("--sorter", default="kilosort4")
    p.add_argument("--baseline-epoch", default="baseline")
    p.add_argument("--offset-a-s", type=float, default=0.0, help="added to A's file time for raw-run time")
    p.add_argument("--offset-b-s", type=float, default=0.0)
    p.add_argument("--switch-a-s", type=float, default=None, help="gas switch in A, file time (for return latencies)")
    p.add_argument("--switch-b-s", type=float, default=None)
    p.add_argument("--recovery-a", default=None, help="anoxia_recovery.py output folder for A")
    p.add_argument("--recovery-b", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


class Sort:
    """One sorted window: units, spikes, baseline templates and per-unit template features."""

    def __init__(self, output_dir: str, sorter: str, baseline_epoch: str):
        self.out = Path(output_dir)
        sdir = self.out / sorter
        self.cfg = yaml.safe_load((self.out / "config_used.yaml").read_text())
        sorting = si.load(sdir / "analyzer" / "sorting")
        self.fs = sorting.get_sampling_frequency()
        self.unit_ids = np.asarray(sorting.unit_ids)
        sv = sorting.to_spike_vector()
        self.sample, self.unit = sv["sample_index"].astype(np.int64), sv["unit_index"].astype(np.int64)
        units = pd.read_csv(sdir / "units.tsv", sep="\t").set_index("unit_id").loc[self.unit_ids].reset_index()
        review = self.out / "review" / f"{sorter}_unit_review.tsv"
        verdicts = pd.read_csv(review, sep="\t").set_index("unit_id")["verdict"] if review.exists() else pd.Series(dtype=object)
        units["verdict"] = units["unit_id"].map(verdicts).fillna("")
        self.units = units
        self.keep = (((units["label"] != "noise") | (units["verdict"] == "single")) & (units["verdict"] != "noise")).to_numpy()
        ep = {e["name"]: (e["start_s"], e["end_s"]) for e in self.cfg["epochs"]}
        self.baseline = ep[baseline_epoch]
        self.epoch_templates = np.load(sdir / "epoch_templates.npz")
        self.window_templates = np.load(sdir / "analyzer" / "extensions" / "templates" / "average.npy").astype(np.float32)
        self.unit_locations = np.load(sdir / "analyzer" / "extensions" / "unit_locations" / "unit_locations.npy")

    def preprocessed(self):
        folder = Path(self.cfg["paths"]["scratch_dir"]) / "preprocessed"
        rec = si.load(folder)
        return folder, rec

    def rates(self, end_window_s: float):
        """Per unit: mean rate in the baseline epoch and in the last end_window_s of the window."""
        folder, rec = self.preprocessed()
        n = rec.get_num_samples()
        nu = len(self.unit_ids)
        b0, b1 = (int(round(t * self.fs)) for t in self.baseline)
        base = np.bincount(self.unit[(self.sample >= b0) & (self.sample < b1)], minlength=nu) / ((b1 - b0) / self.fs)
        e0 = n - int(round(end_window_s * self.fs))
        end = np.bincount(self.unit[self.sample >= e0], minlength=nu) / end_window_s
        return base, end, n / self.fs


def baseline_templates(sort: Sort, cache: Path, rng: np.random.Generator):
    """Mean waveform (uV) on all channels over up to MAX_TEMPLATE_SPIKES baseline spikes per unit."""
    if cache.exists():
        d = np.load(cache)
        if np.array_equal(d["unit_ids"], sort.unit_ids) and int(d["max_spikes"]) == MAX_TEMPLATE_SPIKES:
            return d["templates"], d["n_baseline"]
    folder, rec = sort.preprocessed()
    n, nch = rec.get_num_samples(), rec.get_num_channels()
    traces = binary_memmap(folder, nch)
    gain = rec.get_channel_gains().astype(np.float32)
    nu = len(sort.unit_ids)
    b0, b1 = (int(round(t * sort.fs)) for t in sort.baseline)
    s, u = sort.sample, sort.unit
    sel = np.flatnonzero((s >= b0) & (s < b1) & (s >= WAVEFORM_BEFORE) & (s + WAVEFORM_AFTER <= n))
    n_base = np.bincount(u[sel], minlength=nu)
    # seeded random subset of at most MAX_TEMPLATE_SPIKES per unit
    sel = sel[rng.permutation(sel.size)]
    sel = sel[np.argsort(u[sel], kind="stable")]
    us = u[sel]
    rank = np.arange(sel.size) - np.searchsorted(us, us)
    chosen = np.sort(sel[rank < MAX_TEMPLATE_SPIKES])
    samp, un = s[chosen], u[chosen]

    wwin = np.arange(-WAVEFORM_BEFORE, WAVEFORM_AFTER)
    sums = np.zeros((nu, wwin.size, nch), dtype=np.float64)
    cnt = np.zeros(nu, dtype=np.int64)
    step, batch = int(sort.fs), 256
    for c0 in range(b0, b1, step):
        i0, i1 = np.searchsorted(samp, [c0, c0 + step])
        if i0 == i1:
            continue
        a0, a1 = max(0, c0 - WAVEFORM_BEFORE), min(n, c0 + step + WAVEFORM_AFTER)
        block = np.asarray(traces[a0:a1], dtype=np.float32)
        for k0 in range(i0, i1, batch):
            k1 = min(i1, k0 + batch)
            uu = un[k0:k1]
            o = np.argsort(uu, kind="stable")
            uu = uu[o]
            wf = block[samp[k0:k1][o][:, None] - a0 + wwin[None, :]]
            starts = np.r_[0, np.flatnonzero(np.diff(uu)) + 1]
            sums[uu[starts]] += np.add.reduceat(wf, starts, axis=0)
            cnt[uu[starts]] += np.diff(np.r_[starts, uu.size])
    dense = (sums / np.maximum(cnt, 1)[:, None, None] * gain[None, None, :]).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, unit_ids=sort.unit_ids, templates=dense, n_baseline=n_base,
                        max_spikes=np.int64(MAX_TEMPLATE_SPIKES))
    return dense, n_base


def check_against_epoch_templates(sort: Sort, dense: np.ndarray, n_base: np.ndarray, epoch: str) -> float:
    """Largest |difference| (uV) from epoch_templates.npz's baseline waveforms, for units whose baseline
    spikes were all used (the two should be the same computation)."""
    et = sort.epoch_templates
    j = list(et["epochs"]).index(epoch)
    ok = (n_base >= MIN_SPIKES) & (n_base <= MAX_TEMPLATE_SPIKES)
    i = np.flatnonzero(ok)
    mine = dense[i[:, None, None], np.arange(dense.shape[1])[None, :, None], et["channels"][i][:, None, :]]
    return float(np.abs(mine - et["waveforms"][i, j]).max()) if i.size else np.nan


def features(dense: np.ndarray, locs: np.ndarray, nbefore: int):
    ptp = dense.max(axis=1) - dense.min(axis=1)
    fp = footprint_setup(dense, ptp, nbefore)[:3]
    wch = np.argsort(-ptp, axis=1)[:, :WAVEFORM_CHANNELS]
    w = np.take_along_axis(ptp, wch, axis=1)
    centre = (w[:, :, None] * locs[wch]).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)[:, None]
    return ptp, fp, wch, centre


def footprint_cosine(ta, tb, fa, fb, nbefore):
    """fa, fb = (channels, extremum offsets, signed extremum values) of each unit's footprint. On each
    channel of the union, both templates are measured at the offset and sign of whichever unit is
    larger there."""
    ref = {}
    for ch, off, tv in (fa, fb):
        for c, o, t in zip(ch, off, tv):
            if t != 0 and (c not in ref or abs(t) > abs(ref[c][1])):
                ref[c] = (o, t)
    chans = np.array(sorted(ref))
    k = np.array([nbefore + ref[c][0] for c in chans])
    pos = np.array([ref[c][1] > 0 for c in chans])
    idx = np.clip(k[:, None] + np.arange(-AMP_HALF_WINDOW, AMP_HALF_WINDOW + 1)[None, :], 0, ta.shape[0] - 1)
    va, vb = ((np.where(pos, t[idx, chans[:, None]].max(axis=1), t[idx, chans[:, None]].min(axis=1))) for t in (ta, tb))
    return float(va @ vb / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-9))


def waveform_r(ta, tb, wa, wb):
    w = np.union1d(wa, wb)
    a, b = ta[:, w], tb[:, w]
    n = a.shape[0]
    best, best_lag = -np.inf, 0
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        x = a[max(lag, 0):n + min(lag, 0)].ravel()
        y = b[max(-lag, 0):n + min(-lag, 0)].ravel()
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best:
            best, best_lag = r, lag
    return best, best_lag


def compare(ta, tb, fa, fb, wa, wb, nbefore):
    cos = footprint_cosine(ta, tb, fa, fb, nbefore)
    r, lag = waveform_r(ta, tb, wa, wb)
    return cos, r, lag


def fp_of(fp, i):
    return tuple(x[i] for x in fp)


def cross_pairs(A, B, ta, tb, fa, fb, nbefore):
    """All A-B unit pairs with template centres within MAX_DIST_UM."""
    d = np.linalg.norm(fa["centre"][:, None, :] - fb["centre"][None, :, :], axis=2)
    rows = []
    for i, j in zip(*np.nonzero(d <= MAX_DIST_UM)):
        cos, r, lag = compare(ta[i], tb[j], fp_of(fa["fp"], i), fp_of(fb["fp"], j), fa["wch"][i], fb["wch"][j], nbefore)
        c = int(np.argmax(np.maximum(fa["ptp"][i], fb["ptp"][j])))
        rows.append({"ia": i, "ib": j, "footprint_cosine": cos, "waveform_r": r, "lag_samples": lag,
                     "dist_um": d[i, j], "amp_ratio": fb["ptp"][j, c] / max(fa["ptp"][i, c], 1e-6)})
    return pd.DataFrame(rows)


def nearest_distinct(S, t, f, nbefore):
    """For each kept unit: the most similar other kept unit of the same sort within MAX_DIST_UM."""
    keep = np.flatnonzero(S.keep)
    d = np.linalg.norm(f["centre"][keep][:, None] - f["centre"][keep][None], axis=2)
    rows = []
    for a, i in enumerate(keep):
        best = (np.nan, np.nan, -1)
        for b in np.flatnonzero((d[a] <= MAX_DIST_UM) & (np.arange(keep.size) != a)):
            j = keep[b]
            cos, r, _ = compare(t[i], t[j], fp_of(f["fp"], i), fp_of(f["fp"], j), f["wch"][i], f["wch"][j], nbefore)
            if not cos <= best[0]:
                best = (cos, r, int(S.unit_ids[j]))
        rows.append({"unit_id": int(S.unit_ids[i]), "nn_cos": best[0], "nn_waveform_r": best[1], "nn_unit": best[2]})
    return pd.DataFrame(rows)


def same_unit_reference(S, n_base, nbefore):
    """Same unit, baseline vs the sort's last epoch (epoch_templates.npz, on those 8 channels): how
    similar one neuron's templates are when taken minutes apart in a different breathing state."""
    et = S.epoch_templates
    names = list(et["epochs"])
    jb, jl = names.index("baseline"), len(names) - 1
    rows = []
    for i in np.flatnonzero(S.keep & (et["counts"][:, jb] >= MIN_SPIKES) & (et["counts"][:, jl] >= MIN_SPIKES)):
        ta, tb = et["waveforms"][i, jb], et["waveforms"][i, jl]
        fa = footprint_setup(ta[None], np.ptp(ta, axis=0)[None], nbefore)[:3]
        fb = footprint_setup(tb[None], np.ptp(tb, axis=0)[None], nbefore)[:3]
        cos, r, lag = compare(ta, tb, fp_of(fa, 0), fp_of(fb, 0), np.arange(ta.shape[1]), np.arange(ta.shape[1]), nbefore)
        rows.append({"unit_id": int(S.unit_ids[i]), "epoch": names[jl], "footprint_cosine": cos, "waveform_r": r})
    return pd.DataFrame(rows)


def hungarian_with_reject(pairs: pd.DataFrame, reject_cost: float) -> set:
    """One-to-one assignment minimising the sum of 1 - cosine over the given pairs, where leaving a
    unit unmatched costs reject_cost (dummy rows and columns)."""
    if not len(pairs):
        return set()
    ua, ub = np.unique(pairs["ia"]), np.unique(pairs["ib"])
    na, nb = ua.size, ub.size
    m = np.full((na + nb, nb + na), 1e6)
    m[np.searchsorted(ua, pairs["ia"]), np.searchsorted(ub, pairs["ib"])] = 1 - pairs["footprint_cosine"].to_numpy()
    m[np.arange(na), nb + np.arange(na)] = reject_cost
    m[na + np.arange(nb), np.arange(nb)] = reject_cost
    m[na:, nb:] = 0
    r, c = linear_sum_assignment(m)
    ok = (r < na) & (c < nb)
    return {(int(ua[i]), int(ub[j])) for i, j in zip(r[ok], c[ok])}


def assign(A, B, pairs):
    """Two-pass assignment (module docstring) over kept-kept candidate pairs; adds runner-ups, margins
    and class."""
    ka, kb = np.flatnonzero(A.keep), np.flatnonzero(B.keep)
    kk = pairs[pairs["ia"].isin(ka) & pairs["ib"].isin(kb)]
    passing = kk[(kk["footprint_cosine"] >= MATCH_MIN_COS) & (kk["waveform_r"] >= MATCH_MIN_WAVEFORM_R)]
    first = hungarian_with_reject(passing, (1 - MATCH_MIN_COS) / 2)
    done_a, done_b = {i for i, _ in first}, {j for _, j in first}
    rest = kk[(kk["footprint_cosine"] >= DIFF_MAX_COS) & ~kk["ia"].isin(done_a) & ~kk["ib"].isin(done_b)]
    second = hungarian_with_reject(rest, (1 - DIFF_MAX_COS) / 2)
    chosen = first | second
    kk = kk.assign(assigned=[(i, j) in chosen for i, j in zip(kk["ia"], kk["ib"])])
    best_a = kk.loc[kk.groupby("ia")["footprint_cosine"].idxmax()].set_index("ia")["ib"]
    best_b = kk.loc[kk.groupby("ib")["footprint_cosine"].idxmax()].set_index("ib")["ia"]
    out = []
    for _, p in kk[kk["assigned"]].iterrows():
        i, j = int(p["ia"]), int(p["ib"])
        oa = kk[(kk["ia"] == i) & (kk["ib"] != j)].sort_values("footprint_cosine", ascending=False)
        ob = kk[(kk["ib"] == j) & (kk["ia"] != i)].sort_values("footprint_cosine", ascending=False)
        ru_a = oa["footprint_cosine"].iloc[0] if len(oa) else np.nan
        ru_b = ob["footprint_cosine"].iloc[0] if len(ob) else np.nan
        row = p.drop("assigned").to_dict()
        row.update(
            unit_a=int(A.unit_ids[i]), unit_b=int(B.unit_ids[j]),
            runner_up_of_a=int(B.unit_ids[int(oa["ib"].iloc[0])]) if len(oa) else -1, runner_up_of_a_cos=ru_a,
            runner_up_of_b=int(A.unit_ids[int(ob["ia"].iloc[0])]) if len(ob) else -1, runner_up_of_b_cos=ru_b,
            margin=np.nanmin([p["footprint_cosine"] - ru_a, p["footprint_cosine"] - ru_b, np.inf]),
            mutual_best=bool(best_a[i] == j and best_b[j] == i),
            match_class=("weak" if (i, j) in second else
                         "match; a neighbour looks alike" if np.nanmax([ru_a, ru_b, -np.inf]) >= MATCH_MIN_COS else "match"),
        )
        out.append(row)
    m = pd.DataFrame(out)
    m.loc[~np.isfinite(m["margin"]), "margin"] = np.nan
    return m, kk


def unit_info(S, i, n_base, tmpl_src, rates, prefix):
    u = S.units.iloc[i]
    return {f"label_{prefix}": u["label"], f"verdict_{prefix}": u["verdict"], f"depth_{prefix}_um": u["depth_um"],
            f"peak_ch_{prefix}": int(u["peak_channel_index"]), f"n_base_{prefix}": int(n_base[i]),
            f"template_{prefix}": tmpl_src[i], f"rate_base_{prefix}_hz": rates[0][i], f"rate_end_{prefix}_hz": rates[1][i]}


def unmatched_table(S, other, side, pairs, kk, matched, n_base, tmpl_src, rates):
    """Kept units of one sort without an accepted partner, with the reason and their best partners."""
    me, them = ("ia", "ib") if side == "a" else ("ib", "ia")
    o = "b" if side == "a" else "a"
    acc = matched[matched["match_class"].isin(ACCEPTED)]
    partner_of = dict(zip(acc[f"unit_{o}"], acc[f"unit_{side}"]))
    accepted_units = set(acc[f"unit_{side}"])
    assigned = matched.set_index(f"unit_{side}")
    rows = []
    for i in np.flatnonzero(S.keep):
        uid = int(S.unit_ids[i])
        if uid in accepted_units:
            continue
        row = {"unit_id": uid, **unit_info(S, i, n_base, tmpl_src, rates, side)}
        cand = kk[kk[me] == i].sort_values("footprint_cosine", ascending=False)
        row["n_candidates"] = len(cand)
        if len(cand):
            c = cand.iloc[0]
            best = int(other.unit_ids[int(c[them])])
            row.update(best_candidate=best, best_cos=c["footprint_cosine"], best_waveform_r=c["waveform_r"],
                       best_dist_um=c["dist_um"])
            if uid in assigned.index:
                a = assigned.loc[uid]
                code = "weak_partner"
                reason = f"weak partner #{int(a[f'unit_{o}'])} (cos {a['footprint_cosine']:.3f}, r {a['waveform_r']:.2f})"
            elif best in partner_of:
                code = "best_candidate_taken"
                reason = f"best candidate #{best} (cos {c['footprint_cosine']:.3f}) is matched to #{partner_of[best]}"
            elif c["footprint_cosine"] < DIFF_MAX_COS:
                code = "nearest_differs"
                reason = f"most similar unit #{best} has cos {c['footprint_cosine']:.3f}"
            else:
                code = "unassigned"
                reason = f"best candidate #{best} cos {c['footprint_cosine']:.3f}, r {c['waveform_r']:.2f}"
        else:
            code, reason = "no_candidate", f"no unit within {MAX_DIST_UM:g} um"
        row["reason_code"], row["reason"] = code, reason
        # any unit of the other sort, including noise-labelled ones
        allp = pairs[pairs[me] == i].sort_values("footprint_cosine", ascending=False)
        if len(allp):
            c = allp.iloc[0]
            j = int(c[them])
            row.update(best_any=int(other.unit_ids[j]), best_any_label=other.units.iloc[j]["label"],
                       best_any_cos=c["footprint_cosine"], best_any_waveform_r=c["waveform_r"])
        rows.append(row)
    return pd.DataFrame(rows)


def recovery_join(matched, args, A_rates, B_rates):
    ra = pd.read_csv(Path(args.recovery_a) / "unit_recovery.tsv", sep="\t")
    rb = pd.read_csv(Path(args.recovery_b) / "unit_recovery.tsv", sep="\t")
    cols = ["unit_id", "label", "perturbed", "direction", "base_rate_hz", "band_lo_hz", "band_hi_hz",
            "challenge_extreme_rate_hz", "return_raw_s", "stable_from_raw_s"]
    acc = matched[matched["match_class"].isin(ACCEPTED)]
    keepcols = ["unit_a", "unit_b", "match_class", "footprint_cosine", "waveform_r", "margin",
                "rate_base_a_hz", "rate_end_a_hz", "rate_base_b_hz", "rate_end_b_hz"]
    j = (acc[keepcols]
         .merge(ra[cols].add_prefix("a_"), left_on="unit_a", right_on="a_unit_id", how="left")
         .merge(rb[cols].add_prefix("b_"), left_on="unit_b", right_on="b_unit_id", how="left")
         .drop(columns=["a_unit_id", "b_unit_id"]))
    sw_a = args.offset_a_s + args.switch_a_s
    sw_b = args.offset_b_s + args.switch_b_s
    j["a_return_latency_s"] = j["a_return_raw_s"] - sw_a
    j["b_return_latency_s"] = j["b_return_raw_s"] - sw_b

    def outcome(prefix):
        p = j[f"{prefix}_perturbed"]
        o = np.where(p.isna(), "not in table", np.where(p != True, "not perturbed",  # noqa: E712
                     np.where(j[f"{prefix}_return_raw_s"].notna(), "returned", "not returned")))
        return o
    j["a_outcome"], j["b_outcome"] = outcome("a"), outcome("b")
    # where is the unit at B's baseline relative to its own pre-anoxia-1 band?
    rb_ = j["rate_base_b_hz"]
    j["b_base_vs_a_band"] = np.where(j["a_band_lo_hz"].isna(), "",
                                     np.where(rb_ < j["a_band_lo_hz"], "below",
                                              np.where(rb_ > j["a_band_hi_hz"], "above", "inside")))
    j["b_base_over_a_base"] = (rb_ + 0.1) / (j["rate_base_a_hz"] + 0.1)
    j["b_base_over_a_end"] = (rb_ + 0.1) / (j["rate_end_a_hz"] + 0.1)

    s = {}
    a_out = np.where(ra["perturbed"] != True, "not perturbed",  # noqa: E712
                     np.where(ra["return_raw_s"].notna(), "returned", "not returned"))
    ra = ra.assign(outcome=a_out)
    matched_a = set(acc["unit_a"])
    any_a = set(matched["unit_a"])
    s["a_recovery_units_by_outcome"] = {
        k: {"n": int(len(g)), "accepted_match": int(g["unit_id"].isin(matched_a).sum()),
            "assigned_any_class": int(g["unit_id"].isin(any_a).sum())}
        for k, g in ra.groupby("outcome")}
    both = j[(j["a_outcome"].isin(["returned", "not returned"])) & (j["b_outcome"].isin(["returned", "not returned"]))]
    s["n_pairs_perturbed_in_both"] = int(len(both))
    s["direction_crosstab_rows_a_cols_b"] = pd.crosstab(both["a_direction"], both["b_direction"]).to_dict(orient="index")
    s["direction_same"] = int((both["a_direction"] == both["b_direction"]).sum())
    s["outcome_crosstab_rows_a_cols_b"] = pd.crosstab(both["a_outcome"], both["b_outcome"]).to_dict(orient="index")
    r2 = both.dropna(subset=["a_return_latency_s", "b_return_latency_s"])
    if len(r2) > 2:
        rho = spearmanr(r2["a_return_latency_s"], r2["b_return_latency_s"])
        s["return_latency_both_returned"] = {
            "n": int(len(r2)), "spearman_rho": float(rho.statistic), "p": float(rho.pvalue),
            "median_a_s": float(r2["a_return_latency_s"].median()), "median_b_s": float(r2["b_return_latency_s"].median()),
            "median_b_minus_a_s": float((r2["b_return_latency_s"] - r2["a_return_latency_s"]).median())}
    groups = {}
    for k, g in j.groupby("a_outcome"):
        groups[k] = {"n": int(len(g)),
                     "b_base_vs_a_band": g["b_base_vs_a_band"].value_counts().to_dict(),
                     "median_rate_base_a_hz": float(g["rate_base_a_hz"].median()),
                     "median_rate_end_a_hz": float(g["rate_end_a_hz"].median()),
                     "median_rate_base_b_hz": float(g["rate_base_b_hz"].median()),
                     "median_rate_end_b_hz": float(g["rate_end_b_hz"].median()),
                     "median_b_base_over_a_base": float(g["b_base_over_a_base"].median()),
                     "median_b_base_over_a_end": float(g["b_base_over_a_end"].median()),
                     "b_outcome": g["b_outcome"].value_counts().to_dict()}
    s["by_a_outcome"] = groups
    return j, s, ra


def best_candidates(kk):
    """Each kept unit's most similar kept unit in the other sort (both directions stacked)."""
    return pd.concat([kk.loc[kk.groupby(k)["footprint_cosine"].idxmax(), ["footprint_cosine", "waveform_r"]]
                      for k in ("ia", "ib")])


def plot_calibration(matched, kk, nn_a, nn_b, ref_a, ref_b, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ru = pd.concat([matched["runner_up_of_a_cos"], matched["runner_up_of_b_cos"]]).dropna()
    series = [
        ("same unit, baseline vs last epoch (within each sort)", pd.concat([ref_a, ref_b]), "#1baf7a", "-"),
        ("each unit's most similar unit in the other sort", best_candidates(kk), "#2a78d6", "-"),
        ("runner-up of an assigned pair (other sort)", pd.DataFrame({"footprint_cosine": ru}), "#eda100", "--"),
        ("nearest other unit, same sort", pd.concat([nn_a, nn_b]).rename(columns={"nn_cos": "footprint_cosine",
                                                                              "nn_waveform_r": "waveform_r"}), "#eb6834", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, df, colour, ls in series:
        v = 1 - df["footprint_cosine"].dropna().to_numpy()
        v = np.sort(np.clip(v, 1e-5, None))
        axes[0].step(v, np.arange(1, v.size + 1) / v.size, where="post", color=colour, ls=ls, lw=1.6,
                     label=f"{name} (n={v.size})")
        if "waveform_r" in df:
            w = np.sort(df["waveform_r"].dropna().to_numpy())
            axes[1].step(w, np.arange(1, w.size + 1) / w.size, where="post", color=colour, ls=ls, lw=1.6)
    axes[0].set_xscale("log")
    axes[0].axvline(1 - MATCH_MIN_COS, color="0.25", lw=0.8)
    axes[0].axvline(1 - DIFF_MAX_COS, color="0.25", lw=0.8, ls=":")
    axes[0].text(1 - MATCH_MIN_COS, 0.02, f" match >= {MATCH_MIN_COS}", fontsize=8, color="0.25")
    axes[0].text(1 - DIFF_MAX_COS, 0.10, f" rejected < {DIFF_MAX_COS}", fontsize=8, color="0.25")
    axes[0].set_xlabel("1 - footprint cosine (log)")
    axes[0].set_ylabel("cumulative fraction")
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    axes[1].axvline(MATCH_MIN_WAVEFORM_R, color="0.25", lw=0.8)
    axes[1].set_xlim(0.3, 1.0)
    axes[1].set_xlabel("waveform r (top 8 channels of both, best lag)")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Calibration: how similar the same neuron is, versus the nearest different neuron", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=130)
    plt.close(fig)


def plot_pairs(rows, A, B, ta, tb, fa, fb, locs, fs, out, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    if not n:
        return
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.6 * nrow), squeeze=False)
    t = (np.arange(ta.shape[1]) - WAVEFORM_BEFORE) / fs * 1e3
    for ax, (_, p) in zip(axs.ravel(), rows.iterrows()):
        i = int(np.flatnonzero(A.unit_ids == p["unit_a"])[0])
        j = int(np.flatnonzero(B.unit_ids == p["unit_b"])[0])
        both = np.maximum(fa["ptp"][i], fb["ptp"][j])
        near = np.flatnonzero(np.linalg.norm(locs - locs[np.argmax(both)], axis=1) <= PLOT_RADIUS_UM)
        chans = near[np.argsort(-both[near])][:10]
        scale = 1.0 / max(both[chans].max(), 1e-6)
        xspan = (t[-1] - t[0]) * 1.15
        for c in chans:
            x0 = (locs[c, 0] - locs[:, 0].min()) / 32.0 * xspan
            y0 = locs[c, 1] / 15.0 * 0.5
            ax.plot(x0 + t, y0 + ta[i][:, c] * scale * 1.6, color="#2a78d6", lw=1.3)
            ax.plot(x0 + t, y0 + tb[j][:, c] * scale * 1.6, color="#eb6834", lw=1.0)
        ax.set_title(f"A #{int(p['unit_a'])} ({p['label_a']}) - B #{int(p['unit_b'])} ({p['label_b']})\n"
                     f"{p['match_class']}\ncos {p['footprint_cosine']:.4f}, r {p['waveform_r']:.3f}, "
                     f"{p['dist_um']:.1f} um, amp B/A {p['amp_ratio']:.2f}", fontsize=8)
        ax.text(0.01, 0.01, f"peak ch {int(p['peak_ch_a'])} / {int(p['peak_ch_b'])}, {both[chans[0]]:.0f} uV pp"
                + (f"\n{p['note']}" if isinstance(p.get("note"), str) else ""),
                transform=ax.transAxes, fontsize=7, color="0.35")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}  (blue = sort A baseline template, orange = sort B; largest 10 channels within "
                 f"{PLOT_RADIUS_UM:g} um of the peak, in probe layout)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(out / fname, dpi=110)
    plt.close(fig)


def plot_recovery(j, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"returned": "#2a78d6", "not returned": "#eb6834", "not perturbed": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    eps = 0.05
    for k, c in colours.items():
        g = j[j["a_outcome"] == k]
        axes[0].scatter(g["rate_base_a_hz"] + eps, g["rate_base_b_hz"] + eps, s=14, color=c, alpha=0.8,
                        edgecolor="white", lw=0.4, label=f"anoxia #1: {k} (n={len(g)})")
        axes[1].scatter(g["rate_end_a_hz"] + eps, g["rate_base_b_hz"] + eps, s=14, color=c, alpha=0.8,
                        edgecolor="white", lw=0.4)
    for ax, xl in zip(axes[:2], ["A: pre-anoxia-1 baseline rate (Hz, 1740-1800 s)",
                                 "A: end-of-window rate (Hz, last 60 s)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        lim = [eps * 0.8, 200]
        ax.plot(lim, lim, color="0.5", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"{xl} + {eps}")
        ax.set_ylabel(f"B: pre-anoxia-2 baseline rate (Hz, 2250-2310 s) + {eps}")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    r2 = j[(j["a_outcome"] == "returned") & (j["b_outcome"] == "returned")]
    axes[2].scatter(r2["a_return_latency_s"], r2["b_return_latency_s"], s=14, color="#2a78d6", alpha=0.8,
                    edgecolor="white", lw=0.4)
    lim = [0, max(r2[["a_return_latency_s", "b_return_latency_s"]].max().max() * 1.05, 10)] if len(r2) else [0, 1]
    axes[2].plot(lim, lim, color="0.5", lw=0.8)
    axes[2].set_xlabel("return after switch, anoxia #1 (s)")
    axes[2].set_ylabel("return after switch, anoxia #2 (s)")
    axes[2].set_title(f"units perturbed and returned in both (n={len(r2)})", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "recovery_comparison.png", dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A = Sort(args.output_a, args.sorter, args.baseline_epoch)
    B = Sort(args.output_b, args.sorter, args.baseline_epoch)
    nbefore = WAVEFORM_BEFORE
    locs = A.preprocessed()[1].get_channel_locations()
    if not np.array_equal(locs, B.preprocessed()[1].get_channel_locations()):
        raise SystemExit("the two sorts use different channel maps")

    S, T, F, NB, SRC, R = {}, {}, {}, {}, {}, {}
    for key, X in (("a", A), ("b", B)):
        rng = np.random.default_rng(args.seed)
        dense, n_base = baseline_templates(X, out / "cache" / f"baseline_templates_{key}.npz", rng)
        log.info("sort %s: baseline templates vs epoch_templates.npz, max |diff| %.3g uV",
                 key.upper(), check_against_epoch_templates(X, dense, n_base, args.baseline_epoch))
        few = n_base < MIN_SPIKES
        dense[few] = X.window_templates[few]
        SRC[key] = np.where(few, "window", "baseline")
        ptp, fp, wch, centre = features(dense, locs, nbefore)
        T[key], NB[key] = dense, n_base
        F[key] = {"ptp": ptp, "fp": fp, "wch": wch, "centre": centre}
        R[key] = X.rates(END_WINDOW_S)
        log.info("sort %s: %d units, %d kept, %d kept with < %d baseline spikes (whole-window template)",
                 key.upper(), len(X.unit_ids), X.keep.sum(), (X.keep & few).sum(), MIN_SPIKES)

    pairs = cross_pairs(A, B, T["a"], T["b"], F["a"], F["b"], nbefore)
    matched, kk = assign(A, B, pairs)
    for side, X in (("a", A), ("b", B)):
        idx = matched["ia" if side == "a" else "ib"].astype(int)
        info = pd.DataFrame([unit_info(X, i, NB[side], SRC[side], R[side], side) for i in idx], index=matched.index)
        matched = pd.concat([matched, info], axis=1)
        matched[f"x_{side}_um"] = F[side]["centre"][idx, 0]
        matched[f"y_{side}_um"] = F[side]["centre"][idx, 1]
    order = ["unit_a", "unit_b", "match_class", "footprint_cosine", "waveform_r", "lag_samples", "dist_um", "amp_ratio",
             "margin", "mutual_best", "label_a", "label_b", "verdict_a", "verdict_b", "depth_a_um", "depth_b_um",
             "y_a_um", "y_b_um", "x_a_um", "x_b_um", "peak_ch_a", "peak_ch_b", "runner_up_of_a", "runner_up_of_a_cos",
             "runner_up_of_b", "runner_up_of_b_cos", "template_a", "template_b", "n_base_a", "n_base_b",
             "rate_base_a_hz", "rate_end_a_hz", "rate_base_b_hz", "rate_end_b_hz"]
    matched = matched[order].sort_values(["match_class", "unit_a"]).reset_index(drop=True)
    matched.to_csv(out / "matched_pairs.tsv", sep="\t", index=False, float_format="%.4f")
    un_a = unmatched_table(A, B, "a", pairs, kk, matched, NB["a"], SRC["a"], R["a"])
    un_b = unmatched_table(B, A, "b", pairs, kk, matched, NB["b"], SRC["b"], R["b"])
    un_a.to_csv(out / "unmatched_a.tsv", sep="\t", index=False, float_format="%.4f")
    un_b.to_csv(out / "unmatched_b.tsv", sep="\t", index=False, float_format="%.4f")

    nn_a, nn_b = nearest_distinct(A, T["a"], F["a"], nbefore), nearest_distinct(B, T["b"], F["b"], nbefore)
    ref_a, ref_b = same_unit_reference(A, NB["a"], nbefore), same_unit_reference(B, NB["b"], nbefore)
    calib = pd.concat([ref_a.assign(set="same_unit_a"), ref_b.assign(set="same_unit_b"),
                       nn_a.rename(columns={"nn_cos": "footprint_cosine", "nn_waveform_r": "waveform_r"}).assign(set="nearest_other_a"),
                       nn_b.rename(columns={"nn_cos": "footprint_cosine", "nn_waveform_r": "waveform_r"}).assign(set="nearest_other_b")])
    calib.to_csv(out / "calibration_reference.tsv", sep="\t", index=False, float_format="%.5f")

    def q(v):
        v = pd.Series(v).dropna()
        return {"n": int(v.size), **{f"p{int(p * 100)}": float(v.quantile(p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
                f"frac_ge_{MATCH_MIN_COS}": float((v >= MATCH_MIN_COS).mean()),
                f"frac_lt_{DIFF_MAX_COS}": float((v < DIFF_MAX_COS).mean())}
    ru = pd.concat([matched["runner_up_of_a_cos"], matched["runner_up_of_b_cos"]])
    summary = {
        "thresholds": {"max_dist_um": MAX_DIST_UM, "match_min_cos": MATCH_MIN_COS,
                       "match_min_waveform_r": MATCH_MIN_WAVEFORM_R, "diff_max_cos": DIFF_MAX_COS},
        "units": {k: {"total": int(len(X.unit_ids)), "kept": int(X.keep.sum()),
                      "kept_window_template": int((X.keep & (SRC[k] == "window")).sum())} for k, X in (("a", A), ("b", B))},
        "match_class_counts": matched["match_class"].value_counts().to_dict(),
        "unmatched": {"a": int(len(un_a)), "b": int(len(un_b))},
        "label_crosstab_accepted": pd.crosstab(matched.loc[matched["match_class"].isin(ACCEPTED), "label_a"],
                                               matched.loc[matched["match_class"].isin(ACCEPTED), "label_b"]).to_dict(),
        "calibration_footprint_cosine": {
            "same_unit_baseline_vs_last_epoch": q(pd.concat([ref_a, ref_b])["footprint_cosine"]),
            "best_candidate_other_sort": q(best_candidates(kk)["footprint_cosine"]),
            "accepted_pairs": q(matched.loc[matched["match_class"].isin(ACCEPTED), "footprint_cosine"]),
            "runner_up": q(ru),
            "nearest_other_same_sort": q(pd.concat([nn_a, nn_b])["nn_cos"]),
        },
        "calibration_waveform_r": {
            "same_unit_baseline_vs_last_epoch": q(pd.concat([ref_a, ref_b])["waveform_r"]),
            "accepted_pairs": q(matched.loc[matched["match_class"].isin(ACCEPTED), "waveform_r"]),
            "nearest_other_same_sort": q(pd.concat([nn_a, nn_b])["nn_waveform_r"]),
        },
        "accepted_lag_samples": matched.loc[matched["match_class"].isin(ACCEPTED), "lag_samples"].value_counts().to_dict(),
        "accepted_amp_ratio_quantiles": q(matched.loc[matched["match_class"].isin(ACCEPTED), "amp_ratio"]),
    }
    plot_calibration(matched, kk, nn_a, nn_b, ref_a, ref_b, out)

    acc = matched[matched["match_class"].isin(ACCEPTED)].sort_values("footprint_cosine")
    pick = acc.iloc[np.unique(np.linspace(0, len(acc) - 1, min(8, len(acc))).round().astype(int))] if len(acc) else acc
    fs = A.fs
    if args.recovery_a and args.recovery_b:
        joined, rsum, ra = recovery_join(matched, args, R["a"], R["b"])
        joined.to_csv(out / "recovery_joined.tsv", sep="\t", index=False, float_format="%.3f")
        summary["recovery"] = rsum
        plot_recovery(joined, out)
        nr = acc[acc["unit_a"].isin(joined.loc[joined["a_outcome"] == "not returned", "unit_a"])].sort_values("rate_base_b_hz")
        if len(nr):
            nr = nr.iloc[np.unique(np.linspace(0, len(nr) - 1, min(12, len(nr))).round().astype(int))].copy()
            nr["note"] = [f"rate A base {a:.1f} > A end {b:.1f} > B base {c:.1f} Hz" for a, b, c in
                          zip(nr["rate_base_a_hz"], nr["rate_end_a_hz"], nr["rate_base_b_hz"])]
            plot_pairs(nr, A, B, T["a"], T["b"], F["a"], F["b"], locs, fs, out, "not_returned_pairs.png",
                       "Units not back at the end of anoxia #1, matched in sort B")
        # the not-returned units that found no accepted partner
        nr_un = un_a[un_a["unit_id"].isin(ra.loc[ra["outcome"] == "not returned", "unit_id"])]
        nr_un.to_csv(out / "not_returned_unmatched_a.tsv", sep="\t", index=False, float_format="%.4f")
        summary["recovery"]["not_returned_unmatched_reasons"] = nr_un["reason_code"].value_counts().to_dict()
    differs = un_a[un_a["reason_code"] == "nearest_differs"].sort_values("best_cos", ascending=False).head(2)
    nearest = []
    for _, u in differs.iterrows():
        i = int(np.flatnonzero(A.unit_ids == u["unit_id"])[0])
        j = int(np.flatnonzero(B.unit_ids == u["best_candidate"])[0])
        p = pairs[(pairs["ia"] == i) & (pairs["ib"] == j)].iloc[0]
        nearest.append({"unit_a": u["unit_id"], "unit_b": u["best_candidate"], "label_a": u["label_a"],
                        "label_b": B.units.iloc[j]["label"], "match_class": "unmatched, most similar B unit",
                        "footprint_cosine": p["footprint_cosine"], "waveform_r": p["waveform_r"], "dist_um": p["dist_um"],
                        "amp_ratio": p["amp_ratio"], "peak_ch_a": u["peak_ch_a"],
                        "peak_ch_b": int(B.units.iloc[j]["peak_channel_index"])})
    pick = pd.concat([pick, matched[matched["match_class"] == "match; a neighbour looks alike"].head(2),
                      matched[matched["match_class"] == "weak"].head(2), pd.DataFrame(nearest)])
    plot_pairs(pick.drop_duplicates(["unit_a", "unit_b"]), A, B, T["a"], T["b"], F["a"], F["b"], locs, fs, out,
               "example_pairs.png", "Example A-B pairs")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
