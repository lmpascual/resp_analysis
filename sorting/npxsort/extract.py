"""Pull hand-defined units out of a sorter's output (config: curation.extract_units).

Some neurons are real but too small or too unusual for the sorters to give them their own unit:
for e097 a positive-going cell on ch 148/150 that fires ~30 spikes, only during gasping, and was
absorbed into KS4 #361 and Lupin #1079. A rule describes such a unit by where and how it looks:

    curation:
      extract_units:
        - name: gasp_pos_ch148        # becomes the unit's `extracted_as` label in units.tsv
          sorters: [kilosort4, lupin]  # default: every sorter
          channel: 148                 # channel index where the unit is largest
          polarity: positive           # positive | negative: sign of its dominant peak
          threshold_sd: 5.0            # peak height on `channel`, in noise SDs
          seed_window_s: [95, 125]     # spikes here define the unit's footprint (file time, s)
          search_window_s: null        # where to collect its spikes; null = whole recording
          min_footprint_cosine: 0.95   # how closely a spike's footprint must match the seed footprint
          neighbour_um: 60.0           # channels within this distance of `channel` form the footprint

Detection: polarity-dominant peaks on `channel` (larger than the opposite-sign peak within ±0.5 ms),
at least 1 ms apart. Footprint: each peak's signed amplitude (rule polarity) on the neighbouring
channels, normalised. The unit is every candidate whose footprint has cosine >= min_footprint_cosine
with the median seed footprint. Its spikes are then removed from the units that absorbed them (units
with >= 5 coincidences within ±0.5 ms, covering >= 30% of the extracted spikes and >= 5x chance), and
it is added as a new unit. Existing sorting properties (e.g. Kilosort's main_channel_id) are kept.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks

import spikeinterface.core as si

log = logging.getLogger(__name__)

RULE_DEFAULTS = {"sorters": None, "search_window_s": None, "min_footprint_cosine": 0.95, "neighbour_um": 60.0}
RULE_REQUIRED = ("name", "channel", "polarity", "threshold_sd", "seed_window_s")
COINCIDENCE_MS = 0.5
MIN_ABSORBED = 5
MIN_ABSORBED_FRACTION = 0.3
MIN_ABSORBED_EXCESS = 5.0


def rule_with_defaults(rule: dict) -> dict:
    return {**RULE_DEFAULTS, **rule}


def find_rule_events(rec: si.BaseRecording, rule: dict) -> dict:
    """Detect the rule's unit in the recording. Returns sample indices and diagnostics."""
    r = rule_with_defaults(rule)
    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    ch = int(r["channel"])
    sign = 1.0 if r["polarity"] == "positive" else -1.0
    ids = rec.channel_ids
    x = rec.get_traces(channel_ids=[ids[ch]], return_in_uV=True)[:, 0].astype(np.float32)
    sd = float(np.median(np.abs(x - np.median(x))) / 0.6745)
    y = sign * x
    peaks, _ = find_peaks(y, height=r["threshold_sd"] * sd, distance=max(1, int(0.001 * fs)))
    w = int(COINCIDENCE_MS / 1000 * fs)
    dominant = np.array([y[p] > -y[max(0, p - w):p + w + 1].min() for p in peaks], dtype=bool)
    cand = peaks[dominant] if peaks.size else peaks

    locs = rec.get_channel_locations()
    neigh = np.nonzero(np.linalg.norm(locs - locs[ch], axis=1) <= r["neighbour_um"])[0]
    half = 6
    fps = np.zeros((cand.size, neigh.size), dtype=np.float32)
    for k, p in enumerate(cand):
        a0, a1 = max(0, p - half), min(n, p + half + 1)
        seg = rec.get_traces(start_frame=a0, end_frame=a1, channel_ids=ids[neigh], return_in_uV=True)
        fps[k] = (sign * seg).max(axis=0)
    norm = np.linalg.norm(fps, axis=1, keepdims=True)
    fpn = fps / np.maximum(norm, 1e-6)

    t = cand / fs
    s0, s1 = r["seed_window_s"]
    seeds = (t >= s0) & (t < s1)
    if seeds.sum() < 5:
        raise ValueError(f"extract_units '{r['name']}': only {int(seeds.sum())} seed peaks on ch {ch} in "
                         f"{r['seed_window_s']} s above {r['threshold_sd']} SD; need at least 5.")
    tmpl = np.median(fpn[seeds], axis=0)
    tmpl /= np.linalg.norm(tmpl)
    cos = fpn @ tmpl
    inside = np.ones(cand.size, dtype=bool)
    if r["search_window_s"] is not None:
        w0, w1 = r["search_window_s"]
        inside = (t >= w0) & (t < w1)
    keep = inside & (cos >= r["min_footprint_cosine"])
    return {
        "events": cand[keep].astype(np.int64), "n_candidates": int(cand.size), "n_seeds": int(seeds.sum()),
        "noise_sd_uV": sd, "footprint_channels": neigh.tolist(),
        "footprint": dict(zip((int(c) for c in neigh), np.round(tmpl / tmpl[list(neigh).index(ch)], 3).tolist())),
        "cosine_all": cos, "times_all": t,
    }


def _fill_value(dtype):
    if dtype.kind in "fc":
        return np.nan
    if dtype.kind in "iu":
        return -1
    if dtype.kind == "b":
        return False
    return ""


def extract_units(sorting: si.BaseSorting, rec: si.BaseRecording, cfg: dict, sorter_name: str) -> tuple[si.BaseSorting, dict]:
    rules = [rule_with_defaults(r) for r in cfg["curation"].get("extract_units", [])]
    rules = [r for r in rules if r["sorters"] is None or sorter_name in r["sorters"]]
    if not rules:
        return sorting, {}
    fs = sorting.get_sampling_frequency()
    n = rec.get_num_samples()
    win = int(COINCIDENCE_MS / 1000 * fs)
    trains = {u: sorting.get_unit_spike_train(u).astype(np.int64) for u in sorting.unit_ids}
    keys = sorting.get_property_keys()
    props = {k: dict(zip(sorting.unit_ids, sorting.get_property(k))) for k in keys}
    extracted_as = {u: "" for u in sorting.unit_ids}
    report = {}
    next_id = max(int(u) for u in sorting.unit_ids) + 1
    for r in rules:
        found = find_rule_events(rec, r)
        ev = found["events"]
        absorbed = {}
        for u, st in trains.items():
            if extracted_as.get(u):
                continue
            j = np.searchsorted(st, ev)
            hit_ev, hit_sp = [], []
            for e, jj in zip(ev, j):
                for k in (jj - 1, jj):
                    if 0 <= k < st.size and abs(int(st[k]) - int(e)) <= win:
                        hit_ev.append(e)
                        hit_sp.append(k)
                        break
            c = len(hit_ev)
            chance = ev.size * st.size * (2 * win + 1) / n
            if c >= MIN_ABSORBED and c >= MIN_ABSORBED_FRACTION * ev.size and c >= MIN_ABSORBED_EXCESS * chance:
                trains[u] = np.delete(st, hit_sp)
                absorbed[int(u)] = c
        uid = next_id
        next_id += 1
        trains[uid] = np.sort(ev)
        extracted_as[uid] = r["name"]
        for k in keys:
            if k == "main_channel_id":
                props[k][uid] = rec.channel_ids[int(r["channel"])]
            else:
                props[k][uid] = _fill_value(np.asarray(list(props[k].values())).dtype)
        rep = {"unit_id": uid, "n_spikes": int(ev.size), "n_seeds": found["n_seeds"],
               "n_candidates": found["n_candidates"], "noise_sd_uV": round(found["noise_sd_uV"], 2),
               "footprint": found["footprint"], "removed_from": absorbed,
               "first_last_s": [round(float(ev.min() / fs), 3), round(float(ev.max() / fs), 3)] if ev.size else None}
        report[r["name"]] = rep
        log.info("%s: extracted '%s' as unit %d (%d spikes); removed its spikes from %s",
                 sorter_name, r["name"], uid, ev.size, absorbed or "no unit")
    unit_ids = list(trains)
    new = si.NumpySorting.from_unit_dict({u: trains[u] for u in unit_ids}, sampling_frequency=fs)
    for k in keys:
        new.set_property(k, np.asarray([props[k][u] for u in unit_ids]))
    new.set_property("extracted_as", np.asarray([extracted_as.get(u, "") for u in unit_ids]))
    return new, report
