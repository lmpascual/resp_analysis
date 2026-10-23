"""Remove hand-picked spikes from a sorted unit (config: curation.remove_spikes).

Sometimes a unit is right except for a few spikes that belong to another neuron, e.g. a burst of a
smaller neighbour that the sorter also assigned to another unit. A rule removes the unit's spikes in
a time window, optionally only those the review browser flags (npxsort.review):

    curation:
      remove_spikes:
        - sorter: kilosort4
          unit_id: 435                 # id in units.tsv / the review browser (not Phy's cluster_id)
          start_s: 242.5               # window, s from the start of the sorted recording (as in ALF
          end_s: 242.8                 # spike_times.npy and the review browser); start <= t < end
          reason: "6 spikes of a smaller neighbour; 5 also in #428"
          only_flagged: null           # null = every spike in the window; or a list of
                                       # footprint_mismatch | duplicate: only spikes with one of these flags
          expect_removed: 6            # optional: stop the run if a different number would be removed

Rules run after duplicate censoring and redundant-unit removal, on the units that end up in units.tsv,
so unit ids don't change. Flags are computed on the sorting before any rule, the way the review browser
computes them on that sorting: footprint mismatch (pattern across the unit's largest channels differs
from its template, or a polarity reversal) and duplicate (another nearby unit has a spike within
0.1 ms, for a unit pair that coincides far more often than chance). A unit left with no spikes is dropped.

Removed spikes are listed in <sorter>/removed_spikes.tsv and counted per rule in
postprocessing_report.json; units.tsv gets a `removed_spikes` column.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import spikeinterface.core as si

from .review import (DUPLICATE_FLANK_MS, DUPLICATE_MS, DUPLICATE_RADIUS_UM, FOOTPRINT_MIN_SIGMA, FOOTPRINT_MISMATCH,
                     cross_unit_duplicates)
from .spikefeatures import footprint_setup, noise_levels, polarity_reversed, spike_pass

log = logging.getLogger(__name__)

FLAGS = ("footprint_mismatch", "duplicate")
RULE_DEFAULTS = {"only_flagged": None, "expect_removed": None}


def rules_for(cfg: dict, sorter_name: str) -> list[tuple[int, dict]]:
    """(index in curation.remove_spikes, rule with defaults) for one sorter."""
    return [(i, {**RULE_DEFAULTS, **r}) for i, r in enumerate(cfg["curation"].get("remove_spikes", []))
            if r["sorter"] == sorter_name]


class _StoredTraces:
    """Slices of the recording's stored (unscaled) samples, as spikefeatures reads them from a memmap."""

    def __init__(self, rec: si.BaseRecording):
        self.rec = rec

    def __getitem__(self, s: slice) -> np.ndarray:
        return self.rec.get_traces(start_frame=s.start, end_frame=s.stop)


def review_flags(sorting: si.BaseSorting, rec: si.BaseRecording, cfg: dict, unit_ids, job_kwargs: dict) -> dict:
    """The review browser's per-spike flags for `unit_ids`, computed on this sorting.

    Templates come from a temporary in-memory analyzer with the same sparsity, random spikes and window
    as the postprocessing analyzer, so they match what the browser loads for this sorting.
    Returns {unit_id: {"footprint_mismatch": bool array, "duplicate": bool array, "partner": unit ids
    (-1 = none)}}, each over the unit's spike train in time order.
    """
    pp = cfg["postprocessing"]
    analyzer = si.create_sorting_analyzer(
        sorting, rec, format="memory", sparse=True,
        sparsity_kwargs={"method": "radius", "radius_um": pp["sparsity_radius_um"]}, seed=pp["seed"], job_kwargs=job_kwargs,
    )
    analyzer.compute({
        "random_spikes": {"method": "uniform", "max_spikes_per_unit": pp["max_spikes_per_unit"], "seed": pp["seed"]},
        "templates": {"ms_before": pp["ms_before"], "ms_after": pp["ms_after"], "operators": ["average"]},
    }, **job_kwargs)
    ext = analyzer.get_extension("templates")
    dense = ext.get_data(operator="average").astype("float32")
    nbefore = int(ext.nbefore)
    del analyzer, ext
    ptp = dense.max(axis=1) - dense.min(axis=1)

    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    traces = _StoredTraces(rec)
    gain = rec.get_channel_gains().astype("float32")
    offset = rec.get_channel_offsets().astype("float32")
    spikes = sorting.to_spike_vector()
    sample = spikes["sample_index"].astype(np.int64)
    unit = spikes["unit_index"].astype(np.int64)
    ids = list(sorting.unit_ids)
    targets = [ids.index(u) for u in unit_ids]
    sel = np.isin(unit, targets)

    setup = footprint_setup(dense, ptp, nbefore)
    noise = noise_levels(traces, gain, n, fs)
    res = spike_pass(traces, gain, offset, noise, n, fs, {"s": setup}, {"s": (sample[sel], unit[sel])})["s"]
    mismatch = (((res["rel"] > FOOTPRINT_MISMATCH) & (res["sig"] > FOOTPRINT_MIN_SIGMA))
                | polarity_reversed(res["opp_ratio"], res["opp_sigma"], setup[3][unit[sel]]))

    loc = rec.get_channel_locations()[ptp.argmax(axis=1)]
    near = np.linalg.norm(loc[:, None] - loc[None, :], axis=2) <= DUPLICATE_RADIUS_UM
    center = int(round(DUPLICATE_MS / 1000 * fs))
    flank = tuple(int(round(v / 1000 * fs)) for v in DUPLICATE_FLANK_MS)
    partner, _ = cross_unit_duplicates(sample, unit, near, center, flank)
    partner = partner[sel]

    out = {}
    for u, k in zip(unit_ids, targets):
        mine = unit[sel] == k
        p = partner[mine]
        out[u] = {"footprint_mismatch": mismatch[mine], "duplicate": p >= 0,
                  "partner": np.where(p >= 0, np.asarray(ids)[np.maximum(p, 0)], -1)}
    return out


def remove_spikes(sorting: si.BaseSorting, rec: si.BaseRecording, cfg: dict, sorter_name: str,
                  job_kwargs: dict) -> tuple[si.BaseSorting, list, pd.DataFrame]:
    """Apply this sorter's curation.remove_spikes rules. Returns (sorting, per-rule report, removed spikes)."""
    rules = rules_for(cfg, sorter_name)
    if not rules:
        return sorting, [], pd.DataFrame()
    fs = sorting.get_sampling_frequency()
    ids = list(sorting.unit_ids)
    for i, r in rules:
        if r["unit_id"] not in ids:
            raise ValueError(f"curation.remove_spikes[{i}]: {sorter_name} has no unit {r['unit_id']} after cleaning "
                             "(removed as redundant, or the ids changed because the sort was redone?).")
    trains = {u: sorting.get_unit_spike_train(u).astype(np.int64) for u in ids}
    flagged = sorted({r["unit_id"] for _, r in rules if r["only_flagged"]})
    flags = review_flags(sorting, rec, cfg, flagged, job_kwargs) if flagged else {}

    removed = {u: np.zeros(trains[u].size, dtype=bool) for u in {r["unit_id"] for _, r in rules}}
    report, rows = [], []
    for i, r in rules:
        u = r["unit_id"]
        st = trains[u]
        hit = (st >= r["start_s"] * fs) & (st < r["end_s"] * fs)
        n_window = int(hit.sum())
        if r["only_flagged"]:
            hit &= np.logical_or.reduce([flags[u][f] for f in r["only_flagged"]])
        new = hit & ~removed[u]
        n_new = int(new.sum())
        if r["expect_removed"] is not None and n_new != r["expect_removed"]:
            raise ValueError(f"curation.remove_spikes[{i}]: would remove {n_new} spikes from {sorter_name} unit {u} "
                             f"in {r['start_s']}-{r['end_s']} s, but expect_removed is {r['expect_removed']}. "
                             "Check the unit in the review browser before changing the rule.")
        removed[u] |= new
        rep = {"rule": i, "unit_id": int(u), "start_s": r["start_s"], "end_s": r["end_s"], "reason": r["reason"],
               "only_flagged": r["only_flagged"], "n_spikes_in_window": n_window, "n_removed": n_new}
        if u in flags:
            rep["n_flagged_in_window"] = {f: int((flags[u][f] & (st >= r["start_s"] * fs) & (st < r["end_s"] * fs)).sum())
                                          for f in FLAGS}
            partners = flags[u]["partner"][new]
            rep["duplicate_of"] = {int(p): int((partners == p).sum()) for p in np.unique(partners[partners >= 0])}
        report.append(rep)
        for j in np.nonzero(new)[0]:
            row = {"unit_id": int(u), "sample_index": int(st[j]), "time_s": round(st[j] / fs, 6), "rule": i,
                   "reason": r["reason"]}
            if u in flags:
                row.update({f: bool(flags[u][f][j]) for f in FLAGS}, duplicate_of=int(flags[u]["partner"][j]))
            rows.append(row)
        log.info("%s: remove_spikes[%d] removed %d of %d spikes of unit %s in %s-%s s (%s)", sorter_name, i, n_new,
                 n_window, u, r["start_s"], r["end_s"], "flags " + "/".join(r["only_flagged"]) if r["only_flagged"] else "all")

    counts = {u: int(m.sum()) for u, m in removed.items()}
    for rep in report:
        rep["n_spikes_left"] = int(trains[rep["unit_id"]].size - counts[rep["unit_id"]])
    keep = [u for u in ids if trains[u].size - counts.get(u, 0) > 0]
    for u in set(ids) - set(keep):
        log.warning("%s: remove_spikes left unit %s with no spikes; it is dropped.", sorter_name, u)
    new_sorting = si.NumpySorting.from_unit_dict({u: trains[u][~removed[u]] if u in removed else trains[u] for u in keep},
                                                 sampling_frequency=fs)
    pos = [ids.index(u) for u in keep]
    for k in sorting.get_property_keys():
        new_sorting.set_property(k, np.asarray(sorting.get_property(k))[pos])
    new_sorting.set_property("removed_spikes", np.asarray([counts.get(u, 0) for u in keep], dtype=np.int64))
    columns = ["unit_id", "sample_index", "time_s", "rule", "reason"] + ([*FLAGS, "duplicate_of"] if flags else [])
    return new_sorting, report, pd.DataFrame(rows, columns=columns)
