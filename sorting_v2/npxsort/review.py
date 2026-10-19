"""Local unit-review browser: spike times drawn on the filtered voltage of each unit's peak channels.

The server reads the preprocessed binary on demand, so any stretch of the recording (up to all of
it) can be scrolled without writing per-unit trace files. Everything is served from 127.0.0.1 only.

What it shows per unit: the preprocessed (CatGT-filtered, int16 -> uV) traces of the unit's
highest-amplitude channels with the unit's spikes, ISI violations, the best-matching unit from
the other sorter and nearby units' spikes; the amplitude-over-time overview; the template; the ISI
histogram; and every bombcell criterion with its value, so the reason for each label is visible.
Review verdicts and notes are saved to <output_dir>/review/<sorter>_unit_review.tsv.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import yaml

import spikeinterface.core as si

log = logging.getLogger(__name__)

APP_HTML = Path(__file__).with_name("review_app.html")
N_TEMPLATE_CHANNELS = 8          # channels kept per unit (ranked by template peak-to-peak)
NEIGHBOUR_RADIUS_UM = 60.0       # other units whose peak channel is this close are shown as tick rows
MAX_NEIGHBOURS = 10
ENVELOPE_BIN_S = 0.010           # precomputed min/max envelope used for long windows
MAX_RAW_WINDOW_S = 120.0         # longer windows must use the precomputed envelope
AMP_HALF_WINDOW = 6              # samples (±0.2 ms) searched for each spike's extremum on a channel
REVIEW_COLUMNS = ["unit_id", "verdict", "note", "label_at_review", "reviewed_at"]


# ----------------------------------------------------------------------------- bombcell reasons

def _rule_ok(values: np.ndarray, rule: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit pass flag and NaN flag for one threshold, with bombcell's semantics
    (bounds inclusive; 'abs' applied before comparing)."""
    v = np.abs(values) if rule.get("abs", False) else values
    ok = np.ones(len(v), dtype=bool)
    with np.errstate(invalid="ignore"):
        if rule.get("greater") is not None:
            ok &= v >= rule["greater"]
        if rule.get("less") is not None:
            ok &= v <= rule["less"]
    return ok, np.isnan(v)


SECONDS_METRICS = {"peak_to_trough_duration", "peak_before_width", "trough_width", "peak_after_width"}


def _rule_text(rule: dict, metric: str = "") -> str:
    """Pass condition in words; durations in ms."""
    scale, unit = (1000.0, " ms") if metric in SECONDS_METRICS else (1.0, "")
    parts = []
    name = "|x|" if rule.get("abs", False) else "x"
    if rule.get("greater") is not None:
        parts.append(f"{name} >= {rule['greater'] * scale:g}{unit}")
    if rule.get("less") is not None:
        parts.append(f"{name} <= {rule['less'] * scale:g}{unit}")
    return " and ".join(parts)


def explain_bombcell(table: pd.DataFrame, thresholds: dict) -> tuple[pd.Series, dict]:
    """Re-apply bombcell's labelling rules (SpikeInterface 0.105 bombcell_label_units) and record,
    for every unit, each criterion's value and outcome.

    Order: noise (AND, NaN fails) -> among non-noise, mua (AND, NaN ignored except
    sliding_rp_violation, where NaN fails) -> among non-noise, non_soma overrides good/mua when
    (narrow width AND large ratio) OR large main peak.
    Returns (reproduced labels, {unit_id: [criterion dicts]}).
    """
    n = len(table)
    criteria = {u: [] for u in table.index}

    def run(section: str, metrics: dict, nan_fails: set[str] | bool, group: str | None = None):
        fails = np.zeros(n, dtype=bool)
        for metric, rule in metrics.items():
            if metric not in table.columns or (rule.get("greater") is None and rule.get("less") is None):
                continue
            vals = table[metric].to_numpy(dtype=float)
            ok, nan = _rule_ok(vals, rule)
            nf = nan_fails if isinstance(nan_fails, bool) else metric in nan_fails
            failed = (~ok & ~nan) | (nan & nf)
            fails |= failed
            scale = 1000.0 if metric in SECONDS_METRICS else 1.0
            for i, u in enumerate(table.index):
                criteria[u].append({
                    "section": section, "group": group, "metric": metric,
                    "value": None if nan[i] else float(vals[i]) * scale, "unit": "ms" if scale != 1.0 else "",
                    "rule": _rule_text(rule, metric),
                    "nan_policy": "fail" if nf else "ignore", "failed": bool(failed[i]),
                })
        return fails

    noise_fail = run("noise", thresholds.get("noise", {}), nan_fails=True)
    mua_fail = run("mua", thresholds.get("mua", {}), nan_fails={"sliding_rp_violation"})

    ns = thresholds.get("non-somatic", {})
    width = {m: ns[m] for m in ("peak_before_width", "trough_width") if m in ns}
    ratio = {m: ns[m] for m in ("peak_before_to_trough_ratio", "peak_before_to_peak_after_ratio") if m in ns}
    main = {m: ns[m] for m in ("main_peak_to_trough_ratio",) if m in ns}

    def group_flags(metrics: dict, mode: str) -> np.ndarray:
        """mode 'all_fail': condition true when no valid metric passes (bombcell OR + ignore NaN);
        mode 'any_fail': condition true when any valid metric fails (bombcell AND + ignore NaN)."""
        if not metrics:
            return np.zeros(n, dtype=bool)
        any_ok = np.zeros(n, dtype=bool)
        any_fail = np.zeros(n, dtype=bool)
        for metric, rule in metrics.items():
            if metric not in table.columns:
                continue
            ok, nan = _rule_ok(table[metric].to_numpy(dtype=float), rule)
            any_ok |= ok & ~nan
            any_fail |= ~ok & ~nan
        return ~any_ok if mode == "all_fail" else any_fail

    narrow = group_flags(width, "all_fail")
    large_ratio = group_flags(ratio, "any_fail")
    large_main = group_flags(main, "any_fail")
    non_soma = (narrow & large_ratio) | large_main
    # record the non-somatic metrics too (a "failed" flag here means the metric is in the
    # non-somatic direction; the unit is only non_soma when the compound condition holds)
    for grp, metrics in (("width", width), ("ratio", ratio), ("main_peak", main)):
        run("non-somatic", metrics, nan_fails=False, group=grp)

    labels = np.where(noise_fail, "noise", np.where(non_soma, "non_soma", np.where(mua_fail, "mua", "good")))
    for i, u in enumerate(table.index):
        criteria[u].append({
            "section": "non-somatic", "group": "compound", "metric": "(narrow width AND large ratio) OR large main peak",
            "value": None, "rule": f"narrow={bool(narrow[i])}, large ratio={bool(large_ratio[i])}, large main peak={bool(large_main[i])}",
            "nan_policy": "ignore", "failed": bool(non_soma[i]),
        })
    return pd.Series(labels, index=table.index), criteria


def reason_summary(criteria: list[dict], label: str) -> str:
    if label == "good":
        return ""
    if label == "noise":
        items = [c for c in criteria if c["section"] == "noise" and c["failed"]]
    elif label == "non_soma":
        items = [c for c in criteria if c["section"] == "non-somatic" and c["group"] != "compound" and c["failed"]]
    else:
        items = [c for c in criteria if c["section"] == "mua" and c["failed"]]
    out = []
    for c in items:
        unit = f" {c['unit']}" if c.get("unit") else ""
        out.append(f"{c['metric']}=NaN" if c["value"] is None else f"{c['metric']}={c['value']:.3g}{unit}")
    return "; ".join(out)


# ----------------------------------------------------------------------------- data

class SorterData:
    """Everything the browser needs for one sorter, held in memory."""

    def __init__(self, name: str, out_dir: Path, thresholds_fallback: dict | None):
        self.name = name
        folder = out_dir / name
        self.table = pd.read_csv(folder / "units.tsv", sep="\t", index_col="unit_id")
        thr_file = folder / "bombcell" / "thresholds_used.json"
        self.thresholds = json.loads(thr_file.read_text()) if thr_file.exists() else (thresholds_fallback or {})

        t = time.time()
        analyzer = si.load_sorting_analyzer(folder / "analyzer", load_extensions=False)
        self.unit_ids = list(analyzer.unit_ids)
        templates_ext = analyzer.get_extension("templates")
        self.nbefore = int(templates_ext.nbefore)
        dense = templates_ext.get_data(operator="average").astype("float32")   # (units, samples, channels)
        ptp = dense.max(axis=1) - dense.min(axis=1)                             # (units, channels)
        self.top_channels = np.argsort(-ptp, axis=1)[:, :N_TEMPLATE_CHANNELS]
        self.ptp = ptp
        self.dense_templates = dense  # kept so matched units can be drawn on another unit's channels

        spikes = analyzer.sorting.to_spike_vector()
        order = np.argsort(spikes["unit_index"], kind="stable")
        bounds = np.searchsorted(spikes["unit_index"][order], np.arange(len(self.unit_ids) + 1))
        self.spikes: dict[int, np.ndarray] = {}
        for i, u in enumerate(self.unit_ids):
            s = spikes["sample_index"][order[bounds[i]:bounds[i + 1]]]
            self.spikes[u] = np.sort(s).astype(np.int64)
        self.index_of = {u: i for i, u in enumerate(self.unit_ids)}

        # Two different "peak" channels:
        #  main     - the analyzer's main channel, on which every metric and bombcell rule was computed.
        #             For Kilosort it is the largest peak-to-peak channel of Kilosort's *whitened*
        #             templates (SpikeInterface phykilosortextractors), which can differ from:
        #  largest  - the largest peak-to-peak channel of the mean template in µV. The browser plots
        #             amplitudes on this one by default.
        main = np.asarray(analyzer.get_main_channels(outputs="index"), dtype=int)
        self.main_channel = dict(zip(self.unit_ids, main))
        self.largest_channel = dict(zip(self.unit_ids, ptp.argmax(axis=1).astype(int)))
        self.polarity = {}  # on the main channel, matching bombcell's view of the unit
        for i, u in enumerate(self.unit_ids):
            w = dense[i, :, main[i]]
            self.polarity[u] = 1 if w.max() > -w.min() else -1

        reproduced, self.criteria = explain_bombcell(self.table.reindex(self.unit_ids), self.thresholds)
        stored = self.table["label"].reindex(self.unit_ids)
        mismatch = int((reproduced != stored).sum())
        if mismatch:
            log.warning("%s: bombcell reproduction disagrees with units.tsv for %d units; the stored label is shown.", name, mismatch)
        self.reasons = {u: reason_summary(self.criteria[u], stored[u]) for u in self.unit_ids}
        log.info("%s: %d units loaded in %.1f s", name, len(self.unit_ids), time.time() - t)

        self.review_file = out_dir / "review" / f"{name}_unit_review.tsv"
        self.reviews = self._load_reviews()

    def _load_reviews(self) -> dict:
        if not self.review_file.exists():
            return {}
        df = pd.read_csv(self.review_file, sep="\t", dtype=str, keep_default_na=False)
        return {int(r["unit_id"]): dict(r) for _, r in df.iterrows()}

    def save_review(self, unit: int, verdict: str, note: str) -> dict:
        row = {
            "unit_id": str(unit), "verdict": verdict, "note": note.replace("\t", " ").replace("\n", " "),
            "label_at_review": str(self.table.at[unit, "label"]), "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }
        if verdict == "" and note == "":
            self.reviews.pop(unit, None)
        else:
            self.reviews[unit] = row
        self.review_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(sorted(self.reviews.values(), key=lambda r: int(r["unit_id"])), columns=REVIEW_COLUMNS)
        tmp = self.review_file.with_suffix(".tmp")
        df.to_csv(tmp, sep="\t", index=False)
        tmp.replace(self.review_file)
        return row


class ReviewData:
    def __init__(self, out_dir: Path, scratch_dir: Path | None = None, time_offset_s: float = 0.0):
        self.out_dir = Path(out_dir)
        cfg = yaml.safe_load((self.out_dir / "config_used.yaml").read_text())
        scratch = Path(scratch_dir) if scratch_dir else Path(cfg["paths"]["scratch_dir"])
        pre = scratch / "preprocessed"
        if not pre.exists():
            raise FileNotFoundError(f"Preprocessed binary not found at {pre} (pass --scratch-dir if it moved).")
        self.recording = si.load(pre)
        self.fs = float(self.recording.get_sampling_frequency())
        self.n_samples = int(self.recording.get_num_samples())
        self.n_channels = int(self.recording.get_num_channels())
        self.gain = self.recording.get_channel_gains().astype("float32")
        self.offset = self.recording.get_channel_offsets().astype("float32")
        self.locations = self.recording.get_channel_locations()
        self.channel_ids = [str(c) for c in self.recording.channel_ids]
        info = json.loads((pre / "binary.json").read_text())["kwargs"]
        self.traces = np.memmap(pre / info["file_paths"][0], dtype=info["dtype"], mode="r",
                                offset=int(info.get("file_offset", 0) or 0)).reshape(-1, self.n_channels)
        self.time_offset_s = float(time_offset_s)
        self._amp_cache: dict = {}
        self._amp_lock = threading.Lock()

        names = [n for n, s in cfg["sorters"].items() if s.get("enabled") and (self.out_dir / n / "units.tsv").exists()]
        self.sorters = {n: SorterData(n, self.out_dir, None) for n in names}

        self.agreement = None
        agr = self.out_dir / "comparison" / "agreement_scores.csv"
        if agr.exists() and len(names) >= 2:
            df = pd.read_csv(agr, index_col=0)
            df.columns = [int(c) for c in df.columns]
            self.agreement = (names[0], names[1], df)

        self.envelope = None  # (min, max) int16 arrays, shape (bins, channels)
        self.env_bin = int(round(ENVELOPE_BIN_S * self.fs))
        self.env_cache = scratch / "review_cache" / f"envelope_{self.env_bin}.npz"
        threading.Thread(target=self._build_envelope, daemon=True).start()

    # --- envelope for long windows -------------------------------------------------------
    def _build_envelope(self):
        try:
            if self.env_cache.exists():
                z = np.load(self.env_cache)
                self.envelope = (z["mn"], z["mx"])
                log.info("Loaded trace envelope cache %s", self.env_cache)
                return
            t = time.time()
            b = self.env_bin
            n_bins = self.n_samples // b
            mn = np.empty((n_bins, self.n_channels), dtype=np.int16)
            mx = np.empty_like(mn)
            step = 1000  # bins per chunk (10 s)
            for i0 in range(0, n_bins, step):
                i1 = min(i0 + step, n_bins)
                block = np.asarray(self.traces[i0 * b:i1 * b]).reshape(i1 - i0, b, self.n_channels)
                mn[i0:i1] = block.min(axis=1)
                mx[i0:i1] = block.max(axis=1)
            self.env_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.env_cache, mn=mn, mx=mx)
            self.envelope = (mn, mx)
            log.info("Built trace envelope (%d-sample bins) in %.0f s; cached at %s", b, time.time() - t, self.env_cache)
        except Exception:
            log.exception("Could not build the trace envelope; long windows will be read directly.")

    # --- API payloads --------------------------------------------------------------------
    def meta(self) -> dict:
        return {
            "fs": self.fs, "n_samples": self.n_samples, "duration_s": self.n_samples / self.fs,
            "time_offset_s": self.time_offset_s, "sorters": list(self.sorters),
            "experiment": self.out_dir.name, "output_dir": str(self.out_dir),
            "thresholds": {n: s.thresholds for n, s in self.sorters.items()},
            "envelope_ready": self.envelope is not None,
        }

    def best_match(self, sorter: str, unit: int):
        if self.agreement is None:
            return None
        a, b, df = self.agreement
        if sorter == a and unit in df.index:
            row = df.loc[unit]
            other = b
        elif sorter == b and unit in df.columns:
            row = df[unit]
            other = a
        else:
            return None
        if row.max() <= 0:
            return None
        return other, int(row.idxmax()), float(row.max())

    def units(self, sorter: str) -> list[dict]:
        s = self.sorters[sorter]
        t = s.table
        rows = []
        for u in s.unit_ids:
            r = t.loc[u]
            bm = self.best_match(sorter, u)
            rows.append({
                "id": int(u), "label": r["label"], "depth": float(r["depth_um"]), "x": float(r["x_um"]),
                "peak_channel": int(r["peak_channel_index"]), "num_spikes": int(r["num_spikes"]),
                "firing_rate": float(r["firing_rate"]), "snr": _f(r.get("snr")), "amp": _f(r.get("amplitude_median")),
                "found_by_both": bool(r.get("found_by_both")) if pd.notna(r.get("found_by_both")) else False,
                "best_match": None if bm is None else {"sorter": bm[0], "id": bm[1], "agreement": bm[2]},
                "polarity": s.polarity[u], "reasons": s.reasons[u],
                "failed": sorted({c["metric"] for c in s.criteria[u] if c["failed"] and c["group"] != "compound"}),
                "verdict": s.reviews.get(u, {}).get("verdict", ""),
            })
        return rows

    def unit(self, sorter: str, unit: int) -> dict:
        s = self.sorters[sorter]
        i = s.index_of[unit]
        chans = s.top_channels[i]
        r = s.table.loc[unit]
        neighbours = []
        ploc = self.locations[chans[:3]]
        for v in s.unit_ids:
            if v == unit:
                continue
            d = float(np.min(np.linalg.norm(ploc - self.locations[s.largest_channel[v]], axis=1)))
            if d <= NEIGHBOUR_RADIUS_UM:
                neighbours.append({"id": int(v), "label": s.table.at[v, "label"], "distance_um": d,
                                   "num_spikes": int(s.table.at[v, "num_spikes"]), "polarity": s.polarity[v]})
        neighbours = sorted(neighbours, key=lambda x: x["distance_um"])[:MAX_NEIGHBOURS]

        match = None
        bm = self.best_match(sorter, unit)
        if bm is not None:
            o = self.sorters[bm[0]]
            oi = o.index_of.get(bm[1])
            if oi is not None:
                match = {
                    "sorter": bm[0], "id": bm[1], "agreement": bm[2], "label": o.table.at[bm[1], "label"],
                    "found_by_both": bool(r.get("found_by_both")) if pd.notna(r.get("found_by_both")) else False,
                    "spikes": o.spikes[bm[1]].tolist(),
                    "template": o.dense_templates[oi][:, chans].T.round(2).tolist(),
                }
        metric_cols = [c for c in s.table.columns if c not in ("label", "peak_channel_id")]
        w = s.dense_templates[i]
        def chan_info(c):
            return {"index": int(c), "id": self.channel_ids[c], "x": float(self.locations[c, 0]),
                    "y": float(self.locations[c, 1]), "ptp": float(s.ptp[i, c]),
                    "min": float(w[:, c].min()), "max": float(w[:, c].max())}
        return {
            "sorter": sorter, "id": int(unit), "label": r["label"], "reasons": s.reasons[unit],
            "criteria": s.criteria[unit], "polarity": s.polarity[unit],
            "main_channel": chan_info(s.main_channel[unit]), "largest_channel": chan_info(s.largest_channel[unit]),
            "channels": [chan_info(c) for c in chans],
            "template": w[:, chans].T.round(2).tolist(), "nbefore": s.nbefore,
            "spikes": s.spikes[unit].tolist(),
            "metrics": {c: _jsonable(r[c]) for c in metric_cols},
            "neighbours": neighbours, "match": match,
            "review": s.reviews.get(unit, {"verdict": "", "note": ""}),
        }

    def channel_amplitudes(self, sorter: str, unit: int, ch: int) -> tuple[np.ndarray, int]:
        """Each spike's amplitude (µV) on one channel: the extremum, of the template's sign on that
        channel, within ±AMP_HALF_WINDOW samples of the template's extremum time. Cached."""
        key = (sorter, unit, ch)
        with self._amp_lock:
            if key in self._amp_cache:
                return self._amp_cache[key]
        s = self.sorters[sorter]
        w = s.dense_templates[s.index_of[unit]][:, ch]
        k = int(np.argmax(np.abs(w)))
        pol = 1 if w[k] > 0 else -1
        sp = s.spikes[unit] + (k - s.nbefore)
        win = np.arange(-AMP_HALF_WINDOW, AMP_HALF_WINDOW + 1)
        out = np.empty(sp.size, dtype=np.float32)
        for a in range(0, sp.size, 20000):
            idx = np.clip(sp[a:a + 20000, None] + win[None, :], 0, self.n_samples - 1)
            seg = np.asarray(self.traces[idx, ch])
            out[a:a + 20000] = seg.max(axis=1) if pol > 0 else seg.min(axis=1)
        out = out * self.gain[ch] + self.offset[ch]
        with self._amp_lock:
            if len(self._amp_cache) >= 128:
                self._amp_cache.pop(next(iter(self._amp_cache)))
            self._amp_cache[key] = (out, pol)
        return out, pol

    def window_spikes(self, sorter: str, ids: list[int], start: int, end: int) -> dict:
        s = self.sorters[sorter]
        out = {}
        for u in ids:
            sp = s.spikes.get(u)
            if sp is None:
                continue
            a, b = np.searchsorted(sp, [start, end])
            out[str(u)] = sp[a:b].tolist()
        return out

    def traces_payload(self, channels: list[int], start: int, end: int, px: int) -> tuple[bytes, dict]:
        """Float32 uV. Raw samples when the window is short enough for the canvas, otherwise a
        per-pixel (min, max) envelope."""
        start = max(0, start)
        end = min(self.n_samples, max(end, start + 1))
        n = end - start
        ch = np.asarray(channels, dtype=int)
        gain, off = self.gain[ch], self.offset[ch]
        px = max(50, min(px, 8000))
        if n <= 3 * px:
            data = np.asarray(self.traces[start:end, ch], dtype=np.float32) * gain + off
            return np.ascontiguousarray(data.T).tobytes(), {"mode": "raw", "count": n, "start": start, "end": end}

        spp = n / px
        if self.envelope is not None and spp >= self.env_bin:
            b0, b1 = start // self.env_bin, max(start // self.env_bin + 1, end // self.env_bin)
            mn = self.envelope[0][b0:b1, ch]
            mx = self.envelope[1][b0:b1, ch]
            edges = np.linspace(0, b1 - b0, px + 1).astype(int)
            edges[-1] = b1 - b0
            mn = np.minimum.reduceat(mn, edges[:-1], axis=0)
            mx = np.maximum.reduceat(mx, edges[:-1], axis=0)
        else:
            if n > MAX_RAW_WINDOW_S * self.fs:
                end = start + int(MAX_RAW_WINDOW_S * self.fs)
                n = end - start
            raw = np.asarray(self.traces[start:end, ch])
            edges = np.linspace(0, n, px + 1).astype(int)
            mn = np.minimum.reduceat(raw, edges[:-1], axis=0)
            mx = np.maximum.reduceat(raw, edges[:-1], axis=0)
        env = np.stack([mn.astype(np.float32) * gain + off, mx.astype(np.float32) * gain + off], axis=2)  # (px, ch, 2)
        return np.ascontiguousarray(env.transpose(1, 0, 2)).tobytes(), {"mode": "minmax", "count": int(env.shape[0]), "start": start, "end": end}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _jsonable(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(v)


# ----------------------------------------------------------------------------- HTTP

def make_handler(data: ReviewData):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, body: bytes, ctype: str, headers: dict | None = None, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(json.dumps(obj, allow_nan=False, default=_jsonable).encode(), "application/json", status=status)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(APP_HTML.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/meta":
                    self._json(data.meta())
                elif url.path == "/api/units":
                    self._json(data.units(q["sorter"]))
                elif url.path == "/api/unit":
                    self._json(data.unit(q["sorter"], int(q["unit"])))
                elif url.path == "/api/window":
                    ids = [int(x) for x in q.get("ids", "").split(",") if x]
                    self._json(data.window_spikes(q["sorter"], ids, int(q["start"]), int(q["end"])))
                elif url.path == "/api/amplitudes":
                    amps, pol = data.channel_amplitudes(q["sorter"], int(q["unit"]), int(q["ch"]))
                    self._send(amps.astype("<f4").tobytes(), "application/octet-stream",
                               {"X-Channel": int(q["ch"]), "X-Polarity": pol})
                elif url.path == "/api/traces":
                    chans = [int(x) for x in q["ch"].split(",")]
                    body, h = data.traces_payload(chans, int(q["start"]), int(q["end"]), int(q.get("px", 1500)))
                    self._send(body, "application/octet-stream", {f"X-{k.title()}": v for k, v in h.items()})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as e:
                self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            except Exception as e:  # surface server errors in the page instead of hanging
                log.exception("Request failed: %s", self.path)
                self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if url.path == "/api/review":
                    s = data.sorters[body["sorter"]]
                    self._json(s.save_review(int(body["unit"]), str(body.get("verdict", "")), str(body.get("note", ""))))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as e:
                log.exception("POST failed: %s", self.path)
                self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(data: ReviewData, port: int = 8765) -> ThreadingHTTPServer:
    last_error = None
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), make_handler(data))
        except OSError as e:
            last_error = e
    raise OSError(f"No free port in {port}-{port + 19}: {last_error}")
