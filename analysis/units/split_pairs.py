"""Candidate pairs of units that may be one neuron split in two by the sorter, for review by eye (nothing is merged).

e038 anoxia #1 (2026-10-30): Kilosort 4 cut several neurons in two at the anoxic gasps (#20 -> #22, #25 -> #26, ...): one
unit stops on the breath the other starts, at the same site, with near-identical templates and the same rate.
Left unmerged, such pairs read as one unit silenced and another recruited by anoxia, which is what the analyses ask
about. Kilosort's merge suggestions (merge_suggestions.csv) caught none of them.

Candidates come from templates and timing, not from firing patterns (rates, phase tuning: the things the analyses
compare):
  same site    template cosine >= MIN_COS on the first unit's 12 largest channels, best over shifts of -1, 0, +1 rows
               (NP1.0 rows are 20 um), template centres within MAX_DIST_UM
  handover     the units are rarely active at the same time: of the 5 s bins in which the less active one fires, at
               most MAX_OVERLAP also have the other firing (a unit is active in a bin if it fires >= 10 % of its
               typical rate there, the 90th percentile of its active bins)
plus every pair in the run's merge_suggestions.csv. Ranked by cosine.

Per window (<output>/split_review/): split_candidates.tsv with a blank `your_verdict` column (merge / separate /
unsure) and, per pair, a figure:
  templates of both units drawn at their channel positions;
  firing rate (5 s bins) of both through the window, with the config's epochs shaded;
  spike amplitude and spike depth (SpikeInterface spike_amplitudes / spike_locations) of both through the window:
  continuous across the handover = one neuron whose waveform or position changed; a jump in depth = tissue movement;
  autocorrelograms of both, and their cross-correlogram (a gap at 0 ms when they do overlap = refractory = one neuron).
Each unit can also be checked on the traces: python sorting/review_units.py <output_dir> --time-offset-s <run start>.

    python split_pairs.py <outputs>/grp/e038/anoxia1 [<more output dirs>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

N_CH = 12
MIN_COS = 0.80
MAX_DIST_UM = 60.0
BIN_S = 5.0
ACTIVE_FRAC = 0.10
MAX_OVERLAP = 0.25
MIN_SPIKES = 50
CCG_MS = 20


def cosine(a, b):
    n = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / n) if n > 0 else 0.0


def active_bins(t, n_bins):
    c = np.bincount(np.minimum((t / BIN_S).astype(int), n_bins - 1), minlength=n_bins).astype(float)
    nz = c[c > 0]
    typ = np.percentile(nz, 90) if nz.size else 0.0
    return c, c >= max(1.0, ACTIVE_FRAC * typ)


def ccg(a, b, half_ms=CCG_MS, bin_ms=0.5):
    edges = np.arange(-half_ms, half_ms + bin_ms, bin_ms) / 1e3
    if not a.size or not b.size:
        return edges, np.zeros(edges.size - 1)
    k0 = np.searchsorted(b, a + edges[0])
    k1 = np.searchsorted(b, a + edges[-1])
    parts = [b[i:j] - t for i, j, t in zip(k0, k1, a) if j > i]
    d = np.concatenate(parts) if parts else np.array([])
    if a is b:
        d = d[d != 0]                     # autocorrelogram: drop each spike paired with itself
    return edges, np.histogram(d, edges)[0]


class Run:
    def __init__(self, out_dir: Path, sorter: str = "kilosort4"):
        import spikeinterface.core as si
        self.dir = out_dir
        self.cfg = yaml.safe_load((out_dir / "config_used.yaml").read_text())
        self.offset = float(self.cfg["nidq"]["file_start_s"]) + float(self.cfg["input"].get("start_s") or 0.0)
        sd = out_dir / sorter
        self.units = pd.read_csv(sd / "units.tsv", sep="\t").set_index("unit_id")
        an = si.load_sorting_analyzer(sd / "analyzer", load_extensions=False)
        self.fs = an.sampling_frequency
        self.dur = an.get_num_samples() / self.fs
        self.ids = list(an.unit_ids)
        self.idx = {u: i for i, u in enumerate(self.ids)}
        self.T = an.get_extension("templates").get_data(operator="average")
        self.locs = an.get_channel_locations()
        sv = an.sorting.to_spike_vector()
        amp = an.get_extension("spike_amplitudes").get_data()
        loc = an.get_extension("spike_locations").get_data()
        order = np.argsort(sv["unit_index"], kind="stable")
        bounds = np.searchsorted(sv["unit_index"][order], np.arange(len(self.ids) + 1))
        self.spk, self.amp, self.y = {}, {}, {}
        for i, u in enumerate(self.ids):
            o = order[bounds[i]:bounds[i + 1]]
            self.spk[u] = sv["sample_index"][o] / self.fs
            self.amp[u] = np.abs(amp[o])
            self.y[u] = loc["y"][o]
        ptp = self.T.max(axis=1) - self.T.min(axis=1)
        self.top = np.sort(np.argsort(-ptp, axis=1)[:, :N_CH], axis=1)
        w = np.take_along_axis(ptp, self.top, axis=1)
        self.centre = (self.locs[self.top, 1] * w).sum(1) / np.maximum(w.sum(1), 1e-9)
        self.ks_suggest = set()
        f = sd / "merge_suggestions.csv"
        if f.exists():
            for g in pd.read_csv(f)["unit_ids"]:
                us = [int(x) for x in str(g).split()]
                self.ks_suggest |= {tuple(sorted(p)) for p in zip(us, us[1:])}

    def shifted_cos(self, u, v):
        i, j = self.idx[u], self.idx[v]
        ch = self.top[i]
        best, bk = -1.0, 0
        for k in (-1, 0, 1):
            c2 = ch + 2 * k
            if c2.min() < 0 or c2.max() >= self.T.shape[2]:
                continue
            c = cosine(self.T[i][:, ch], self.T[j][:, c2])
            if c > best:
                best, bk = c, k
        return best, bk * 20.0


def candidates(R: Run):
    keep = [u for u in R.ids if R.units.loc[u, "label"] != "noise" and R.spk[u].size >= MIN_SPIKES]
    n_bins = int(np.ceil(R.dur / BIN_S))
    act = {u: active_bins(R.spk[u], n_bins) for u in keep}
    rows = []
    for a_i, u in enumerate(keep):
        for v in keep[a_i + 1:]:
            pair = tuple(sorted((u, v)))
            ks = pair in R.ks_suggest
            if abs(R.centre[R.idx[u]] - R.centre[R.idx[v]]) > MAX_DIST_UM and not ks:
                continue
            c, sh = R.shifted_cos(u, v)
            au, av = act[u][1], act[v][1]
            n_small = min(au.sum(), av.sum())
            overlap = (au & av).sum() / max(n_small, 1)
            if not ks and (c < MIN_COS or overlap > MAX_OVERLAP):
                continue
            # earlier unit first: by the median time of its active bins
            first, second = (u, v) if np.median(np.flatnonzero(au)) <= np.median(np.flatnonzero(av)) else (v, u)
            a1, a2 = act[first][1], act[second][1]
            last1 = np.flatnonzero(a1).max() * BIN_S + BIN_S
            first2 = np.flatnonzero(a2).min() * BIN_S
            r1 = R.spk[first].size / max(a1.sum() * BIN_S, BIN_S)
            r2 = R.spk[second].size / max(a2.sum() * BIN_S, BIN_S)
            rows.append({"window": R.dir.name, "unit_first": int(first), "unit_second": int(second),
                         "template_cosine": round(c, 4), "shift_um": sh,
                         "centre_um_first": round(float(R.centre[R.idx[first]]), 1),
                         "centre_um_second": round(float(R.centre[R.idx[second]]), 1),
                         "label_first": R.units.loc[first, "label"], "label_second": R.units.loc[second, "label"],
                         "n_spikes_first": int(R.spk[first].size), "n_spikes_second": int(R.spk[second].size),
                         "overlap_frac": round(float(overlap), 3),
                         "first_last_active_run_s": round(last1 + R.offset, 1),
                         "second_first_active_run_s": round(first2 + R.offset, 1),
                         "rate_when_active_first_hz": round(r1, 2), "rate_when_active_second_hz": round(r2, 2),
                         "median_amp_first_uv": round(float(np.median(R.amp[first])), 1),
                         "median_amp_second_uv": round(float(np.median(R.amp[second])), 1),
                         "kilosort_merge_suggestion": ks, "your_verdict": "", "note": ""})
    return pd.DataFrame(rows).sort_values("template_cosine", ascending=False).reset_index(drop=True) if rows else \
        pd.DataFrame(rows)


def figure(R: Run, r, path):
    u, v = r["unit_first"], r["unit_second"]
    cu, cv = "#2a78d6", "#d93636"
    fig = plt.figure(figsize=(17, 8.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.3, 2.4, 0.9, 0.9], hspace=0.45, wspace=0.3)
    # templates at channel positions
    ax = fig.add_subplot(gs[:, 0])
    i, j = R.idx[u], R.idx[v]
    ch = R.top[i]
    tu, tv = R.T[i][:, ch], R.T[j][:, ch]
    scale = 25.0 / max(np.abs(tu).max(), np.abs(tv).max(), 1e-9)
    n = tu.shape[0]
    xs = np.linspace(-10, 10, n)
    for k, c in enumerate(ch):
        x0, y0 = R.locs[c]
        ax.plot(x0 + xs, y0 + tu[:, k] * scale, color=cu, lw=1.2)
        ax.plot(x0 + xs, y0 + tv[:, k] * scale, color=cv, lw=1.0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("depth (um)")
    ax.set_title(f"templates on #{u}'s channels\ncos {r['template_cosine']:.3f} (shift {r['shift_um']:+.0f} um)",
                 fontsize=9)
    # rate, amplitude, depth through the window
    edges = np.arange(0, R.dur + BIN_S, BIN_S)
    ep = R.cfg.get("epochs") or []
    for row, (lab, fn) in enumerate((("rate (Hz, 5 s bins)", None), ("spike amplitude (uV)", "amp"),
                                     ("spike depth (um)", "y"))):
        ax = fig.add_subplot(gs[row, 1])
        for e in ep:
            ax.axvspan(e["start_s"] + R.offset, e["end_s"] + R.offset, color="0.93", lw=0)
            if row == 0:
                ax.text((e["start_s"] + e["end_s"]) / 2 + R.offset, 1.0, e["name"], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=6, color="0.4")
        for w, col in ((u, cu), (v, cv)):
            t = R.spk[w]
            if fn is None:
                ax.step(edges[:-1] + R.offset, np.histogram(t, edges)[0] / BIN_S, where="post", color=col, lw=1,
                        label=f"#{w} ({R.units.loc[w, 'label']}, {t.size} spikes)")
            else:
                vals = R.amp[w] if fn == "amp" else R.y[w]
                sub = slice(None, None, max(1, t.size // 4000))
                ax.plot(t[sub] + R.offset, vals[sub], ".", ms=1.2, color=col, alpha=0.5)
        ax.set_ylabel(lab, fontsize=8)
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        if row == 2:
            ax.set_xlabel("run time (s)")
    # ACGs and CCG
    for row, (a, b, title) in enumerate(((u, u, f"ACG #{u}"), (v, v, f"ACG #{v}"), (u, v, f"CCG #{u} x #{v}"))):
        ax = fig.add_subplot(gs[row, 2])
        e, h = ccg(R.spk[a], R.spk[b])
        ax.bar((e[:-1] + e[1:]) / 2 * 1e3, h, width=0.5, color=cu if a == b == u else (cv if a == b else "0.3"))
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
    ax.set_xlabel("ms")
    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")
    txt = "\n".join([f"window {r['window']}", "",
                     f"#{u} last active: {r['first_last_active_run_s']:.0f} s",
                     f"#{v} first active: {r['second_first_active_run_s']:.0f} s",
                     f"active together: {r['overlap_frac']:.0%} of bins", "",
                     f"rate when active: {r['rate_when_active_first_hz']:.1f} vs {r['rate_when_active_second_hz']:.1f} Hz",
                     f"median amplitude: {r['median_amp_first_uv']:.0f} vs {r['median_amp_second_uv']:.0f} uV",
                     f"centres: {r['centre_um_first']:.0f} vs {r['centre_um_second']:.0f} um", "",
                     "Kilosort merge suggestion" if r["kilosort_merge_suggestion"] else "",
                     "", f"traces: review_units.py {R.dir.name}", f"  --time-offset-s {R.offset:g}"])
    ax.text(0, 1, txt, va="top", fontsize=8, family="monospace")
    fig.suptitle(f"{R.dir.parent.name} {R.dir.name}: #{u} -> #{v}, one neuron split in two?", fontsize=11)
    fig.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dirs", nargs="+")
    p.add_argument("--sorter", default="kilosort4")
    a = p.parse_args()
    for d in a.output_dirs:
        R = Run(Path(d), a.sorter)
        out = Path(d) / "split_review"
        (out / "figures").mkdir(parents=True, exist_ok=True)
        C = candidates(R)
        f = out / "split_candidates.tsv"
        if f.exists() and len(C):                     # keep verdicts already entered
            old = pd.read_csv(f, sep="\t").set_index(["unit_first", "unit_second"])[["your_verdict", "note"]]
            C = C.drop(columns=["your_verdict", "note"]).join(old, on=["unit_first", "unit_second"])
            C[["your_verdict", "note"]] = C[["your_verdict", "note"]].fillna("")
        C.to_csv(f, sep="\t", index=False)
        for r in C.to_dict(orient="records"):
            figure(R, r, out / "figures" / f"{r['template_cosine']:.3f}_{r['unit_first']}_{r['unit_second']}.png")
        n_hi = int((C["template_cosine"] >= 0.92).sum()) if len(C) else 0
        print(f"{R.dir.parent.name}/{R.dir.name}: {len(C)} candidate pairs ({n_hi} with cosine >= 0.92) -> {f}")


if __name__ == "__main__":
    main()
