"""Population spiking signatures behind the HMM states, and single-unit PETHs (called from figures_population).

Outputs in <figures>/signatures and <figures>/units:
  state_fingerprints.png      each state's emission rate per unit relative to the unit's mean rate (log2), every K
  state_examples_K{k}.png     per condition in order: an example raster over the state probabilities and each
                              unit's firing rate in each state of that condition (as Jones et al. 2007 Fig. 2)
  population_peth_ion.png     unit PETHs (rows) per condition (columns) around inspiration onset / offset, scaled to
  population_peth_ioff.png    each unit's maximum over conditions, with the reference model's state occupancy
  unit_phase_polar.png        phase curves (cibrrig get_phase_curve) of every unit, one line per condition
  units_slides/unit_<id>_raster.png  per unit, slide-sized: breath-by-breath rasters (onset and offset aligned,
                              recording order), condition bars next to the condition labels
  units_slides/unit_<id>_peth.png    per unit, slide-sized: PETHs and phase curves as a progression (each condition
                              bold in its colour over the earlier ones thin) plus a column with all conditions
  units_slides/slides.json    drafted slide text per unit, read by make_unit_slides.js
  units/unit_<id>_states.png  per unit: the same rasters on each breath's population state (reference model), its rate
                              in each state, its share of spikes vs time per state
Colour means HMM state in the state figures; the PETH cards (no HMM) use condition colours.
  unit_peth_features.tsv      per unit and condition: onset-aligned peak latency and rate, pre-onset ramp slope and
                              ratio, decay from the peak (written next to the figures folder)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from figures_population import INK, INK2, GRID, condition_block_labels, save_fig, state_colour, unit_rows  # noqa: E402

PRE_S, POST_S = 0.4, 0.6          # PETH / raster window around the alignment event
PETH_BIN_S, PETH_SIGMA_BINS = 0.010, 1.0
RASTER_BIN_S = 0.002
RAMP_WIN = (-0.15, 0.0)           # pre-onset ramp: inside expiration at baseline (Te ~ 190 ms)
EARLY_WIN = (-0.25, -0.15)


def events(c, align):
    return c["dia_df"]["on_sec" if align == "on" else "off_sec"].to_numpy()


def aligned_counts(ts, ev, binw, pre=PRE_S, post=POST_S):
    """Spike counts per breath (rows) and bin around each event; NaN rows where the window leaves the data."""
    nb = int(round((pre + post) / binw))
    M = np.zeros((ev.size, nb), np.float32)
    lo, hi = np.searchsorted(ts, ev - pre), np.searchsorted(ts, ev + post)
    for i in range(ev.size):
        k = np.floor((ts[lo[i]:hi[i]] - ev[i] + pre) / binw).astype(int)
        np.add.at(M[i], k[(k >= 0) & (k < nb)], 1)
    return (np.arange(nb) + 0.5) * binw - pre, M


def valid_events(c, ev):
    dur = c["dia_sub"].size / c["sr_sub"]
    return (ev - PRE_S >= 0) & (ev + POST_S <= dur)


def peths(c, u, align):
    """Per condition: mean and SEM rate (Hz) around the event, smoothed."""
    ev = events(c, align)
    t, M = aligned_counts(c["spikes"][u], ev, PETH_BIN_S)
    ok = valid_events(c, ev)
    out = {}
    for name in c["conditions"]:
        sel = ok & (c["cond"] == name)
        if sel.sum() < 3:
            continue
        R = gaussian_filter1d(M[sel] / PETH_BIN_S, PETH_SIGMA_BINS, axis=1)
        out[name] = (R.mean(axis=0), R.std(axis=0) / np.sqrt(sel.sum()), int(sel.sum()))
    return t, out


def peth_features(t, m):
    """Onset-aligned PETH features: peak latency/rate (-200..+300 ms), pre-onset ramp, decay after the peak."""
    w = (t >= -0.2) & (t <= 0.3)
    i = np.flatnonzero(w)[np.argmax(m[w])]
    peak, lat = float(m[i]), float(t[i])
    ramp = (t >= RAMP_WIN[0]) & (t < RAMP_WIN[1])
    slope = float(np.polyfit(t[ramp], m[ramp], 1)[0] * 0.1) if ramp.sum() > 2 else np.nan   # Hz per 100 ms
    early = (t >= EARLY_WIN[0]) & (t < EARLY_WIN[1])
    late = (t >= -0.05) & (t < 0.0)
    ratio = float((m[late].mean() + 0.5) / (m[early].mean() + 0.5))
    after = np.flatnonzero((t > t[i]) & (m <= peak / 2))
    half = float(t[after[0]] - t[i]) if after.size else np.nan
    return {"peak_latency_ms": lat * 1e3, "peak_hz": peak, "ramp_slope_hz_per_100ms": slope,
            "ramp_ratio_last50_vs_early": ratio, "decay_to_half_ms": half * 1e3}


def fig_fingerprints(c):
    ks, rows = c["ks"], unit_rows(c)
    mean_hz = c["X"].mean(axis=0) / c["hmm_bin_s"]
    fig, axes = plt.subplots(len(ks), 1, figsize=(13, 1.2 + 0.36 * sum(ks)), gridspec_kw={"height_ratios": ks},
                             layout="constrained")
    for ax, k in zip(axes, ks):
        order = np.argsort(c["colours"][k])
        L = np.log2((c["F"][k]["rates_hz"][order][:, rows] + 0.5) / (mean_hz[rows] + 0.5))
        im = ax.imshow(L, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
        ax.set_yticks(range(k))
        ax.set_yticklabels([c["F"][k]["feats"].loc[s, "label"] for s in order], fontsize=6.5)
        for y, s in enumerate(order):
            ax.add_patch(Rectangle((-0.03, y - 0.4), 0.018, 0.8, color=state_colour(c, k, s), clip_on=False,
                                   transform=ax.get_yaxis_transform()))
        ax.tick_params(axis="y", pad=12)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"#{u}" for u in c["order_units"]], rotation=90, fontsize=6.5)
        ax.set_ylabel(f"K = {k}" + ("\n(reference)" if k == c["ref_k"] else ""), rotation=0, ha="right", va="center",
                      labelpad=30)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, shrink=0.5)
    cb.set_label("log2 (state rate / unit's mean rate)")
    fig.suptitle("State fingerprints: which units fire more (red) or less (blue) than their average in each state "
                 "(units ordered by baseline preferred φ, tonic last)", x=0.01, ha="left")
    save_fig(fig, c["sig_out"] / "state_fingerprints.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def example_window(c, name, k):
    """A few breaths around the condition's most typical breath: among the middle half of its breaths, the one whose
    mix of states (fraction of the cycle in each state of model k) is closest to the condition's average mix.
    1.3 s from just before that inspiration onset, or 2.5 s around a gasp."""
    d, dt, seq = c["dia_df"], c["hmm_bin_s"], c["F"][k]["seq"]
    idx = np.flatnonzero(c["cond"] == name)
    if not idx.size:
        return None
    ons = d["on_sec"].to_numpy()
    mix = np.stack([np.bincount(seq[int(ons[i] / dt):int(ons[i + 1] / dt)], minlength=k) for i in idx]).astype(float)
    mix /= np.maximum(mix.sum(axis=1, keepdims=True), 1)
    dist = np.abs(mix - mix.mean(axis=0)).sum(axis=1)
    middle = np.arange(len(idx) // 4, max(len(idx) // 4 + 1, 3 * len(idx) // 4))
    i = idx[middle[int(np.argmin(dist[middle]))]]
    on = ons[i]
    t0, t1 = (on - 1.0, on + 1.5) if name == "gasping" else (on - 0.15, on + 1.15)
    dur = c["dia_sub"].size / c["sr_sub"]
    return max(0.0, t0), min(dur, t1)


def cycle_order(c, k, states):
    """States in breath-cycle order starting at inspiration onset (by preferred phase over all conditions)."""
    ph = c["F"][k]["feats"].loc[states, "phase"].to_numpy() % (2 * np.pi)
    return [states[i] for i in np.argsort(ph)]


def fig_state_progression(c, k):
    """Like Jones et al. 2007 (PNAS) Fig. 2: for each condition in order, an example raster of all units over the
    probability of each HMM state (shaded, bold outline), and each unit's firing rate in each state of that
    condition (one bar panel per state, in breath-cycle order)."""
    F, dt, off = c["F"][k], c["hmm_bin_s"], c["time_offset_s"]
    conds = [n for n in c["conditions"] if (c["cond"] == n).any()]
    rows_order = c["order_units"]
    nu = len(rows_order)
    pos = {u: i for i, u in enumerate(c["ids"])}
    nbin = F["seq"].size
    counts = np.stack([np.bincount(np.clip(np.floor(c["spikes"][u] / dt).astype(int), 0, nbin - 1), minlength=nbin)
                       for u in rows_order], axis=1)
    present_by = {}
    for name in conds:
        in_cond = c["bcond"] == name
        present_by[name] = cycle_order(c, k, [s for s in range(k) if (F["seq"][in_cond] == s).mean() >= 0.02])
    ncol = max(1, max(len(v) for v in present_by.values()))
    fig = plt.figure(figsize=(9 + 2.3 * ncol, 4.3 * len(conds) + 1.2))
    outer = fig.add_gridspec(len(conds), 2, width_ratios=[7.5, 2.3 * ncol], hspace=0.42, wspace=0.07,
                             top=0.95, bottom=0.04)
    for r, name in enumerate(conds):
        win = example_window(c, name, k)
        left = outer[r, 0].subgridspec(2, 1, height_ratios=[4, 0.9], hspace=0.06)
        ax = fig.add_subplot(left[0])
        axd = fig.add_subplot(left[1], sharex=ax)
        t0, t1 = win
        b0, b1 = int(t0 / dt), int(t1 / dt)
        tt = (np.arange(b0, b1) + 0.5) * dt - t0
        post = F["post"][b0:b1]
        for s in np.argsort(c["colours"][k]):
            if post[:, s].max() < 0.05:
                continue
            col = np.array(matplotlib.colors.to_rgb(state_colour(c, k, s)))
            ax.fill_between(tt, 0, post[:, s], color=col, alpha=0.35, lw=0)
            ax.plot(tt, post[:, s], color=col * 0.75, lw=1.6)
        for y, u in enumerate(rows_order):
            ts = c["spikes"][u]
            ts = ts[(ts >= t0) & (ts < t1)] - t0
            ax.vlines(ts, 1 - (y + 0.85) / nu, 1 - (y + 0.15) / nu, color=INK, lw=0.7)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, t1 - t0)
        ax.set_ylabel("probability in state (colour)", fontsize=8)
        ax.tick_params(labelbottom=False, labelsize=7)
        ax.set_title(f"{r + 1}. {name}: example {t0 + off:.2f}–{t1 + off:.2f} s", loc="left", fontsize=10, color=INK)
        ay = ax.twinx()
        ay.set_ylim(0, 1)
        ay.set_yticks(1 - (np.arange(nu) + 0.5) / nu)
        ay.set_yticklabels([f"#{u}" for u in rows_order], fontsize=5)
        ay.tick_params(length=0, pad=1)
        ay.spines[["top", "right", "left", "bottom"]].set_visible(False)
        i0, i1 = int(t0 * c["sr_sub"]), int(t1 * c["sr_sub"])
        axd.plot(np.arange(i0, i1) / c["sr_sub"] - t0, c["dia_sub"][i0:i1], color=INK, lw=0.7)
        axd.set_yticks([])
        axd.set_ylabel("∫dia", fontsize=7)
        axd.set_xlabel("time (s)", fontsize=8)
        axd.tick_params(labelsize=7)
        # firing rate of every unit in each state present in this condition, rows level with the raster's rows
        in_cond = c["bcond"] == name
        present = present_by[name]
        right = outer[r, 1].subgridspec(2, ncol, height_ratios=[4, 0.9], hspace=0.06, wspace=0.12)
        rates = {}
        for s in present:
            m = in_cond & (F["seq"] == s)
            rates[s] = counts[m].sum(axis=0) / max(m.sum() * dt, 1e-9)
        xmax = max((v.max() for v in rates.values()), default=1) * 1.08
        for j, s in enumerate(present):
            axb = fig.add_subplot(right[0, j])
            axb.barh(np.arange(nu), rates[s], color=state_colour(c, k, s), height=0.75)
            axb.set_ylim(nu - 0.5, -0.5)
            axb.set_xlim(0, xmax)
            axb.set_yticks(np.arange(nu))
            axb.set_yticklabels([])
            axb.tick_params(axis="x", labelsize=6.5)
            occ = (F["seq"][in_cond] == s).mean()
            lbl = F["feats"].loc[s, "label"].split(" | ")[0]
            axb.set_title(f"{lbl}\n{100 * occ:.0f}% of {name}", fontsize=7.5)
            axb.set_xlabel("Hz", fontsize=7)
        if not present:
            fig.add_subplot(right[0, 0]).axis("off")
    order = np.argsort(c["colours"][k])
    handles = [Rectangle((0, 0), 1, 1, color=state_colour(c, k, s), alpha=0.6) for s in order]
    fig.legend(handles, [F["feats"].loc[s, "label"] for s in order], loc="upper center", ncol=min(k, 4),
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"K = {k} population HMM, condition by condition. Left: spikes of all {nu} units (rows, baseline "
                 f"preferred φ order) over the probability of each state (shaded, bold outline).\nRight: each unit's "
                 f"firing rate in each state during that condition, states in breath-cycle order from inspiration "
                 f"onset", x=0.01, y=0.998, ha="left", fontsize=10)
    save_fig(fig, c["sig_out"] / f"state_examples_K{k}.png", dpi=100)
    plt.close(fig)


def fig_population_peth(c, align):
    k = c["ref_k"]
    conds = c["conditions"]
    rows_order = c["order_units"]
    data = {u: peths(c, u, align) for u in rows_order}
    t = data[rows_order[0]][0]
    unit_max = {u: max((v[0].max() for v in data[u][1].values()), default=1.0) for u in rows_order}
    ev = events(c, align)
    ok = valid_events(c, ev)
    dt = c["hmm_bin_s"]
    nb = int(round((PRE_S + POST_S) / dt))
    tt = (np.arange(nb) + 0.5) * dt - PRE_S
    fig, axes = plt.subplots(3, len(conds), figsize=(3.6 * len(conds), 11.5), sharex=True,
                             gridspec_kw={"height_ratios": [4.2, 1.0, 0.8]})
    axes = axes[[1, 0, 2]]          # draw PETHs first in the figure, the HMM strip below them
    sym = "inspiration onset" if align == "on" else "inspiration offset"
    for j, name in enumerate(conds):
        sel = np.flatnonzero(ok & (c["cond"] == name))
        axes[1, j].set_title(f"{name} ({sel.size} breaths)", fontsize=10)
        ax = axes[0, j]
        if sel.size:
            S = np.stack([c["F"][k]["seq"][int(e / dt) - int(PRE_S / dt):int(e / dt) - int(PRE_S / dt) + nb]
                          for e in ev[sel]])
            order = np.argsort(c["colours"][k])
            P = np.stack([(S == s).mean(axis=0) for s in order])
            ax.stackplot(tt * 1e3, P, colors=[state_colour(c, k, s) for s in order], lw=0)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel(f"population HMM\nstate (K = {k})\nfraction of breaths")
        else:
            ax.set_yticklabels([])
        ax = axes[1, j]
        img = np.full((len(rows_order), t.size), np.nan)
        for y, u in enumerate(rows_order):
            if name in data[u][1]:
                img[y] = data[u][1][name][0] / max(unit_max[u], 1e-9)
        ax.imshow(img, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                  extent=[t[0] * 1e3 - 5, t[-1] * 1e3 + 5, len(rows_order), 0])
        ax.set_yticks(np.arange(len(rows_order)) + 0.5)
        ax.set_yticklabels([f"#{u}" for u in rows_order] if j == 0 else [], fontsize=6)
        ax.axvline(0, color="white", lw=0.8, ls=":")
        ax = axes[2, j]
        if sel.size:
            ns = int((PRE_S + POST_S) * c["sr_sub"])
            i0 = (ev[sel] * c["sr_sub"]).astype(int) - int(PRE_S * c["sr_sub"])
            dia = np.stack([c["dia_sub"][a:a + ns] for a in i0])
            td = np.arange(ns) / c["sr_sub"] - PRE_S
            ax.plot(td * 1e3, dia.mean(axis=0), color=INK, lw=1)
            ax.fill_between(td * 1e3, *np.percentile(dia, [25, 75], axis=0), color=GRID, lw=0)
        ax.axvline(0, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel(f"ms from {sym}")
        if j == 0:
            ax.set_ylabel("∫dia (z)")
    axes[1, 0].set_ylabel("one row per unit (baseline preferred φ, tonic last)")
    fig.suptitle(f"Population PETHs around {sym}: each row is one unit's mean rate (10 ms bins, σ 10 ms), scaled to that "
                 f"unit's maximum over the conditions (columns).\nMiddle strip: which population state the HMM "
                 f"(K = {k}, fitted to all units together) is in at each time, as a fraction of breaths; colours are "
                 f"HMM states (legend), not conditions.", x=0.01, ha="left", fontsize=10)
    order = np.argsort(c["colours"][k])
    handles = [Rectangle((0, 0), 1, 1, color=state_colour(c, k, s)) for s in order]
    fig.legend(handles, [c["F"][k]["feats"].loc[s, "label"] for s in order], loc="lower center", ncol=3,
               frameon=False, fontsize=8, title=f"HMM states (K = {k})", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save_fig(fig, c["sig_out"] / f"population_peth_i{align}.png", dpi=120)
    plt.close(fig)


def fig_polar(c):
    rows_order = c["order_units"]
    pos = {u: i for i, u in enumerate(c["ids"])}
    ncol = 7
    nrow = int(np.ceil(len(rows_order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow), subplot_kw={"projection": "polar"})
    b = np.r_[c["curve_bins"], c["curve_bins"][0]]
    tu = c["tuning"].set_index(["unit_id", "condition"])
    for ax, u in zip(axes.ravel(), rows_order):
        for name in c["conditions"]:
            if name not in c["curves"]:
                continue
            r = gaussian_filter1d(np.nan_to_num(c["curves"][name][:, pos[u]]), 1.0, mode="wrap")
            ax.plot(b, np.r_[r, r[0]], color=c["cond_colours"][name], lw=1.0)
        cls = tu.loc[(u, "baseline"), "class"]
        ax.set_title(f"#{u} ({cls})", fontsize=8, pad=8)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["on", "", "off", ""], fontsize=6)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_rlabel_position(135)
    for ax in axes.ravel()[len(rows_order):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=c["cond_colours"][n]) for n in c["conditions"]]
    fig.legend(handles, c["conditions"], loc="lower center", ncol=len(c["conditions"]), frameon=False)
    fig.suptitle("Phase curves (Hz vs φ; 0 = inspiration onset at right, π = offset at left; counter-clockwise "
                 "in time); units ordered by baseline preferred φ, class at baseline", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, c["sig_out"] / "unit_phase_polar.png", dpi=120)
    plt.close(fig)


def cond_abbr(name):
    """'early recovery' -> 'ER', 'baseline' -> 'B'."""
    return "".join(w[0].upper() for w in name.split())


def short_label(label):
    """'early I | baseline–anoxia, slow breathing' -> 'early I | B–A, SB' (condition names as initials)."""
    if " | " not in label:
        return label
    part, where = label.split(" | ", 1)
    runs = [("–".join(cond_abbr(x) for x in run.split("–"))) for run in where.split(", ")]
    return part + " | " + ", ".join(runs)


def abbr_key(c):
    return ", ".join(f"{cond_abbr(n)} {n}" for n in c["conditions"])


def unit_header(c, u):
    info = c["info"].set_index("unit_id").loc[u]
    tu = c["tuning"].set_index(["unit_id", "condition"])
    rate_txt = ", ".join(f"{n} {tu.loc[(u, n), 'rate_hz']:.1f}" for n in c["conditions"] if (u, n) in tu.index)
    base_rate = tu.loc[(u, "baseline"), "rate_hz"] if (u, "baseline") in tu.index else 0.0
    tuning_txt = ("silent at baseline" if base_rate < 0.2 else
                  f"baseline: {info['class']}, preferred φ {np.degrees(info['preferred_phase']):.0f}°, "
                  f"L {info['modulation_L']:.2f}")
    return f"Unit #{u}  |  depth {info['depth_um']:.0f} µm, {info['label']}, {tuning_txt}  |  rate (Hz): {rate_txt}"


def state_rows(c, k, align):
    """Viterbi state of model k in each HMM bin from -PRE_S to +POST_S around every breath (-1 outside the data)."""
    dt, seq = c["hmm_bin_s"], c["F"][k]["seq"]
    n0, n1 = int(round(PRE_S / dt)), int(round(POST_S / dt))
    ev = events(c, align)
    rows = np.full((ev.size, n0 + n1), -1)
    for i, t in enumerate(ev):
        b = int(np.floor(t / dt))
        if b - n0 >= 0 and b + n1 <= seq.size:
            rows[i] = seq[b - n0:b + n1]
    return rows


def spike_marker_size(c, u):
    """Marker area (pt²) for raster spikes: ~959 breath rows share < 1000 px, so each spike is drawn as a solid mark of
    a few px; smaller for high-rate units so their rasters don't fill in."""
    rate = c["spikes"][u].size / (c["dia_sub"].size / c["sr_sub"])
    return float(np.clip(25.0 / max(rate, 1e-3), 0.2, 2.2))


def draw_unit_raster(ax, c, u, align, k=None, edges=True, marker_scale=1.0, dpi=None):
    """Spikes (black marks) around each breath's onset or offset, one row per breath in recording order; with k, on
    a background of that model's Viterbi state (lightened colours). Spikes are drawn as markers, not an image: an image
    of 2 ms bins x 959 rows is downsampled on screen and loses most spikes. With dpi (of the saved PNG), the marker
    side is rounded to whole pixels: Agg snaps small square markers to the pixel grid, so a side of 2.99 px drew as 2 px
    and 3.03 px as 3 px, and two units (or trials) with near-equal rates looked very different."""
    ev = events(c, align)
    ts = c["spikes"][u]
    n = ev.size
    ext = [-PRE_S * 1e3, POST_S * 1e3, n, 0]
    if k is not None:
        cols = np.array([matplotlib.colors.to_rgb(state_colour(c, k, s)) for s in range(k)] + [(1, 1, 1)])
        light = 1 - 0.55 * (1 - cols)
        ax.imshow(light[state_rows(c, k, align)], aspect="auto", interpolation="nearest", extent=ext)
    if edges:   # under the spikes, light so they don't read as spikes
        d = c["dia_df"]
        t = d["on_sec" if align == "on" else "off_sec"]
        other = d["off_sec"] if align == "on" else d["on_sec"]
        for x in ((other - t).to_numpy() * 1e3, (d["on_sec"].shift(-1) - t).to_numpy() * 1e3):
            m = (x > -PRE_S * 1e3) & (x < POST_S * 1e3)
            ax.scatter(x[m], np.flatnonzero(m) + 0.5, s=0.6, color="#c3c2b7", lw=0, zorder=1, rasterized=True)
    lo, hi = np.searchsorted(ts, ev - PRE_S), np.searchsorted(ts, ev + POST_S)
    xs = np.concatenate([(ts[a:b] - e) * 1e3 for a, b, e in zip(lo, hi, ev)])
    ys = np.repeat(np.arange(n) + 0.5, hi - lo)
    size = marker_scale * spike_marker_size(c, u)
    if dpi:
        side_px = max(1, round(np.sqrt(size) * dpi / 72))
        size = (side_px * 72 / dpi) ** 2 * 1.001
    ax.scatter(xs, ys, s=size, marker="s", color=INK, lw=0, zorder=2, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(-PRE_S * 1e3, POST_S * 1e3)
    ax.set_ylim(n, 0)


SLIDE_FONT = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11}
SLIDE_GROUPS = [("expiratory", "Expiratory units"), ("inspiratory", "Inspiratory units"), ("tonic", "Tonic units"),
                ("silent at baseline", "Units silent at baseline")]


def unit_slide_raster(c, u):
    """Slide-sized rasters (8.7 x 6.6 in, 12-13 pt text): onset and offset aligned, condition bars next to the
    condition labels on the right."""
    with plt.rc_context(SLIDE_FONT):
        fig = plt.figure(figsize=(8.7, 6.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.42], wspace=0.08, left=0.1, right=0.995, top=0.9,
                              bottom=0.11)
        for j, align in enumerate(("on", "off")):
            name = "inspiration onset" if align == "on" else "inspiration offset"
            ax = fig.add_subplot(gs[j])
            draw_unit_raster(ax, c, u, align, marker_scale=0.6, dpi=220)
            ax.set_title(f"aligned to {name}")
            ax.set_xlabel("ms from onset" if align == "on" else "ms from offset")
            ax.set_xticks([-400, -200, 0, 200, 400])     # a 600 label would run into the next panel's -400
            if j == 0:
                ax.set_ylabel("breath (recording order)")
            else:
                ax.set_yticklabels([])
                condition_block_labels(ax, c, POST_S * 1e3, gap_frac=0.075, colours=c["cond_colours"], fontsize=10)
        fig.suptitle(f"#{u}: spikes on every breath", x=0.01, ha="left", fontsize=14)
        save_fig(fig, c["slide_out"] / f"unit_{u}_raster.png", dpi=220)
        plt.close(fig)


def unit_slide_peth(c, u):
    """Slide-sized PETHs and phase curves (12.5 x 5.75 in): one column per condition, that condition bold in its
    colour over thin lines of the earlier ones, and a last column with all conditions."""
    conds = c["conditions"]
    nc, col = len(conds), c["cond_colours"]
    peth = {align: peths(c, u, align) for align in ("on", "off")}
    ymax = max((m + se).max() for _, P in peth.values() for m, se, _ in P.values()) * 1.05
    pos = {x: i for i, x in enumerate(c["ids"])}
    b = np.r_[c["curve_bins"], c["curve_bins"][0]]
    curves = {x: gaussian_filter1d(np.nan_to_num(c["curves"][x][:, pos[u]]), 1.0, mode="wrap")
              for x in conds if x in c["curves"]}
    rmax = max(v.max() for v in curves.values()) * 1.05 if curves else 1
    with plt.rc_context({**SLIDE_FONT, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig = plt.figure(figsize=(12.5, 5.75))
        gs = fig.add_gridspec(3, nc + 1, height_ratios=[1, 1, 1.25], hspace=0.62, wspace=0.28, left=0.065,
                              right=0.995, top=0.8, bottom=0.04)
        for r, align in enumerate(("on", "off")):
            t, P = peth[align]
            for j in range(nc + 1):
                ax = fig.add_subplot(gs[r, j])
                bold = conds[j] if j < nc else None
                shown = conds[:j + 1] if j < nc else conds
                for cond in shown:
                    if cond in P and cond != bold:
                        ax.plot(t * 1e3, P[cond][0], color=col[cond], lw=1.0 if bold else 1.6,
                                alpha=0.85 if bold else 1.0)
                if bold in P:
                    m, se, _ = P[bold]
                    ax.fill_between(t * 1e3, m - se, m + se, color=col[bold], alpha=0.22, lw=0)
                    ax.plot(t * 1e3, m, color=col[bold], lw=2.6)
                ax.axvline(0, color=INK2, lw=0.7, ls=":")
                ax.set_ylim(0, ymax)
                ax.set_xlim(-PRE_S * 1e3, POST_S * 1e3)
                ax.set_xticks([-400, 0, 400])
                if r == 0:
                    # two-word names on two lines: 8 columns share the slide width
                    ax.set_title(f"{j + 1}. " + bold.replace(" ", "\n", 1) if bold else "all", fontsize=12)
                ax.set_xlabel("ms from onset" if align == "on" else "ms from offset", fontsize=10.5, labelpad=1)
                if j == 0:
                    ax.set_ylabel(("onset" if align == "on" else "offset") + " (Hz)", fontsize=11.5)
                else:
                    ax.set_yticklabels([])
        for j in range(nc + 1):
            ax = fig.add_subplot(gs[2, j], projection="polar")
            bold = conds[j] if j < nc else None
            shown = conds[:j + 1] if j < nc else conds
            for cond in shown:
                if cond in curves and cond != bold:
                    ax.plot(b, np.r_[curves[cond], curves[cond][0]], color=col[cond], lw=1.0 if bold else 1.6,
                            alpha=0.85 if bold else 1.0)
            if bold in curves:
                ax.plot(b, np.r_[curves[bold], curves[bold][0]], color=col[bold], lw=2.6)
            ax.set_ylim(0, max(rmax, 1e-3))
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax.set_xticklabels([""] * 4)        # on/off labels collide with the next panel; see the row label
            ax.set_yticks(ax.get_yticks()[1::2])
            ax.tick_params(axis="y", labelsize=8)
            ax.set_rlabel_position(112.5)
            if j == 0:
                ax.text(-0.3, 0.5, "phase\n(onset at right)", transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=10.5, color=INK)
        n_breaths = {x: v[2] for x, v in peth["on"][1].items()}
        handles = [Rectangle((0, 0), 1, 1, color=col[x]) for x in conds]
        names = [f"{x} ({n_breaths.get(x, 0)} breaths)" for x in conds]
        fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=min(nc, 4), frameon=False,
                   fontsize=11, handlelength=1.2, columnspacing=1.6)
        save_fig(fig, c["slide_out"] / f"unit_{u}_peth.png", dpi=200)
        plt.close(fig)


def unit_story(c, u):
    """Draft slide text from the unit's numbers: group, headline and up to four bullets (for the user to edit)."""
    tu = c["tuning"].set_index(["unit_id", "condition"])

    def get(cond, col):
        return tu.loc[(u, cond), col] if (u, cond) in tu.index else np.nan
    conds = c["conditions"]
    r = {x: get(x, "rate_hz") for x in conds}
    after_gasp = conds[conds.index("gasping") + 1:]
    first, last = after_gasp[0], after_gasp[-1]          # e.g. early recovery, later recovery
    base, gasp, rec, post = r["baseline"], r["gasping"], r[first], r[last]
    t, P = peths(c, u, "on")
    feat = {x: peth_features(t, P[x][0]) for x in P}
    ramp_in = ("gasping" in feat and feat["gasping"]["ramp_ratio_last50_vs_early"] >= 2.5
               and feat["gasping"]["ramp_slope_hz_per_100ms"] > 5 and abs(feat["gasping"]["peak_latency_ms"]) <= 100)
    if not base >= 0.2:
        group = "silent at baseline"
        kind = "Unit silent at baseline"
        if gasp >= 1:
            act = "fires during anoxia and gasping" if r["anoxia"] >= 1 else "fires only around the gasps"
            if ramp_in:
                act = "ramps up into each gasp"
        elif r["anoxia"] >= 1:
            act = "fires mainly during anoxia"
        else:
            act = "barely fires"
        after = "silent again afterwards" if not post >= 0.2 else "keeps firing afterwards"
    else:
        group = get("baseline", "class")
        kind = f"{group.capitalize()} unit"
        g = gasp / base
        if g < 0.1:
            act = "falls silent during gasping"
        elif g < 0.5:
            act = "fires much less during gasping"
        elif g <= 2:
            act = "keeps firing during gasping"
        else:
            act = "fires more during gasping"
        if ramp_in and g >= 0.1:
            act += " and ramps up into each gasp"
        elif g >= 0.1 and get("gasping", "modulation_L") >= 0.4 and group != "inspiratory":
            act += " and locks to the gasps"
        rr, pp = rec / base, post / base
        if pp > 1.33:
            after = "then fires above baseline"
        elif pp >= 0.75:
            after = "slowly recovers firing" if rr < 0.75 else "returns to baseline afterwards"
        elif post >= 1.5 * rec:
            after = "slowly recovers firing, still below baseline"
        elif pp >= 0.5:
            after = "recovers only partly"
        else:
            after = "stays suppressed afterwards"
    headline = f"{kind}: {act}, {after}"
    pct = (lambda x: f" ({100 * x / base:.0f}%)" if base >= 0.2 else "")
    bullets = [f"Baseline {base:.1f} Hz, gasping {gasp:.1f} Hz",
               f"{first.capitalize()} {rec:.1f} Hz{pct(rec)}, {last} {post:.1f} Hz{pct(post)}"]
    Lb, Lp = get("baseline", "modulation_L"), get(last, "modulation_L")
    if base >= 0.2 and Lb >= 0.2:
        txt = f"Breathing phase at baseline {np.degrees(get('baseline', 'preferred_phase')):.0f}° (L {Lb:.2f})"
        if Lp >= 0.2:
            txt += f", {last} {np.degrees(get(last, 'preferred_phase')):.0f}° (L {Lp:.2f})"
        bullets.append(txt)
        lat = feat["baseline"]["peak_latency_ms"] if "baseline" in feat else np.nan
        bullets.append(f"Peak firing {abs(lat):.0f} ms {'before' if lat < 0 else 'after'} inspiration onset at baseline")
    elif base >= 0.2:
        bullets.append(f"Not modulated by breathing at baseline (L {Lb:.2f})")
    if ramp_in:
        fg = feat["gasping"]
        bullets.append(f"Before each gasp: +{fg['ramp_slope_hz_per_100ms']:.0f} Hz per 100 ms, peak "
                       f"{fg['peak_latency_ms']:+.0f} ms from onset")
    return group, headline, bullets


def condition_spans_raw(c):
    """Each condition's first breath onset and last breath end, raw-run time (for the deck's timeline)."""
    d, off = c["dia_df"], c["time_offset_s"]
    out = []
    for n in c["conditions"]:
        idx = np.flatnonzero(c["cond"] == n)
        if idx.size:
            out.append({"name": n, "start_raw_s": round(float(d["on_sec"].iloc[idx[0]] + off), 1),
                        "end_raw_s": round(float(d["on_sec"].iloc[min(idx[-1] + 1, len(d) - 1)] + off), 1),
                        "colour": c["cond_colours"][n]})
    return out


def write_slide_manifest(c):
    rows = []
    info = c["info"].set_index("unit_id")
    for u in c["order_units"]:
        group, headline, bullets = unit_story(c, u)
        rows.append({"unit": int(u), "group": group, "headline": headline, "bullets": bullets,
                     "depth_um": float(info.loc[u, "depth_um"]), "label": str(info.loc[u, "label"]),
                     "raster": str(c["slide_out"] / f"unit_{u}_raster.png"),
                     "peth": str(c["slide_out"] / f"unit_{u}_peth.png")})
    order = {g: i for i, (g, _) in enumerate(SLIDE_GROUPS)}
    rows.sort(key=lambda x: order.get(x["group"], 99))
    manifest = {"title": f"e097 anoxia #1: {len(rows)} reviewed single units",
                "subtitle": f"Spikes on every breath, and PETHs and phase curves from baseline to {c['conditions'][-1]}",
                "groups": [{"key": g, "title": t, "n": sum(r["group"] == g for r in rows)} for g, t in SLIDE_GROUPS],
                "condition_colours": c["cond_colours"], "conditions": condition_spans_raw(c), "units": rows,
                "breathing_figure": str(c["out"] / "breathing_epochs.png")}     # figures_population.fig_breathing_epochs
    transitions = c["out"].parent / "transitions" / "units_slides" / "transitions.json"     # transition_slides.py
    if transitions.exists():
        manifest["transitions"] = str(transitions)
    old = c["slide_out"] / "slides.json"          # keep other trials' sections (transition_slides.py)
    if old.exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        for key in ("transition_sections", "intro"):
            if prev.get(key):
                manifest[key] = prev[key]
    (c["slide_out"] / "slides.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def unit_state_card(c, u):
    """The unit's spikes on the population HMM state (reference model) of every breath, and its rate per state."""
    k = c["ref_k"]
    F, dt = c["F"][k], c["hmm_bin_s"]
    order = np.argsort(c["colours"][k])
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.85], height_ratios=[3.0, 1.15], hspace=0.25, wspace=0.42,
                          top=0.9, bottom=0.12)
    for j, align in enumerate(("on", "off")):
        name = "inspiration onset" if align == "on" else "inspiration offset"
        ax = fig.add_subplot(gs[0, j])
        draw_unit_raster(ax, c, u, align, k=k, edges=False)
        ax.set_title(f"around {name}", fontsize=9)
        ax.set_xlabel(f"ms from {name}")
        if j == 0:
            ax.set_ylabel("breath (recording order)")
        else:
            ax.set_yticklabels([])
            condition_block_labels(ax, c, POST_S * 1e3)
    fig.text(0.12, 0.925, f"This unit's spikes (black) on the population HMM state (K = {k}) of each breath "
             "(background colour)", fontsize=10, color=INK)
    # rate in each state: from the spikes (bars) and the model's emission rate (ticks)
    ts = c["spikes"][u]
    cnt = np.bincount(np.floor(ts / dt).astype(int), minlength=F["seq"].size)[:F["seq"].size]
    emp = np.array([cnt[F["seq"] == s].sum() / max((F["seq"] == s).sum() * dt, 1e-9) for s in order])
    em = F["rates_hz"][order, c["ids"].index(u)]
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(k), emp, color=[state_colour(c, k, s) for s in order], edgecolor="white")
    ax.scatter(range(k), em, marker="_", s=300, color=INK, lw=1.6, zorder=3, label="model emission rate")
    ax.set_xticks(range(k))
    ax.set_xticklabels([str(i + 1) for i in range(k)])
    ax.set_xlabel("state (legend on the right)")
    ax.set_ylabel("rate in state (Hz)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("firing rate while the population is in each state", fontsize=9)
    ax2 = fig.add_subplot(gs[1, 1])
    share = np.array([cnt[F["seq"] == s].sum() for s in order]) / max(cnt.sum(), 1)
    occ = np.array([(F["seq"] == s).mean() for s in order])
    w = 0.38
    ax2.bar(np.arange(k) - w / 2, occ, width=w, color="#d9d8d4")
    ax2.bar(np.arange(k) + w / 2, share, width=w, color=[state_colour(c, k, s) for s in order])
    ax2.set_xticks(range(k))
    ax2.set_xticklabels([str(i + 1) for i in range(k)])
    ax2.set_xlabel("state")
    ax2.set_ylabel("fraction")
    ax2.text(0.02, 0.97, "grey: share of recording time in the state\ncoloured: share of this unit's spikes",
             transform=ax2.transAxes, va="top", fontsize=7.5, color=INK2)
    ax2.set_title("share of spikes vs share of time", fontsize=9)
    axl = fig.add_subplot(gs[1, 2])
    axl.axis("off")
    for i, s in enumerate(order):
        axl.add_patch(Rectangle((0, 1 - (i + 1) / (k + 0.5)), 0.06, 0.8 / (k + 0.5), color=state_colour(c, k, s),
                                transform=axl.transAxes, clip_on=False))
        axl.text(0.09, 1 - (i + 0.6) / (k + 0.5), f"{i + 1}  {F['feats'].loc[s, 'label']}", transform=axl.transAxes,
                 va="center", fontsize=8, color=INK)
    axl.set_title(f"HMM states, K = {k} model", fontsize=9, loc="left")
    fig.suptitle(unit_header(c, u), x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.02, f"Background colour = the population state (Viterbi path of the K = {k} Poisson HMM fitted to "
             f"all {len(c['ids'])} single units together) in each 5 ms bin of each breath; black = this unit's spikes "
             "(2 ms bins). Labels on the right = condition blocks. State labels: where in the breath cycle | conditions "
             "it occurs in.", fontsize=7.5, color=INK2)
    save_fig(fig, c["unit_out"] / f"unit_{u}_states.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_features(c):
    rows = []
    for u in c["order_units"]:
        t, P = peths(c, u, "on")
        for name, (m, se, nn) in P.items():
            rows.append({"unit_id": u, "condition": name, "n_breaths": nn, **peth_features(t, m)})
    df = pd.DataFrame(rows)
    df.to_csv(c["sig_out"].parent.parent / "unit_peth_features.tsv", sep="\t", index=False, float_format="%.3f")
    return df


def make_all(c):
    c["sig_out"] = c["out"] / "signatures"
    c["unit_out"] = c["out"] / "units"
    c["sig_out"].mkdir(exist_ok=True)
    c["unit_out"].mkdir(exist_ok=True)
    fig_fingerprints(c)
    for k in c["ks"]:
        fig_state_progression(c, k)
    for align in ("on", "off"):
        fig_population_peth(c, align)
    fig_polar(c)
    c["slide_out"] = c["out"] / "units_slides"
    c["slide_out"].mkdir(exist_ok=True)
    for u in c["order_units"]:
        unit_state_card(c, u)
        unit_slide_raster(c, u)
        unit_slide_peth(c, u)
    write_slide_manifest(c)
    write_features(c)
