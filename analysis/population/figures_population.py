"""Figures for population_dynamics.py (HMM state models, PCA in the style of Bush & Ramirez 2024/2025)."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
PHI_CMAP = "twilight_shifted"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "normal"})


def save_fig(fig, path, **kw):
    """savefig that survives a PNG held open by a viewer on Windows: retry, then write <name>_new.png."""
    path = Path(path)
    for _ in range(4):
        try:
            fig.savefig(path, **kw)
            return path
        except OSError:
            time.sleep(2)
    alt = path.with_name(path.stem + "_new" + path.suffix)
    fig.savefig(alt, **kw)
    print(f"{path.name} is locked (open elsewhere?); wrote {alt.name}")
    return alt


def phase_axis(ax, label=True):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    if label:
        ax.set_xlabel("φ (0 = insp. onset, ±π = insp. off)")


def state_colour(c, k, s):
    return c["palette"][c["colours"][k][s] % len(c["palette"])]


def cycles_of(c, name, n_max=None):
    idx = np.flatnonzero(c["cond"] == name)
    if n_max and idx.size > n_max:
        idx = idx[np.linspace(0, idx.size - 1, n_max).round().astype(int)]
    return idx


def cycle_bins(c, i, dt):
    ons = c["dia_df"]["on_sec"].to_numpy()
    return np.arange(int(ons[i] / dt), int(ons[i + 1] / dt))


def condition_spans(c):
    ons = c["dia_df"]["on_sec"].to_numpy()
    spans = []
    for name in c["conditions"]:
        idx = np.flatnonzero(c["cond"] == name)
        if not idx.size:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            spans.append((name, ons[r[0]], ons[r[-1] + 1]))
    return spans


def shade(ax, c, labels=False):
    """Conditions as alternating neutral bands (colour is kept for HMM states)."""
    off = c["time_offset_s"]
    for i, (name, t0, t1) in enumerate(sorted(condition_spans(c), key=lambda s: s[1])):
        ax.axvspan(t0 + off, t1 + off, color=INK, alpha=0.05 if i % 2 == 0 else 0.10, lw=0)
        if labels:
            ax.text((t0 + t1) / 2 + off, 1.02, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7, color=INK2)


def phase_binned(values, ph, sel, nbins=50, stat="mean"):
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    b = np.clip(np.digitize(ph, edges) - 1, 0, nbins - 1)
    out = np.full((nbins,) + values.shape[1:], np.nan)
    for i in range(nbins):
        s = sel & (b == i)
        if s.any():
            out[i] = np.nanmedian(values[s], axis=0) if stat == "median" else np.nanmean(values[s], axis=0)
    return (edges[:-1] + edges[1:]) / 2, out


def unit_rows(c):
    pos = {u: i for i, u in enumerate(c["ids"])}
    return [pos[u] for u in c["order_units"]]


# ------------------------------------------------------------------------------------------- overview

def fig_overview(c):
    ks, off, dt = c["ks"], c["time_offset_s"], c["hmm_bin_s"]
    n_k = len(ks)
    hr = [1.0, 0.8, 2.2] + [0.55] * n_k + [0.7, 0.7, 0.7]
    fig, axes = plt.subplots(len(hr), 1, figsize=(15, 17), sharex=True, gridspec_kw={"height_ratios": hr})
    t_dia = np.arange(c["dia_sub"].size) / c["sr_sub"] + off
    axes[0].plot(t_dia[::5], c["dia_sub"][::5], color=INK, lw=0.4)
    axes[0].set_ylabel("∫dia (z)")
    shade(axes[0], c, labels=True)
    d = c["dia_df"]
    rate = 60 / d["ibi_s"]
    axes[1].plot(d["on_sec"] + off, rate, ".", ms=2, color=INK2)
    axes[1].plot(d.loc[d["gasp"], "on_sec"] + off, rate[d["gasp"]], ".", ms=4, color=c["cond_colours"]["gasping"],
                 label="gasp (Bush 2024 rule)")
    axes[1].set_ylabel("breaths/min")
    axes[1].legend(loc="upper right", frameon=False, fontsize=7)
    # unit rates, 100 ms bins, sqrt, scaled to each unit's 99th percentile
    step = int(round(0.1 / dt))
    n = c["X"].shape[0] // step
    R = np.sqrt(c["X"][:n * step].reshape(n, step, -1).sum(axis=1) / 0.1)
    R = R / np.maximum(np.percentile(R, 99, axis=0), 1e-6)
    rows = unit_rows(c)
    axes[2].imshow(R[:, rows].T, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                   extent=[off, off + n * 0.1, len(rows), 0])
    axes[2].set_yticks(np.arange(len(rows)) + 0.5)
    axes[2].set_yticklabels([f"#{u}" for u in c["order_units"]], fontsize=5)
    axes[2].set_ylabel("single units\n(by baseline preferred φ)")
    win = int(round(2.0 / dt))
    for ax, k in zip(axes[3:3 + n_k], ks):
        seq = c["F"][k]["seq"]
        m = seq.size // win
        frac = np.stack([(seq[:m * win].reshape(m, win) == s).mean(axis=1) for s in range(k)])
        order = np.argsort(c["colours"][k])
        tt = off + (np.arange(m) + 0.5) * win * dt
        ax.stackplot(tt, frac[order], colors=[state_colour(c, k, s) for s in order], lw=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(f"K={k}", rotation=0, ha="right", va="center")
    axes[3].set_title("HMM state occupancy per 2 s (colours matched across K)", loc="left", fontsize=8, color=INK2)
    for i, ax in enumerate(axes[-3:]):
        ax.plot(c["centres"] + off, c["P"][:, i], color=INK, lw=0.3)
        ax.set_ylabel(f"PC{i + 1}")
    for i, ax in enumerate(axes):
        if i in (1, len(axes) - 3, len(axes) - 2, len(axes) - 1):
            shade(ax, c)
        ax.set_xlim(off, off + c["dia_sub"].size / c["sr_sub"])
    axes[-1].set_xlabel("raw-run time (s)")
    fig.suptitle(f"e097 anoxia #1: {len(c['ids'])} reviewed single units (preliminary sort snapshot)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_fig(fig, c["out"] / "overview.png", dpi=110)
    plt.close(fig)


def zoom_windows(c):
    """Per condition a few breaths from its middle: 4 s from just before an inspiration onset (10 s for gasping)."""
    ons = c["dia_df"]["on_sec"].to_numpy()
    out = {}
    for name in c["conditions"]:
        idx = np.flatnonzero(c["cond"] == name)
        if not idx.size:
            continue
        on = ons[idx[len(idx) // 2]]
        out[name] = (on - 5.0, on + 5.0) if name == "gasping" else (on - 0.2, on + 3.8)
    return out


def fig_zoom(c):
    ks, dt, off = c["ks"], c["hmm_bin_s"], c["time_offset_s"]
    zoom = zoom_windows(c)
    names = list(zoom)
    widths = [zoom[n][1] - zoom[n][0] for n in names]
    fig, axes = plt.subplots(5, len(names), figsize=(3.6 * len(names) + 2, 11),
                             gridspec_kw={"width_ratios": widths, "height_ratios": [0.8, 0.6, 2.2, 1.6, 0.9]})
    rows = unit_rows(c)
    for j, name in enumerate(names):
        t0, t1 = zoom[name]
        ax = axes[0, j]
        i0, i1 = int(t0 * c["sr_sub"]), int(t1 * c["sr_sub"])
        ax.plot(np.arange(i0, i1) / c["sr_sub"] + off, c["dia_sub"][i0:i1], color=INK, lw=0.6)
        ax.set_title(name, color=INK)
        ax = axes[1, j]
        k0, k1 = np.searchsorted(c["phi_t"], [t0, t1])
        ax.scatter(c["phi_t"][k0:k1:5] + off, c["phi"][k0:k1:5], c=c["phi"][k0:k1:5], cmap=PHI_CMAP, s=1,
                   vmin=-np.pi, vmax=np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(["-π", "0", "π"])
        ax = axes[2, j]
        for r, u in enumerate(c["order_units"]):
            ts = c["spikes"][u]
            ts = ts[(ts >= t0) & (ts < t1)] + off
            ax.vlines(ts, r + 0.1, r + 0.9, color=INK, lw=0.5)
        ax.set_ylim(len(rows), 0)
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels([f"#{u}" for u in c["order_units"]] if j == 0 else [], fontsize=5)
        ax = axes[3, j]
        b0, b1 = int(t0 / dt), int(t1 / dt)
        img = np.zeros((len(ks), b1 - b0, 3))
        for r, k in enumerate(ks):
            cols = np.array([matplotlib.colors.to_rgb(state_colour(c, k, s)) for s in range(k)])
            img[r] = cols[c["F"][k]["seq"][b0:b1]]
        ax.imshow(img, aspect="auto", interpolation="nearest", extent=[t0 + off, t1 + off, len(ks), 0])
        ax.set_yticks(np.arange(len(ks)) + 0.5)
        ax.set_yticklabels([f"K={k}" for k in ks] if j == 0 else [])
        ax = axes[4, j]
        b0p, b1p = int(t0 / c["pca_bin_s"]), int(t1 / c["pca_bin_s"])
        tt = (np.arange(b0p, b1p) + 0.5) * c["pca_bin_s"] + off
        for i, ls in zip((0, 2), ["-", "--"]):
            ax.plot(tt, c["P"][b0p:b1p, i], color=INK, lw=0.7, ls=ls, label=f"PC{i + 1}")
        if j == 0:
            ax.legend(frameon=False, fontsize=7)
        for r in range(5):
            axes[r, j].set_xlim(t0 + off, t1 + off)
            if r < 4:
                axes[r, j].set_xticklabels([])
        axes[4, j].set_xlabel("raw-run time (s)")
    axes[0, 0].set_ylabel("∫dia (z)")
    axes[1, 0].set_ylabel("φ")
    axes[2, 0].set_ylabel("single units (by baseline φ)")
    axes[3, 0].set_ylabel("Viterbi states")
    axes[4, 0].set_ylabel("PCA (baseline fit)")
    fig.suptitle("Breath-level structure in each condition (HMM colours matched across K)", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, c["out"] / "zoom_by_condition.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ HMM

def fig_hmm_models(c):
    ks = c["ks"]
    rows = unit_rows(c)
    conds = c["conditions"]
    fig, axes = plt.subplots(len(ks), 2 + len(conds), figsize=(24, 3.0 * len(ks)),
                             gridspec_kw={"width_ratios": [3.2] + [1.2] * len(conds) + [1.4]})
    for r, k in enumerate(ks):
        F = c["F"][k]
        order = np.argsort(c["colours"][k])
        ax = axes[r, 0]
        im = ax.imshow(np.sqrt(F["rates_hz"][order][:, rows]), aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(k))
        ax.set_yticklabels([F["feats"].loc[s, "label"] for s in order], fontsize=6)
        for y, s in enumerate(order):
            ax.add_patch(Rectangle((-0.035, y - 0.4), 0.025, 0.8, color=state_colour(c, k, s), clip_on=False,
                                   transform=ax.get_yaxis_transform()))
        ax.tick_params(axis="y", pad=16)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{u}" for u in c["order_units"]], fontsize=5, rotation=90)
        ax.set_ylabel(f"K = {k}", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label("√Hz", fontsize=7)
        if r == 0:
            ax.set_title("emission rates (units by baseline preferred φ)", loc="left")
        for j, name in enumerate(conds):
            ax = axes[r, 1 + j]
            P = np.nan_to_num(F["profiles"][name][:, order]).T
            P = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-12)
            ax.stackplot(F["phase_centres"], P, colors=[state_colour(c, k, s) for s in order], lw=0)
            phase_axis(ax, label=(r == len(ks) - 1))
            ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(f"P(state | φ), {name}", loc="left")
            if j:
                ax.set_yticklabels([])
        ax = axes[r, -1]
        occ = F["feats"][[f"occ_{n}" for n in c["conditions"]]].to_numpy()[order]
        bottom = np.zeros(len(c["conditions"]))
        for y, s in enumerate(order):
            ax.bar(range(len(c["conditions"])), occ[y], bottom=bottom, color=state_colour(c, k, s), width=0.8,
                   edgecolor="white", lw=0.5)
            bottom += occ[y]
        ax.set_xticks(range(len(c["conditions"])))
        ax.set_xticklabels(c["conditions"] if r == len(ks) - 1 else [], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        if r == 0:
            ax.set_title("occupancy per condition", loc="left")
    fig.tight_layout()
    save_fig(fig, c["out"] / "hmm_models.png", dpi=105)
    plt.close(fig)

    sel = c["sel"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(sel["k"], sel["log_likelihood"], "o-", color=INK, ms=4)
    axes[0].set_ylabel("log-likelihood")
    axes[1].plot(sel["k"], sel["bic"], "o-", color=INK, ms=4, label="BIC")
    axes[1].plot(sel["k"], sel["aic"], "s--", color=INK2, ms=4, label="AIC")
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("information criterion")
    for k in ks:
        f = c["F"][k]["feats"]
        for s in range(k):
            axes[2].plot(k, f.loc[s, "dwell_ms"], "o", color=state_colour(c, k, s), ms=5, mec="white", mew=0.5)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("mean dwell time (ms)")
    for ax in axes:
        ax.set_xlabel("number of states K")
        ax.grid(alpha=0.3, lw=0.5)
    fig.suptitle("Model order (BIC/AIC treat 5 ms bins as independent; smoothing makes them favour large K)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    save_fig(fig, c["out"] / "hmm_model_selection.png", dpi=120)
    plt.close(fig)


def fig_lineage(c):
    ks, ref = c["ks"], c["ref_k"]
    others = [k for k in ks if k != ref]
    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(2, len(others), height_ratios=[1.6, 1.0])
    ax = fig.add_subplot(gs[0, :])
    occ_max = max(c["F"][k]["feats"]["occupancy"].max() for k in ks)
    used = sorted({int(x) for k in ks for x in c["colours"][k]})
    row = {col: i for i, col in enumerate(used)}            # one row per colour in use (greys last)
    grey0 = min([x for x in used if x >= 8], default=None)
    for ka, kb in zip(ks[:-1], ks[1:]):
        for col in set(c["colours"][ka]) & set(c["colours"][kb]):
            if col < 8:
                ax.plot([ka, kb], [row[col], row[col]], color=GRID, lw=1.5, zorder=1)
    for k in ks:
        f = c["F"][k]["feats"]
        sim = {}
        if k in c["sims"]:
            S, r_, c_ = c["sims"][k]
            sim = {int(i): S[i, j] for i, j in zip(r_, c_)}
        for s in range(k):
            y = row[int(c["colours"][k][s])]
            ax.scatter(k, y, s=40 + 900 * f.loc[s, "occupancy"] / occ_max, color=state_colour(c, k, s),
                       edgecolor=INK if k == ref else "white", lw=1.2 if k == ref else 0.8, zorder=2)
            txt = f"{np.degrees(f.loc[s, 'phase']):.0f}°" + (f"  s={sim[s]:.2f}" if s in sim else "")
            ax.text(k + 0.08, y + 0.3, txt, fontsize=6, color=INK2)
    fr = c["F"][ref]["feats"]
    for s in range(ref):
        ax.text(ks[-1] + 0.35, row[int(c["colours"][ref][s])], fr.loc[s, "label"], va="center", fontsize=7, color=INK)
    if grey0 is not None:
        ax.text(ks[-1] + 0.35, row[grey0], f"no K={ref} counterpart (redundant or unmatched)", va="center",
                fontsize=7, color=INK2)
    ax.set_xlim(ks[0] - 0.3, ks[-1] + 2.8)
    ax.set_ylim(len(used) - 0.5, -0.7)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}\n(reference)" if k == ref else str(k) for k in ks])
    ax.set_xlabel("number of states K")
    ax.set_yticks(range(len(used)))
    ref_no = {int(col): i + 1 for i, col in enumerate(sorted(c["colours"][ref]))}   # numbering of the unit state cards
    ax.set_yticklabels([f"K={ref} state {ref_no[x]}" if x in ref_no else "unmatched" for x in used])
    ax.set_title(f"Every model matched to the K={ref} reference (outlined); dot area = occupancy; number = preferred φ "
                 f"over all conditions; s = similarity to the matched K={ref} state (max 2; grey if < 1.5 or no partner)",
                 loc="left")
    for i, k in enumerate(others):
        axs = fig.add_subplot(gs[1, i])
        S, r_, c_ = c["sims"][k]
        oa, ob = np.argsort(c["colours"][k]), np.argsort(c["colours"][ref])
        axs.imshow(S[oa][:, ob], cmap="Greys", vmin=-1, vmax=2, interpolation="nearest")
        inv_a, inv_b = np.argsort(oa), np.argsort(ob)
        for a_, b_ in zip(r_, c_):
            axs.add_patch(Rectangle((inv_b[b_] - 0.5, inv_a[a_] - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        axs.set_xticks(range(ref))
        axs.set_yticks(range(k))
        axs.set_xticklabels([str(c["colours"][ref][s] + 1) for s in ob], fontsize=6)
        axs.set_yticklabels([str(c["colours"][k][s] + 1) if c["colours"][k][s] < 8 else "grey" for s in oa], fontsize=6)
        axs.set_xlabel(f"K={ref} colour")
        axs.set_ylabel(f"K={k} colour")
        axs.set_title(f"similarity K={k} vs K={ref}", fontsize=8)
    fig.tight_layout()
    save_fig(fig, c["out"] / "hmm_state_lineage.png", dpi=115)
    plt.close(fig)


# ----------------------------------------------------------------------------------- Bush & Ramirez style

def fig_phase_tuning(c):
    conds = [n for n in c["conditions"] if n in c["curves"]]
    rows = unit_rows(c)
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, len(conds), height_ratios=[2.2, 1.3])
    for j, name in enumerate(conds):
        ax = fig.add_subplot(gs[0, j])
        R = c["curves"][name][:, rows].T
        mn, mx = np.nanmin(R, axis=1, keepdims=True), np.nanmax(R, axis=1, keepdims=True)
        R = (R - mn) / np.maximum(mx - mn, 1e-9)
        ax.imshow(R, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest",
                  extent=[-np.pi, np.pi, len(rows), 0])
        phase_axis(ax)
        ax.axvline(0, color="white", lw=0.6, ls=":")
        n_cyc = int((c["cond"] == name).sum())
        ax.set_title(f"{name} ({n_cyc} cycles)")
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels([f"#{u}" for u in c["order_units"]] if j == 0 else [], fontsize=5)
    tu = c["tuning"]
    base = tu[tu["condition"] == "baseline"]
    cls_col = {"inspiratory": "#1baf7a", "expiratory": "#2a78d6", "tonic": "#9a9893"}
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    for cl, g in base.groupby("class"):
        ax.scatter(g["preferred_phase"], g["modulation_L"], color=cls_col[cl], s=18, label=cl, edgecolor="white", lw=0.4)
    ax.set_title("baseline: preferred φ (angle) and modulation L (radius)", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(-0.35, -0.2))
    ax = fig.add_subplot(gs[1, 1])
    counts = tu.groupby(["condition", "class"]).size().unstack(fill_value=0).reindex(conds)
    bottom = np.zeros(len(conds))
    for cl in ["inspiratory", "expiratory", "tonic"]:
        v = counts.get(cl, 0 * bottom).to_numpy() if cl in counts else 0 * bottom
        ax.bar(range(len(conds)), v, bottom=bottom, color=cls_col[cl], label=cl, edgecolor="white", lw=0.5)
        bottom += v
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha="right")
    ax.set_ylabel("units")
    ax.set_title("2025 classes (tonic: L < 0.2)", fontsize=8)
    ax = fig.add_subplot(gs[1, 2:4])
    piv = tu.pivot(index="unit_id", columns="condition", values="modulation_L")[conds]
    for u, row in piv.iterrows():
        ax.plot(range(len(conds)), row.to_numpy(), color=GRID, lw=0.8)
    ax.plot(range(len(conds)), piv.median().to_numpy(), "o-", color=INK, lw=1.6, ms=4, label="median")
    ax.axhline(0.2, color=INK2, ls=":", lw=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel("respiratory modulation L")
    ax.legend(frameon=False)
    ax = fig.add_subplot(gs[1, 4])
    pp = tu.pivot(index="unit_id", columns="condition", values="preferred_phase")
    ll = tu.pivot(index="unit_id", columns="condition", values="modulation_L")
    ok = (ll["baseline"] >= 0.2) & (ll["gasping"] >= 0.2)
    ax.scatter(pp.loc[ok, "baseline"], pp.loc[ok, "gasping"], color=INK, s=14)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color=GRID, lw=0.8)
    phase_axis(ax, label=False)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("preferred φ, baseline")
    ax.set_ylabel("preferred φ, gasping")
    ax.set_title(f"units modulated in both (n={int(ok.sum())})", fontsize=8)
    fig.suptitle("Phase tuning per condition (cibrrig get_phase_curve; rows scaled 0-1 per unit and condition; "
                 "units ordered by baseline preferred φ, tonic last)", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, c["out"] / "bush_phase_tuning.png", dpi=115)
    plt.close(fig)


def _phi_line3d(ax, X, phi, lw=0.6, alpha=0.6):
    seg = np.stack([X[:-1], X[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=PHI_CMAP, lw=lw, alpha=alpha)
    lc.set_array(phi[:-1])
    lc.set_clim(-np.pi, np.pi)
    ax.add_collection3d(lc)
    return lc


def fig_pca(c):
    P, ph, bcond, dt = c["P"], c["ph"], c["bcond"], c["pca_bin_s"]
    conds = [n for n in c["conditions"] if (bcond == n).any()]
    ncol = max(5, len(conds))
    fig = plt.figure(figsize=(18 * ncol / 5, 13))
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.0, 1.6, 1.1])
    ax = fig.add_subplot(gs[0, 0])
    ev = c["pca"].explained_variance_ratio_
    ax.bar(np.arange(1, ev.size + 1), ev * 100, color=INK2)
    ax.plot(np.arange(1, ev.size + 1), np.cumsum(ev) * 100, "o-", color=INK, ms=3)
    ax.set_xlabel("PC (fit on baseline)")
    ax.set_ylabel("% variance (bars) / cumulative")
    for i in range(3):
        ax = fig.add_subplot(gs[0, 1 + i])
        for name in conds:
            x, m = phase_binned(P[:, i], ph, bcond == name)
            ax.plot(x, m, color=c["cond_colours"][name], lw=1.5, label=name)
        phase_axis(ax)
        ax.set_ylabel(f"PC{i + 1}")
        if i == 0:
            ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(gs[0, 4])
    for name in conds:
        x, m = phase_binned(c["speed"], ph, bcond == name, stat="median")
        ax.plot(x, m, color=c["cond_colours"][name], lw=1.5)
    phase_axis(ax)
    ax.set_ylabel("speed in PC1-3 (a.u./s), median")
    lim = np.nanpercentile(P[np.isfinite(ph), :3], [1, 99], axis=0)
    for j, name in enumerate(conds):
        ax = fig.add_subplot(gs[1, j], projection="3d")
        for i in cycles_of(c, name, 30):
            b = cycle_bins(c, i, dt)
            lc = _phi_line3d(ax, P[b, :3], ph[b])
        x, m = phase_binned(P[:, :3], ph, bcond == name)
        ax.plot(m[:, 0], m[:, 1], m[:, 2], color=INK, lw=2.0)
        ax.scatter(*c["ioff_point"], color=INK, s=40, marker="X", depthshade=False)
        ax.set_xlim(lim[:, 0])
        ax.set_ylim(lim[:, 1])
        ax.set_zlim(lim[:, 2])
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.set_zlabel("PC3", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{name}: 30 cycles by φ, black = mean", fontsize=8)
        ax.view_init(elev=22, azim=-55)
    cax = fig.add_axes([0.972, 0.42, 0.007, 0.18])
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label("φ")
    # distances (Bush 2024 Fig 3i; 2025 inspiration-off point)
    xb, mb = phase_binned(P[:, :3], ph, bcond == "baseline")
    edges = np.linspace(-np.pi, np.pi, mb.shape[0] + 1)
    b = np.clip(np.digitize(ph, edges) - 1, 0, mb.shape[0] - 1)
    d_mean = np.linalg.norm(P[:, :3] - mb[b], axis=1)
    d_ioff = np.linalg.norm(P[:, :3] - c["ioff_point"], axis=1)
    for col, (d, lab) in enumerate([(d_mean, "distance to baseline mean trajectory"),
                                    (d_ioff, "distance to baseline inspiration-off point")]):
        ax = fig.add_subplot(gs[2, col * 2:col * 2 + 2])
        for name in conds:
            x, m = phase_binned(d, ph, bcond == name, stat="median")
            ax.plot(x, m, color=c["cond_colours"][name], lw=1.5, label=name)
        phase_axis(ax)
        ax.set_ylabel(lab + " (median)")
        if col == 0:
            ax.legend(frameon=False, fontsize=7)
    ax = fig.add_subplot(gs[2, 4:])
    names, M = c["pa_names"], c["pa_M"]
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("subspace similarity\n(mean cos of 3 principal angles)", fontsize=8)
    fig.suptitle("PCA as Bush & Ramirez 2024: 5 ms bins, Gaussian σ 10 ms, √, fit on baseline and projected over "
                 "the whole window; X = baseline inspiration-off point (2025)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.965, 0.98))
    save_fig(fig, c["out"] / "bush_pca_trajectories.png", dpi=110)
    plt.close(fig)


def fig_gasping(c):
    P, ph, bcond, dt, Z = c["P"], c["ph"], c["bcond"], c["pca_bin_s"], c["Z"]
    tu = c["tuning"]
    rows = unit_rows(c)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    piv = tu.pivot(index="unit_id", columns="condition", values="rate_hz")
    base = tu[tu["condition"] == "baseline"].set_index("unit_id")
    tonic = base["modulation_L"] < 0.2
    sc = ax.scatter(np.sqrt(piv["baseline"]), np.sqrt(piv["gasping"]), c=base.loc[piv.index, "preferred_phase"],
                    cmap=PHI_CMAP, vmin=-np.pi, vmax=np.pi, s=26, edgecolor="white", lw=0.5)
    ax.scatter(np.sqrt(piv.loc[tonic[tonic].index, "baseline"]), np.sqrt(piv.loc[tonic[tonic].index, "gasping"]),
               facecolor="none", edgecolor=INK2, s=40, lw=0.8, label="tonic at baseline")
    m = np.sqrt(np.nanmax(piv[["baseline", "gasping"]].to_numpy())) * 1.05
    ax.plot([0, m], [0, m], color=GRID, lw=0.8)
    ax.set_xlabel("√rate, baseline (√Hz)")
    ax.set_ylabel("√rate, gasping (√Hz)")
    ax.legend(frameon=False, fontsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.04).set_label("baseline preferred φ")
    ax.set_title("per-unit rate, eupnea vs gasps (Bush 2024 Fig 7d)", fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    for i in cycles_of(c, "baseline", 40):
        b = cycle_bins(c, i, dt)
        ax.plot(P[b, 0], P[b, 2], color=c["cond_colours"]["baseline"], lw=0.4, alpha=0.35)
    for i in cycles_of(c, "gasping"):
        b = cycle_bins(c, i, dt)
        seg = np.stack([P[b[:-1]][:, [0, 2]], P[b[1:]][:, [0, 2]]], axis=1)
        lc = LineCollection(seg, cmap=PHI_CMAP, lw=0.9)
        lc.set_array(ph[b[:-1]])
        lc.set_clim(-np.pi, np.pi)
        ax.add_collection(lc)
    for name in ["baseline", "gasping", c["conditions"][-1]]:
        x, mm = phase_binned(P[:, [0, 2]], ph, bcond == name)
        ax.plot(mm[:, 0], mm[:, 1], color=c["cond_colours"][name], lw=2.2, label=f"{name} mean")
    ax.scatter(*c["ioff_point"][[0, 2]], color=INK, marker="X", s=50, zorder=5, label="baseline I-off point")
    ax.autoscale()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC3")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("PC1-PC3: baseline cycles (blue), gasps (coloured by φ)", fontsize=8)
    ax = fig.add_subplot(gs[0, 2])
    for name in ["baseline", "gasping", c["conditions"][-1]]:
        x, mm = phase_binned(c["speed"], ph, bcond == name, stat="median")
        ax.plot(x, mm, color=c["cond_colours"][name], lw=1.5, label=name)
    phase_axis(ax)
    ax.set_ylabel("speed in PC1-3 (a.u./s), median")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("trajectory speed vs φ (Bush 2024 Fig 7g)", fontsize=8)
    ax = fig.add_subplot(gs[0, 3])
    t_on = c["dia_df"]["on_sec"].to_numpy()
    g_idx = cycles_of(c, "gasping")
    win = np.arange(int(-0.3 / dt), int(1.0 / dt))
    for name, idx in [("baseline", cycles_of(c, "baseline")), ("gasping", g_idx)]:
        tr = np.stack([P[int(t_on[i] / dt) + win, 0] for i in idx if int(t_on[i] / dt) + win[-1] < P.shape[0]])
        ax.plot(win * dt, tr.mean(axis=0), color=c["cond_colours"][name], lw=1.6, label=name)
        ax.fill_between(win * dt, *np.percentile(tr, [25, 75], axis=0), color=c["cond_colours"][name], alpha=0.15, lw=0)
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("time from inspiration onset (s)")
    ax.set_ylabel("PC1 (mean, IQR)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("PC1 around onset (Bush 2024 Fig 7e)", fontsize=8)
    corr = {}
    for j, name in enumerate(["baseline", "gasping", c["conditions"][-1]]):
        s = bcond == name
        C = np.corrcoef(Z[s][:, rows].T)
        corr[name] = C
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"pairwise r, {name} (√ rates, σ 10 ms)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[1, 3])
    iu = np.triu_indices(len(rows), 1)
    xa, ya = corr["baseline"][iu], corr["gasping"][iu]
    ok = np.isfinite(xa) & np.isfinite(ya)
    ax.scatter(xa[ok], ya[ok], s=6, color=INK, alpha=0.6, edgecolor="none")
    if ok.sum() > 2:
        p = np.polyfit(xa[ok], ya[ok], 1)
        xx = np.linspace(xa[ok].min(), xa[ok].max(), 10)
        ax.plot(xx, np.polyval(p, xx), color=c["cond_colours"]["gasping"], lw=1.4,
                label=f"slope {p[0]:.2f}, r = {np.corrcoef(xa[ok], ya[ok])[0, 1]:.2f}")
    ax.plot([-1, 1], [-1, 1], color=GRID, lw=0.8)
    ax.set_xlabel("pairwise r, baseline")
    ax.set_ylabel("pairwise r, gasping")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("correlation structure kept in gasping? (Bush 2024 Fig 7i-k)", fontsize=8)
    fig.suptitle(f"Gasping vs eupnea ({len(g_idx)} Bush-rule gasps)", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, c["out"] / "bush_gasping.png", dpi=115)
    plt.close(fig)


def fig_hmm_vs_pca(c):
    P, bcond = c["P"], c["bcond"]
    show_k = [k for k in (4, 6, 8) if k in c["ks"]]
    conds = [n for n in c["conditions"] if n != "anoxia"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(conds), len(show_k), figsize=(4.6 * len(show_k), 3.9 * len(conds)),
                             sharex=True, sharey=True, squeeze=False)
    for r, name in enumerate(conds):
        idx = np.flatnonzero(bcond == name)
        sub = idx if idx.size <= 6000 else np.sort(rng.choice(idx, 6000, replace=False))
        for j, k in enumerate(show_k):
            ax = axes[r, j]
            seq = c["F"][k]["seq"]
            cols = [state_colour(c, k, s) for s in seq[sub]]
            ax.scatter(P[sub, 0], P[sub, 2], c=cols, s=2, alpha=0.5, edgecolor="none")
            for s in range(k):
                m = seq[idx] == s
                if m.sum() > 20:
                    ax.scatter(P[idx[m], 0].mean(), P[idx[m], 2].mean(), color=state_colour(c, k, s), s=90,
                               edgecolor=INK, lw=0.8, zorder=4)
            ax.scatter(*c["ioff_point"][[0, 2]], color=INK, marker="X", s=50, zorder=5)
            if r == 0:
                ax.set_title(f"K = {k}")
            if j == 0:
                ax.set_ylabel(f"{name}\nPC3")
            if r == len(conds) - 1:
                ax.set_xlabel("PC1")
    fig.suptitle("HMM states (matched colours) in baseline-fit PC space; big dots = state means in that condition; "
                 "X = baseline inspiration-off point", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, c["out"] / "hmm_vs_pca.png", dpi=110)
    plt.close(fig)


RASTER_PRE_S, RASTER_POST_S = 0.2, 0.6     # breath raster window around the alignment event
ALIGN = {"on": ("on_sec", "inspiration onset", "I$_{on}$"), "off": ("off_sec", "inspiration offset", "I$_{off}$")}


def breath_state_rows(c, k, align, pre=RASTER_PRE_S, post=RASTER_POST_S):
    """Viterbi state in each 5 ms bin from -pre to +post around every breath's inspiration onset or offset, one row
    per breath in recording order (-1 where the window leaves the recording)."""
    dt, seq = c["hmm_bin_s"], c["F"][k]["seq"]
    n0, n1 = int(round(pre / dt)), int(round(post / dt))
    rows = np.full((len(c["dia_df"]), n0 + n1), -1)
    for i, t in enumerate(c["dia_df"][ALIGN[align][0]].to_numpy()):
        b = int(np.floor(t / dt))
        if b - n0 >= 0 and b + n1 <= seq.size:
            rows[i] = seq[b - n0:b + n1]
    return rows


def draw_breath_raster(ax, c, k, align, labels=True, marks=True, pre=RASTER_PRE_S, post=RASTER_POST_S,
                       neighbours=False):
    """marks: the other edge of this breath and the next onset (black). neighbours: instead, every onset (black) and
    offset (white) of the previous, this and the next breath inside the window."""
    rows = breath_state_rows(c, k, align, pre, post)
    cols = np.array([matplotlib.colors.to_rgb(state_colour(c, k, s)) for s in range(k)] + [(0.9, 0.9, 0.89)])
    n = rows.shape[0]
    ax.imshow(cols[rows], aspect="auto", interpolation="nearest", extent=[-pre * 1e3, post * 1e3, n, 0])
    d = c["dia_df"]
    y = np.arange(n) + 0.5
    t = d[ALIGN[align][0]]
    if neighbours:
        for col_name, colour in (("on_sec", INK), ("off_sec", "white")):
            for shift in (-1, 0, 1):
                x = (d[col_name].shift(-shift) - t).to_numpy() * 1e3
                m = (x > -pre * 1e3) & (x < post * 1e3) & (np.abs(x) > 1e-6)
                ax.scatter(x[m], y[m], s=0.7, color=colour, lw=0, alpha=0.85)
    elif marks:
        other = d["off_sec"] if align == "on" else d["on_sec"]
        for x in ((other - t).to_numpy() * 1e3, (d["on_sec"].shift(-1) - t).to_numpy() * 1e3):
            m = (x > -pre * 1e3) & (x < post * 1e3)
            ax.scatter(x[m], y[m], s=0.6, color=INK, lw=0, alpha=0.7)
    ax.axvline(0, color=INK, lw=1.4, ls=(0, (4, 3)))
    if labels:
        condition_block_labels(ax, c, post * 1e3)
    ax.set_xlim(-pre * 1e3, post * 1e3)
    ax.set_ylim(n, 0)


def condition_block_labels(ax, c, x0, gap_frac=0.045, colours=None, fontsize=6.5):
    """Label each block of consecutive breaths with its condition, raw time span and breath count, to the right of
    data x = x0 on a breath-row axis, spreading labels of short blocks apart with leader lines. With `colours`
    (condition -> colour), a bar in the condition's colour spans each block's rows next to its label."""
    cond, d, off, n = c["cond"], c["dia_df"], c["time_offset_s"], len(c["cond"])
    change = np.flatnonzero(cond[1:] != cond[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, n]
    mid = (starts + ends) / 2
    gap = gap_frac * n                     # keep two-line labels of short blocks apart
    ypos = mid.copy()
    for i in range(1, ypos.size):
        ypos[i] = max(ypos[i], ypos[i - 1] + gap)
    ypos -= max(0, ypos[-1] - (n - gap / 2))
    for i in range(ypos.size - 2, -1, -1):
        ypos[i] = min(ypos[i], ypos[i + 1] - gap)
    span = np.diff(ax.get_xlim())[0] if ax.get_xlim()[1] > ax.get_xlim()[0] else 800.0
    dx = 0.0175 * span
    if colours is not None:
        bx, bw = x0 + 0.5 * dx, 1.6 * dx
        for s0, s1 in zip(starts, ends):
            name = cond[s0]
            ax.add_patch(Rectangle((bx, s0), bw, s1 - s0, facecolor=colours.get(name, "white") if name else "white",
                                   edgecolor=GRID if not name else "none", lw=0.6, clip_on=False))
        x0 = bx + bw
    for s0, s1, m, y in zip(starts, ends, mid, ypos):
        if not cond[s0] and s1 - s0 < 2:      # the last breath has no complete cycle: not a real block
            continue
        name = cond[s0] or "transition"
        t0, t1 = d["on_sec"].iloc[s0] + off, d["on_sec"].iloc[s1 - 1] + off
        ax.plot([x0, x0 + dx, x0 + 1.7 * dx], [m, m, y], color=INK2, lw=0.5, clip_on=False)
        ax.text(x0 + 2 * dx, y, f"{name}\n{t0:.0f}–{t1:.0f} s ({s1 - s0})", va="center", ha="left", fontsize=fontsize,
                color=INK2 if name == "transition" else INK, clip_on=False)


def state_legend(fig, c, k, y=0.0, ncol=2):
    order = np.argsort(c["colours"][k])
    f = c["F"][k]["feats"]
    handles = [Rectangle((0, 0), 1, 1, color=state_colour(c, k, s)) for s in order]
    names = [f"{f.loc[s, 'label']}  ({f.loc[s, 'rate_mean_hz']:.0f} Hz)" for s in order]
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=7,
               handlelength=1.2)


def fig_breath_raster_wide(c, pre=0.4, post=0.4):
    """Reference model only, a symmetric window around onset and offset, with the neighbouring breaths' edges."""
    k, n = c["ref_k"], len(c["dia_df"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 9.2), sharey=True)
    for j, align in enumerate(("on", "off")):
        _, name, sym = ALIGN[align]
        draw_breath_raster(axes[j], c, k, align, labels=(j == 1), pre=pre, post=post, neighbours=True)
        axes[j].set_title(f"aligned to {name} ({sym})", color=INK)
        axes[j].set_xlabel(f"time from {name} (ms)")
        axes[j].set_xticks(np.arange(-pre * 1e3, post * 1e3 + 1, 100))
    axes[0].set_ylabel("breath (recording order)")
    fig.suptitle(f"{k}-state HMM, {n} breaths, −{pre * 1e3:.0f} to +{post * 1e3:.0f} ms", x=0.01, ha="left")
    state_legend(fig, c, k, ncol=2)
    fig.text(0.01, 0.112, "dots: inspiration onsets (black) and offsets (white) of the previous, this and the next "
             "breath; light grey: window leaves the recording", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save_fig(fig, c["out"] / f"breath_raster_K{k}_pm{pre * 1e3:.0f}ms.png", dpi=150)
    plt.close(fig)


def fig_breath_rasters(c):
    fig_breath_raster_wide(c)
    n = len(c["dia_df"])
    for k in c["ks"]:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.8), sharey=True)
        for j, align in enumerate(("on", "off")):
            _, name, sym = ALIGN[align]
            draw_breath_raster(axes[j], c, k, align, labels=(j == 1))
            axes[j].set_title(f"aligned to {name} ({sym})", color=INK)
            axes[j].set_xlabel(f"time from {name} (ms)")
        axes[0].set_ylabel("breath (recording order)")
        fig.suptitle(f"{k}-state HMM, {n} breaths", x=0.01, ha="left")
        state_legend(fig, c, k, ncol=2)
        fig.text(0.01, 0.112, "dots: the other edge of the breath (offset in the onset panel, onset in the offset "
                 "panel) and the next onset; light grey: window leaves the recording", fontsize=6.5, color=INK2)
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        save_fig(fig, c["out"] / f"breath_raster_K{k}.png", dpi=150)
        plt.close(fig)
    ks = c["ks"]
    for align in ("on", "off"):
        _, name, sym = ALIGN[align]
        fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks) + 1.4, 8.5), sharey=True)
        for j, (ax, k) in enumerate(zip(axes, ks)):
            draw_breath_raster(ax, c, k, align, labels=(j == len(ks) - 1), marks=False)
            ax.set_title(f"K = {k}")
            ax.set_xlabel(f"ms from {sym}")
        axes[0].set_ylabel("breath (recording order)")
        fig.suptitle(f"Breath-by-breath HMM states around {name} ({n} breaths; colours matched to the K = "
                     f"{c['ref_k']} reference; mid/dark grey = no K = {c['ref_k']} counterpart, light grey = outside "
                     f"the recording)", x=0.01, ha="left")
        fig.tight_layout()
        save_fig(fig, c["out"] / f"breath_raster_i{align}_all_K.png", dpi=130)
        plt.close(fig)


def write_breath_raster_viewer(c):
    """Scrollable HTML version of the breath rasters (template: breath_raster_viewer.html next to this file)."""
    d, off = c["dia_df"], c["time_offset_s"]
    rows = {}
    for k in c["ks"]:
        rows[k] = {}
        for align in ("on", "off"):
            r = breath_state_rows(c, k, align)
            rows[k][align] = base64.b64encode(np.where(r < 0, 255, r).astype(np.uint8).tobytes()).decode()
    feats = {k: c["F"][k]["feats"] for k in c["ks"]}
    data = {
        "n": len(d), "nbins": int(round((RASTER_PRE_S + RASTER_POST_S) / c["hmm_bin_s"])),
        "pre_ms": RASTER_PRE_S * 1e3, "post_ms": RASTER_POST_S * 1e3, "bin_ms": c["hmm_bin_s"] * 1e3, "ks": c["ks"],
        "ref_k": c["ref_k"],
        "on_s": (d["on_sec"] + off).round(4).tolist(), "off_s": (d["off_sec"] + off).round(4).tolist(),
        "cond": [x or "transition" for x in c["cond"]], "gasp": d["gasp"].astype(int).tolist(), "rows": rows,
        "colours": {k: [state_colour(c, k, s) for s in range(k)] for k in c["ks"]},
        "labels": {k: [f"{feats[k].loc[s, 'label']} ({feats[k].loc[s, 'rate_mean_hz']:.0f} Hz)" for s in range(k)]
                   for k in c["ks"]},
        "order": {k: [int(s) for s in np.argsort(c["colours"][k])] for k in c["ks"]},
        "units": f"{len(c['ids'])} single units",
    }
    template = (Path(__file__).resolve().parent / "breath_raster_viewer.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (c["out"] / "breath_raster_viewer.html").write_text(html, encoding="utf-8")


N2_RAW_S = (1800.0, 1815.0)      # logged 15 s N2 (experiment log 30:00-30:15)


def fig_breathing_epochs(c):
    """Breath-by-breath breathing rate coloured by condition, over the condition timeline (name, span, duration,
    breaths, median rate) on the same time axis; sighs by cibrrig's rule, the logged N2 window shaded."""
    import pandas as pd
    from epoch_change_points import sighs
    d, off, col = c["dia_df"], c["time_offset_s"], c["cond_colours"]
    t = d["on_sec"].to_numpy() + off
    rate = 60.0 / d["ibi_s"].to_numpy()
    cond = c["cond"]
    spans = []
    for n in c["conditions"]:
        idx = np.flatnonzero(cond == n)
        if idx.size:
            t0, t1 = t[idx[0]], t[min(idx[-1] + 1, len(t) - 1)]
            spans.append((n, t0, t1, idx.size, float(np.median(rate[idx]))))
    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12}):
        fig, (ax, axb) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 2.3]})
        ax.axvspan(*N2_RAW_S, color=GRID, alpha=0.6, lw=0)
        ax.text(sum(N2_RAW_S) / 2, 1.0, "N2\n15 s", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=10.5, color=INK2)
        un = cond == ""
        ax.scatter(t[un], rate[un], s=7, color="#c3c2b7", lw=0, label="no condition")
        for n, *_ in spans:
            m = cond == n
            ax.scatter(t[m], rate[m], s=9, color=col[n], lw=0)
        med = pd.Series(rate).rolling(9, center=True, min_periods=1).median().to_numpy()
        ax.plot(t, med, color=INK, lw=0.9, alpha=0.8, label="9-breath median")
        sg = sighs(d) & ~d["gasp"].to_numpy().astype(bool)     # the rule also flags 2 of the gasps
        ax.scatter(t[sg], np.full(sg.sum(), 258), marker="v", s=36, color=INK, lw=0,
                   label="sigh (cibrrig rule, gasps excluded)")
        ax.set_ylim(0, 266)
        ax.set_yticks([0, 50, 100, 150, 200, 250])
        ax.set_ylabel("breaths / min")
        ax.legend(frameon=False, loc="lower left", fontsize=10.5, ncol=3, bbox_to_anchor=(0.0, -0.02))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        # timeline: blocks on one row, labels alternating below the bar
        axb.set_ylim(0, 1)
        axb.axis("off")
        for i, (n, t0, t1, nb, mr) in enumerate(spans):
            axb.add_patch(Rectangle((t0, 0.8), t1 - t0, 0.2, color=col[n], lw=0))
            y = 0.72 if i % 2 == 0 else 0.38
            axb.plot([(t0 + t1) / 2] * 2, [0.8, y + 0.03], color=INK2, lw=0.6)
            axb.text(t0 if t1 - t0 > 30 else (t0 + t1) / 2 - 8, y,
                     f"{n}\n{t0:.0f}–{t1:.0f} s ({t1 - t0:.0f} s)\n{nb} breaths, {mr:.0f}/min",
                     ha="left", va="top", fontsize=10.5, color=INK, linespacing=1.15)
        lo, hi = t[0] - 2, t[-1] + 2
        ax.set_xlim(lo, hi)
        axb.set_xlabel("raw-run time (s)")
        axb.axis("on")
        axb.set_yticks([])
        axb.spines[["left", "right", "top"]].set_visible(False)
        fig.suptitle("e097 anoxia #1: breathing rate and conditions", x=0.01, ha="left", fontsize=14)
        fig.tight_layout(h_pad=0.4)
        save_fig(fig, c["out"] / "breathing_epochs.png", dpi=200)
        plt.close(fig)


def make_all(c):
    for f in (fig_overview, fig_zoom, fig_hmm_models, fig_lineage, fig_phase_tuning, fig_pca, fig_gasping,
              fig_hmm_vs_pca, fig_breath_rasters, write_breath_raster_viewer, fig_breathing_epochs):
        f(c)
    import figures_signatures
    figures_signatures.make_all(c)
