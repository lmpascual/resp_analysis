"""When does each unit's firing return to its pre-anoxia baseline?

For one sorted window that contains an anoxia challenge, this compares every unit's firing rate
(sliding 10 s windows) with its own rate before the gas switch and reports when it leaves and
re-enters its baseline band. Use it to decide how much recovery time a window needs.

    python anoxia_recovery.py <outputs>/e097/intact_window_v2 ^
        --switch-s 60 --gasp-start-s 85 --gasp-end-s 130 --time-offset-s 1740 ^
        --breathing-json <outputs>/e097/e097_whole_session_data.json ^
        --out <outputs>/e097/anoxia_recovery/intact_window_kilosort4

All times on the command line are file time (seconds from the start of the sorted window).

Definitions (also written to the output README):
- rate r(t): spikes in a `--win-s` window centred on t, stepped by `--step-s`.
- baseline: windows that lie entirely in [--baseline-start-s, --switch-s). mu = mean rate there,
  sigma = SD of the windowed rate there.
- band: mu +/- max(k*sigma, rel*mu, floor_hz)  (defaults k=3, rel=0.25, floor 0.5 Hz).
- perturbed: r(t) stays outside the band for >= --min-out-s somewhere between the switch and
  --gasp-end-s + --perturb-grace-s.
- return time: first t after --gasp-end-s from which r(t) stays inside the band for
  >= --sustain-s. Units that never do are "not returned by end of window".
- stable-from time: earliest t after --gasp-end-s after which r(t) never leaves the band
  again (for >= --min-out-s) until the end of the window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", help="npxsort run output_dir (holds <sorter>/alf, <sorter>/units.tsv)")
    p.add_argument("--sorter", default="kilosort4")
    p.add_argument("--labels", default="good,mua", help="unit labels to include (comma list); 'all' for every unit")
    p.add_argument("--include-reviewed-single", action="store_true", default=True,
                   help="also include units the review marked 'single', whatever their label (default on)")
    p.add_argument("--baseline-start-s", type=float, default=0.0)
    p.add_argument("--switch-s", type=float, required=True, help="gas switch (anoxia onset), file time")
    p.add_argument("--gasp-start-s", type=float, required=True)
    p.add_argument("--gasp-end-s", type=float, required=True)
    p.add_argument("--end-s", type=float, default=None, help="default: last spike time")
    p.add_argument("--time-offset-s", type=float, default=0.0, help="added to file time for raw-run time labels")
    p.add_argument("--marks", default="", help="extra vertical marks, 'name:t_s,name:t_s' (file time)")
    p.add_argument("--win-s", type=float, default=10.0)
    p.add_argument("--step-s", type=float, default=2.0)
    p.add_argument("--k-sigma", type=float, default=3.0)
    p.add_argument("--rel", type=float, default=0.25)
    p.add_argument("--floor-hz", type=float, default=0.5)
    p.add_argument("--min-out-s", type=float, default=10.0)
    p.add_argument("--perturb-grace-s", type=float, default=10.0)
    p.add_argument("--sustain-s", type=float, default=30.0)
    p.add_argument("--breathing-json", default=None, help="session JSON with timeline.time_s / resp_rate_bpm (raw-run time)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def windowed_rates(spike_times, centres, win_s):
    lo = np.searchsorted(spike_times, centres - win_s / 2)
    hi = np.searchsorted(spike_times, centres + win_s / 2)
    return (hi - lo) / win_s


def runs(mask):
    """(start, stop) index pairs of True runs in a boolean array (stop exclusive)."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def analyse_unit(r, centres, a, step):
    base = (centres - a.win_s / 2 >= a.baseline_start_s) & (centres + a.win_s / 2 <= a.switch_s)
    mu, sigma = float(r[base].mean()), float(r[base].std(ddof=1))
    half = max(a.k_sigma * sigma, a.rel * mu, a.floor_hz)
    lo, hi = mu - half, mu + half
    out = (r < lo) | (r > hi)
    min_out = int(round(a.min_out_s / step))
    sustain = int(round(a.sustain_s / step))

    # perturbation during the challenge
    chal = (centres >= a.switch_s) & (centres <= a.gasp_end_s + a.perturb_grace_s)
    long_out = np.zeros_like(out)
    for s, e in runs(out):
        if e - s >= min_out:
            long_out[s:e] = True
    pert_idx = np.flatnonzero(long_out & chal)
    perturbed = len(pert_idx) > 0
    direction = ""
    peak_dev = np.nan
    if perturbed:
        dev = r[chal] - mu
        k = int(np.argmax(np.abs(dev)))
        peak_dev = float(r[chal][k])
        direction = "up" if dev[k] > 0 else "down"
        # mixed: both a long excursion above and below during the challenge
        ups = np.any(long_out & chal & (r > hi))
        downs = np.any(long_out & chal & (r < lo))
        if ups and downs:
            direction = "up+down"
    first_out = float(centres[pert_idx[0]]) if perturbed else np.nan

    post = np.flatnonzero(centres >= a.gasp_end_s)
    t_return = np.nan
    inside = ~out
    for i in post:
        j = min(i + sustain, len(r))
        if j - i >= sustain and inside[i:j].all():
            t_return = float(centres[i])
            break
    # stable-from: last long excursion after gasp end, then the band holds to the end
    t_stable = np.nan
    late_long = [(s, e) for s, e in runs(out) if e - s >= min_out and e > post[0]]
    if late_long:
        last_end = late_long[-1][1]
        if last_end < len(r) - sustain:
            t_stable = float(centres[last_end])
    else:
        t_stable = float(centres[post[0]])
    return dict(
        base_rate_hz=mu, base_sd_hz=sigma, band_lo_hz=lo, band_hi_hz=hi,
        perturbed=perturbed, direction=direction, first_out_s=first_out,
        challenge_extreme_rate_hz=peak_dev, return_s=t_return, stable_from_s=t_stable,
    )


def main():
    a = parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdir = Path(a.output_dir) / a.sorter
    st = np.load(sdir / "alf" / "spike_times.npy")
    sc = np.load(sdir / "alf" / "spike_clusters.npy")
    units = pd.read_csv(sdir / "units.tsv", sep="\t")
    review_path = Path(a.output_dir) / "review" / f"{a.sorter}_unit_review.tsv"
    review = pd.read_csv(review_path, sep="\t") if review_path.exists() else pd.DataFrame(columns=["unit_id", "verdict"])
    units = units.merge(review[["unit_id", "verdict"]], on="unit_id", how="left")

    if a.labels == "all":
        keep = np.ones(len(units), bool)
    else:
        keep = units["label"].isin(a.labels.split(",")).to_numpy().copy()
    if a.include_reviewed_single:
        keep |= (units["verdict"] == "single").to_numpy()
    keep &= ~units["verdict"].isin(["noise"]).to_numpy()
    units = units[keep].reset_index(drop=True)

    end_s = a.end_s if a.end_s is not None else float(st.max())
    step = a.step_s
    centres = np.arange(a.baseline_start_s + a.win_s / 2, end_s - a.win_s / 2 + 1e-9, step)

    order = np.argsort(sc, kind="stable")
    sc_sorted, st_sorted = sc[order], st[order]
    bounds = np.searchsorted(sc_sorted, units["unit_id"].to_numpy(), side="left"), \
        np.searchsorted(sc_sorted, units["unit_id"].to_numpy(), side="right")

    rows, rates = [], []
    for i, u in units.iterrows():
        t = np.sort(st_sorted[bounds[0][i]:bounds[1][i]])
        r = windowed_rates(t, centres, a.win_s)
        res = analyse_unit(r, centres, a, step)
        res.update(
            unit_id=int(u["unit_id"]), label=u["label"], verdict=u["verdict"] if isinstance(u["verdict"], str) else "",
            depth_um=u["depth_um"], num_spikes=int(u["num_spikes"]), found_by_both=bool(u.get("found_by_both", False)),
        )
        # rate in late-recovery blocks relative to baseline
        rows.append(res)
        rates.append(r)
    rates = np.array(rates)
    df = pd.DataFrame(rows)
    cols = ["unit_id", "label", "verdict", "found_by_both", "depth_um", "num_spikes", "base_rate_hz", "base_sd_hz",
            "band_lo_hz", "band_hi_hz", "perturbed", "direction", "first_out_s", "challenge_extreme_rate_hz",
            "return_s", "stable_from_s"]
    df = df[cols]
    for c in ["first_out_s", "return_s", "stable_from_s"]:
        df[c.replace("_s", "_raw_s")] = df[c] + a.time_offset_s
    df.to_csv(out_dir / "unit_recovery.tsv", sep="\t", index=False, float_format="%.3f")
    np.savez_compressed(out_dir / "rates.npz", centres_s=centres, rates_hz=rates, unit_ids=df["unit_id"].to_numpy())

    # population curves
    lo = df["band_lo_hz"].to_numpy()[:, None]
    hi = df["band_hi_hz"].to_numpy()[:, None]
    inside = (rates >= lo) & (rates <= hi)
    pert = df["perturbed"].to_numpy()
    base = (centres - a.win_s / 2 >= a.baseline_start_s) & (centres + a.win_s / 2 <= a.switch_s)
    logr = np.log2((rates + 0.2) / (df["base_rate_hz"].to_numpy()[:, None] + 0.2))
    basevec = np.log2(df["base_rate_hz"].to_numpy() + 0.2)
    popcorr = np.array([np.corrcoef(np.log2(rates[:, k] + 0.2), basevec)[0, 1] for k in range(len(centres))])
    pop = pd.DataFrame({
        "t_s": centres, "t_raw_s": centres + a.time_offset_s,
        "frac_in_band_all": inside.mean(0),
        "frac_in_band_perturbed": inside[pert].mean(0) if pert.any() else np.nan,
        "median_abs_log2_ratio_perturbed": np.median(np.abs(logr[pert]), axis=0) if pert.any() else np.nan,
        "popvec_corr_to_baseline": popcorr,
    })
    pop.to_csv(out_dir / "population_recovery.tsv", sep="\t", index=False, float_format="%.4f")

    breathing = None
    if a.breathing_json:
        d = json.load(open(a.breathing_json))
        tl = d["timeline"]
        bt = np.asarray(tl["time_s"], float) - a.time_offset_s
        br = np.asarray(tl["resp_rate_bpm"], float)
        m = (bt >= centres[0] - a.win_s) & (bt <= end_s)
        breathing = (bt[m], br[m])

    summary = summarise(df, pop, a, centres, base)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(df, rates, centres, pop, breathing, a, out_dir)
    print(json.dumps(summary, indent=2))


def summarise(df, pop, a, centres, base):
    p = df[df["perturbed"]]
    s = {
        "n_units": int(len(df)),
        "n_perturbed": int(len(p)),
        "direction_counts": p["direction"].value_counts().to_dict(),
        "n_returned": int(p["return_s"].notna().sum()),
        "n_not_returned_by_end": int(p["return_s"].isna().sum()),
        "n_stable_by_end": int(p["stable_from_s"].notna().sum()),
    }
    for col in ["return_s", "stable_from_s"]:
        v = p[col].dropna()
        if len(v):
            s[col + "_quantiles_file_s"] = {q: float(np.quantile(v, q)) for q in (0.25, 0.5, 0.75, 0.9)}
    # population: fraction of perturbed units in band; baseline level vs time to reach it
    f = pop["frac_in_band_perturbed"].to_numpy()
    base_level = float(np.nanmean(f[base]))
    s["frac_in_band_perturbed_baseline"] = base_level
    post = centres >= a.gasp_end_s
    for frac in (0.75, 0.9):
        target = frac * base_level
        idx = np.flatnonzero(post & (f >= target))
        s[f"first_time_frac_in_band_ge_{int(frac * 100)}pct_of_baseline_file_s"] = float(centres[idx[0]]) if len(idx) else None
    c = pop["popvec_corr_to_baseline"].to_numpy()
    s["popvec_corr_baseline_mean"] = float(np.nanmean(c[base]))
    s["popvec_corr_min"] = float(np.nanmin(c))
    s["popvec_corr_min_at_file_s"] = float(centres[int(np.nanargmin(c))])
    return s


def plot(df, rates, centres, pop, breathing, a, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    off = a.time_offset_s
    x = centres + off
    marks = [("switch", a.switch_s), ("gasp start", a.gasp_start_s), ("gasp end", a.gasp_end_s)]
    for m in filter(None, a.marks.split(",")):
        n, t = m.split(":")
        marks.append((n, float(t)))

    pert = df["perturbed"].to_numpy()
    ratio = np.log2((rates + 0.2) / (df["base_rate_hz"].to_numpy()[:, None] + 0.2))
    # sort: perturbed first, by return time (not returned last), then unperturbed by depth
    ret = df["return_s"].fillna(np.inf).to_numpy()
    order = np.lexsort((df["depth_um"].to_numpy(), ret, ~pert))

    fig, axes = plt.subplots(4, 1, figsize=(14, 15), sharex=True,
                             gridspec_kw={"height_ratios": [1, 6, 1.6, 1.6]})
    ax = axes[0]
    if breathing is not None:
        ax.plot(breathing[0] + off, breathing[1], color="0.2", lw=0.8)
    ax.set_ylabel("breaths/min")
    ax.set_title(f"{a.sorter}: firing rate relative to each unit's pre-switch baseline "
                 f"({a.win_s:g} s windows), {pert.sum()} of {len(df)} units perturbed")
    ax = axes[1]
    im = ax.imshow(ratio[order], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
                   extent=[x[0] - a.step_s / 2, x[-1] + a.step_s / 2, len(df), 0])
    n_p = int(pert.sum())
    ax.axhline(n_p, color="k", lw=0.8)
    rs = df["return_s"].to_numpy()[order]
    for row, t in enumerate(rs[:n_p]):
        if np.isfinite(t):
            ax.plot(t + off, row + 0.5, "k|", ms=3)
    ax.set_ylabel(f"units (top {n_p}: perturbed, sorted by return time | = return)")
    # colour bar outside the heat map's axes so all panels keep the same width and their time axes line up
    cb = fig.colorbar(im, cax=axes[1].inset_axes([1.005, 0.25, 0.012, 0.5]))
    cb.set_label("log2 rate / baseline")
    ax = axes[2]
    ax.plot(x, pop["frac_in_band_perturbed"], label="perturbed units in baseline band")
    ax.plot(x, pop["frac_in_band_all"], label="all units in baseline band", alpha=0.7)
    ret_sorted = np.sort(df.loc[df["perturbed"], "return_s"].dropna().to_numpy())
    if n_p:
        ax.step(np.r_[x[0], ret_sorted + off], np.r_[0, np.arange(1, len(ret_sorted) + 1) / n_p], where="post",
                color="k", lw=1, label="cumulative fraction returned (sustained)")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction")
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[3]
    ax.plot(x, pop["popvec_corr_to_baseline"], color="C2")
    ax.set_ylabel("population-vector r\nvs baseline")
    ax.set_xlabel("raw-run time (s)" if off else "file time (s)")
    for axx in axes:
        for n, t in marks:
            axx.axvline(t + off, color="0.5", ls="--", lw=0.8)
    for n, t in marks:
        axes[0].text(t + off, axes[0].get_ylim()[1], n, rotation=90, va="top", ha="right", fontsize=7, color="0.3")
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_overview.png", dpi=120)
    plt.close(fig)

    # example units: perturbed, sorted by return time, show 16 spread across the range
    p_idx = np.flatnonzero(pert)
    if len(p_idx):
        p_idx = p_idx[np.argsort(ret[p_idx])]
        pick = p_idx[np.unique(np.linspace(0, len(p_idx) - 1, min(16, len(p_idx))).round().astype(int))]
        fig, axs = plt.subplots(4, 4, figsize=(16, 11), sharex=True)
        for axx, i in zip(axs.ravel(), pick):
            row = df.iloc[i]
            axx.plot(x, rates[i], lw=0.8)
            axx.axhspan(row["band_lo_hz"], row["band_hi_hz"], color="0.85")
            axx.axhline(row["base_rate_hz"], color="0.5", lw=0.6)
            for n, t in marks:
                axx.axvline(t + off, color="0.6", ls="--", lw=0.6)
            if np.isfinite(row["return_s"]):
                axx.axvline(row["return_s"] + off, color="g", lw=1)
            rv = f"{row['return_s'] + off:.0f}" if np.isfinite(row["return_s"]) else "not by end"
            axx.set_title(f"#{row['unit_id']} {row['label']} {row['direction']}  return {rv}", fontsize=8)
        for axx in axs.ravel()[len(pick):]:
            axx.axis("off")
        fig.supxlabel("raw-run time (s)" if off else "file time (s)")
        fig.supylabel("rate (Hz), grey band = baseline band, green = return")
        fig.tight_layout()
        fig.savefig(out_dir / "recovery_examples.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
