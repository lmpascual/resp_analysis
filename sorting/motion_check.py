"""Does a run's motion estimate flatten the spike bands? Check it before sorting (run_sorting.py --stop-after preprocess).

Reads the peaks DREDGE used (<output>/preprocessing/motion: peaks.npy, peak_locations.npy) and the motion that was
applied (the bridged one, <output>/preprocessing/motion_bridge.npz, if the config bridges windows; otherwise the
estimate in motion/motion), and draws, per depth region, the peak depths before and after subtracting the
displacement at each peak's time and depth. Flat bands after correction = the estimate is right there. Also writes,
per region and time block, the residual band movement: the shift that best aligns each block's corrected depth
histogram with the reference block's (+-40 um search).

    python motion_check.py <output_dir> --events 1575 2057 --ref 700 1200 [--out <folder>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load(output_dir: Path):
    pre = output_dir / "preprocessing"
    mf = pre / "motion"
    peaks = np.load(mf / "peaks.npy")
    locs = np.load(mf / "peak_locations.npy")
    fs = 30000.0
    import json
    try:
        fs = float(json.loads((pre / "preprocessing_info.json").read_text())["sampling_frequency"])
    except Exception:
        pass
    t = peaks["sample_index"] / fs
    if (pre / "motion_bridge.npz").exists():
        z = np.load(pre / "motion_bridge.npz")
        tb, sb, disp = z["temporal_bins_s"], z["spatial_bins_um"], z["bridged"]
    else:
        from spikeinterface.core.motion import Motion
        mo = Motion.load(mf / "motion")
        tb, sb, disp = np.asarray(mo.temporal_bins_s[0]), np.asarray(mo.spatial_bins_um), np.asarray(mo.displacement[0])
        tb = tb - tb[0] + (tb[1] - tb[0]) / 2 if tb[0] > 1 else tb
    return t, locs["y"], np.abs(peaks["amplitude"]), tb, sb, disp


def displacement_at(t, y, tb, sb, disp):
    ti = np.clip(np.searchsorted(tb, t) - 1, 0, tb.size - 1)
    if sb.size == 1:
        return disp[ti, 0]
    yi = np.clip(np.searchsorted(sb, y) - 1, 0, sb.size - 2)     # linear in depth between window centres
    w = np.clip((y - sb[yi]) / (sb[yi + 1] - sb[yi]), 0, 1)
    return disp[ti, yi] * (1 - w) + disp[ti, yi + 1] * w


def residual_shift(y, sel_ref, sel, lo, hi):
    edges = np.arange(lo - 60, hi + 60, 2.0)
    k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    hr = np.convolve(np.histogram(y[sel_ref], edges)[0], k, "same").astype(float)
    h = np.convolve(np.histogram(y[sel], edges)[0], k, "same").astype(float)
    if h.sum() < 100 or hr.sum() < 100:
        return np.nan, np.nan
    hr, h = hr - hr.mean(), h - h.mean()
    lags = np.arange(-20, 21)
    cc = np.array([np.dot(np.roll(h, L), hr) for L in lags])
    i = int(np.argmax(cc))
    return -lags[i] * 2.0, cc[i] / (np.linalg.norm(h) * np.linalg.norm(hr) + 1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir")
    p.add_argument("--events", nargs="*", type=float, default=[], help="file times to mark (s)")
    p.add_argument("--ref", nargs=2, type=float, required=True, help="reference span for the residuals (file s)")
    p.add_argument("--region-um", type=float, default=600.0)
    p.add_argument("--block-s", type=float, default=60.0)
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out or Path(a.output_dir) / "motion_check")
    out.mkdir(parents=True, exist_ok=True)
    t, y, amp, tb, sb, disp = load(Path(a.output_dir))
    keep = amp > np.percentile(amp, 30)
    t, y, amp = t[keep], y[keep], amp[keep]
    yc = y - displacement_at(t, y, tb, sb, disp)
    top = np.percentile(y, 99.5)
    regions = [(r, r + a.region_um) for r in np.arange(0, top, a.region_um)]
    sub = slice(None, None, a.decimate)
    fig, axes = plt.subplots(len(regions), 2, figsize=(26, 3.6 * len(regions)), sharex=True)
    for row, (lo, hi) in zip(axes[::-1], regions):
        for ax, yy, title in ((row[0], y, "uncorrected"), (row[1], yc, "corrected")):
            m = (yy >= lo) & (yy < hi)
            ax.scatter(t[m][sub], yy[m][sub], s=0.2, c=np.log(amp[m][sub]), cmap="viridis", lw=0, alpha=0.5)
            ax.set_ylim(lo, hi)
            ax.set_title(title, fontsize=8, loc="left")
            for e in a.events:
                ax.axvline(e, color="r", lw=0.4, alpha=0.5)
    axes[-1, 0].set_xlabel("file time (s)")
    axes[-1, 1].set_xlabel("file time (s)")
    fig.tight_layout()
    fig.savefig(out / "raster_uncorrected_vs_corrected.png", dpi=60)
    plt.close(fig)

    # residual movement of the corrected bands, per region and block
    ref = (t >= a.ref[0]) & (t < a.ref[1])
    blocks = np.arange(t.min(), t.max(), a.block_s)
    rows = []
    for lo, hi in regions:
        inr = (yc >= lo) & (yc < hi)
        inr_u = (y >= lo) & (y < hi)
        for b0 in blocks:
            sel = (t >= b0) & (t < b0 + a.block_s)
            s_c, q_c = residual_shift(yc, ref & inr, sel & inr, lo, hi)
            s_u, q_u = residual_shift(y, ref & inr_u, sel & inr_u, lo, hi)
            rows.append({"region_um": f"{lo:.0f}-{hi:.0f}", "block_start_s": b0, "residual_shift_um": s_c,
                         "match": q_c, "uncorrected_shift_um": s_u, "uncorrected_match": q_u})
    R = pd.DataFrame(rows)
    R.to_csv(out / "residual_shift.tsv", sep="\t", index=False, float_format="%.2f")
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    for reg, g in R.groupby("region_um", sort=False):
        axes[0].plot(g["block_start_s"], g["uncorrected_shift_um"], lw=0.8, label=reg)
        axes[1].plot(g["block_start_s"], g["residual_shift_um"], lw=0.8, label=reg)
    axes[0].set_ylabel("band shift vs reference,\nuncorrected (um)")
    axes[1].set_ylabel("after correction (um)")
    axes[0].legend(fontsize=7, ncol=7)
    for ax in axes:
        ax.axvspan(*a.ref, color="0.9")
        for e in a.events:
            ax.axvline(e, color="r", lw=0.4, alpha=0.5)
    axes[1].set_xlabel("file time (s)")
    fig.tight_layout()
    fig.savefig(out / "residual_shift.png", dpi=100)
    print(R.groupby("region_um", sort=False)[["uncorrected_shift_um", "residual_shift_um"]]
          .agg(lambda s: np.nanpercentile(np.abs(s), 90)).rename(columns=lambda c: c + "_p90abs").to_string())


if __name__ == "__main__":
    main()
