"""Does anything in the recorded network track the diaphragm's tonic activity or the sighs? (e040 question 3)

Reads a population_dynamics.py run over a long sort (cache/ctx.pkl) and its sort config (NIDQ file, channels, sync
map). Tonic activity = diaphragm motor-unit spikes in expiration (analysis/session/diaphragm_events.py); sighs =
cibrrig's rule (breath AUC > 7 rolling MADs over 20 s), gasps excluded. Analysis span: --span (run s).

1. Slow covariation (30 s bins): Spearman correlation of each unit's rate (whole cycle, and expiration only) with
   the tonic EMG rate and with the sigh count. Both drift slowly, so p values come from circular shifts of the unit's
   series against the EMG series (shifts >= 10 bins; --n-shift), Benjamini-Hochberg over units.
2. CNO step (--cno run s): tonic activity stops for minutes after CNO and comes back. Units whose expiratory rate
   falls and returns with it: rate before (cno - 120 to cno), during the gap (--gap run s span), after (--after).
3. Breath-by-breath coupling: the same correlations on the residuals after a 2-minute running median, which removes
   the shared slow trends.
4. Spike-level: cross-correlogram of each unit's spikes with the tonic EMG spikes, in expiration only, +-50 ms in
   1 ms bins, against 100 surrogates with the unit's spikes jittered +-20 ms; peak z and lag (negative = unit first).
5. Sighs: each unit's rate around sigh onsets (-3 to +5 s, 100 ms bins) against normal breaths from the same minutes;
   pre-sigh (-1.5 to 0 s) and sigh-cycle changes (z over normal breaths).

    python tonic_sigh_network.py --run <out>/grp/e040/population_dynamics_long --span 5770 9985 --cno 7260 ^
        --gap 7290 7600 --after 7700 8100 --out <out>/grp/e040/tonic_sighs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "session"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sorting"))
import population_dynamics as pdm  # noqa: E402,F401  (cibrrig shims)
from cibrrig.preprocess import physiology  # noqa: E402
from diaphragm_events import emg_ecg_events, expiratory_counts  # noqa: E402
from npxsort.paths import expand  # noqa: E402
from recovery_compare import load_context  # noqa: E402

BIN_S = 30.0


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    r = rankdata(p[ok])
    qq = p[ok] * ok.sum() / r
    order = np.argsort(p[ok])[::-1]
    qq[order] = np.minimum.accumulate(qq[order])
    q[ok] = np.minimum(qq, 1)
    return q


def shift_p(x, y, n_shift, rng, min_shift=10):
    """Spearman rho of x and y, p from circular shifts of x (two-sided)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 * min_shift or np.std(x) == 0:
        return np.nan, np.nan
    rho = spearmanr(x, y)[0]
    sh = rng.integers(min_shift, x.size - min_shift, n_shift)
    null = np.array([spearmanr(np.roll(x, s), y)[0] for s in sh])
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_shift + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--config", required=True, help="the sort's config_used.yaml (NIDQ file, channels, sync map)")
    p.add_argument("--span", nargs=2, type=float, required=True, help="run s")
    p.add_argument("--cno", type=float, default=None)
    p.add_argument("--gap", nargs=2, type=float, default=None)
    p.add_argument("--after", nargs=2, type=float, default=None)
    p.add_argument("--min-rate", type=float, default=0.2)
    p.add_argument("--n-shift", type=int, default=500)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    c = load_context(Path(a.run))
    off, d, spikes, ids_all = c["time_offset_s"], c["dia_df"].copy(), c["spikes"], c["ids"]
    cfg = yaml.safe_load(Path(a.config).read_text())
    nq = expand(cfg["nidq"])
    dur = float(d["on_sec"].max() + 5)
    ev_file = out / "emg_ecg_events.npz"
    if ev_file.exists():
        ev = dict(np.load(ev_file))
    else:
        ev = emg_ecg_events(nq["bin"], nq["n_channels"], nq["emg_channel"], nq["ecg_channel"], nq.get("sync_map"),
                            off, dur)
        np.savez_compressed(ev_file, **ev)
    emg = ev["emg_spikes_s"]
    ons, offs = d["on_sec"].to_numpy(), d["off_sec"].to_numpy()
    gasp = d["gasp"].to_numpy().astype(bool)
    d["sigh"] = physiology.compute_sighs(ons, d["auc"].to_numpy()) & ~gasp
    n_e, dur_e = expiratory_counts(emg, ons, offs)
    d["tonic_hz"] = n_e / dur_e
    t0, t1 = a.span[0] - off, a.span[1] - off
    insp = (ons >= t0) & (ons < t1) & ~gasp
    insp[-1] = False

    # per breath: unit spikes per cycle and in expiration
    sel = np.flatnonzero(insp)
    cyc_dur = ons[sel + 1] - ons[sel]
    rate_cyc, rate_exp = {}, {}
    for u in ids_all:
        s = spikes[u]
        n_c = np.searchsorted(s, ons[sel + 1]) - np.searchsorted(s, ons[sel])
        ne_u, de_u = expiratory_counts(s, ons, offs)
        rate_cyc[u] = n_c / cyc_dur
        rate_exp[u] = (ne_u / de_u)[sel]
    mean_rate = {u: rate_cyc[u].mean() for u in ids_all}
    ids = [u for u in ids_all if mean_rate[u] >= a.min_rate]
    B = pd.DataFrame({"on_s": ons[sel], "raw_s": ons[sel] + off, "tonic_hz": d["tonic_hz"].to_numpy()[sel],
                      "sigh": d["sigh"].to_numpy()[sel], "ibi_s": d["ibi_s"].to_numpy()[sel],
                      "ti_s": d["duration_sec"].to_numpy()[sel]})
    B.to_csv(out / "breaths_tonic_sigh.tsv", sep="\t", index=False, float_format="%.4f")

    # 1. slow covariation in 30 s bins
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    bi = np.digitize(ons[sel], edges) - 1
    nb = edges.size - 1
    emg_bin = np.array([np.nanmean(B["tonic_hz"][bi == k]) if (bi == k).any() else np.nan for k in range(nb)])
    sigh_bin = np.array([B["sigh"][bi == k].sum() for k in range(nb)], float) * (60 / BIN_S)
    rate_bin = {u: np.array([np.mean(rate_cyc[u][bi == k]) if (bi == k).any() else np.nan for k in range(nb)]) for u in ids}
    rexp_bin = {u: np.array([np.nanmean(rate_exp[u][bi == k]) if (bi == k).any() else np.nan for k in range(nb)]) for u in ids}
    pop_bin = np.nansum([rate_bin[u] for u in ids], axis=0)
    rows = []
    for u in ids:
        r = {"unit_id": u, "mean_rate_hz": mean_rate[u]}
        for nm, ser in (("tonic", emg_bin), ("sighs", sigh_bin)):
            r[f"rho_rate_{nm}"], r[f"p_rate_{nm}"] = shift_p(rate_bin[u], ser, a.n_shift, rng)
            r[f"rho_exp_{nm}"], r[f"p_exp_{nm}"] = shift_p(rexp_bin[u], ser, a.n_shift, rng)
        # 3. breath-by-breath residuals (2-min running median removed)
        win = max(int(120 / np.median(cyc_dur)), 5)
        res_u = pd.Series(rate_exp[u]) - pd.Series(rate_exp[u]).rolling(win, center=True, min_periods=5).median()
        res_e = B["tonic_hz"] - B["tonic_hz"].rolling(win, center=True, min_periods=5).median()
        ok = np.isfinite(res_u) & np.isfinite(res_e)
        r["rho_breath_residual_tonic"] = spearmanr(res_u[ok], res_e[ok])[0] if ok.sum() > 20 else np.nan
        rows.append(r)
    U = pd.DataFrame(rows)
    for col in [c_ for c_ in U.columns if c_.startswith("p_")]:
        U["q" + col[1:]] = bh(U[col])

    # 2. CNO step
    if a.cno is not None and a.gap and a.after:
        def mean_exp(u, lo, hi):
            m = (B["raw_s"] >= lo) & (B["raw_s"] < hi)
            return np.nanmean(rate_exp[u][m.to_numpy()]) if m.any() else np.nan

        def mean_emg(lo, hi):
            m = (B["raw_s"] >= lo) & (B["raw_s"] < hi)
            return np.nanmean(B["tonic_hz"][m])
        U["exp_rate_pre_cno"] = [mean_exp(u, a.cno - 120, a.cno) for u in U["unit_id"]]
        U["exp_rate_gap"] = [mean_exp(u, *a.gap) for u in U["unit_id"]]
        U["exp_rate_after"] = [mean_exp(u, *a.after) for u in U["unit_id"]]
        cno_emg = (mean_emg(a.cno - 120, a.cno), mean_emg(*a.gap), mean_emg(*a.after))
        U["follows_cno_step"] = (U["exp_rate_gap"] < 0.5 * U["exp_rate_pre_cno"]) & (U["exp_rate_after"] > 2 * U["exp_rate_gap"])
    else:
        cno_emg = None

    # 4. spike-level CCG with tonic EMG spikes, expiration only
    e0 = offs[sel] + 0.04
    e1 = ons[sel + 1] - 0.03
    in_exp = lambda t: (lambda k: (k >= 0) & (t < e1[np.clip(k, 0, e1.size - 1)]) & (t >= e0[np.clip(k, 0, e0.size - 1)]))(np.searchsorted(e0, t, side="right") - 1)  # noqa: E731
    emg_exp = emg[in_exp(emg)]
    lags = np.arange(-0.05, 0.0505, 0.001)
    ccg_rows, ccgs = [], {}

    def ccg(st):
        k0 = np.searchsorted(st, emg_exp + lags[0])
        k1 = np.searchsorted(st, emg_exp + lags[-1])
        diffs = np.concatenate([st[i:j] - t for i, j, t in zip(k0, k1, emg_exp) if j > i]) if emg_exp.size else np.array([])
        return np.histogram(diffs, lags)[0]
    for u in ids:
        st = spikes[u][in_exp(spikes[u])]
        if st.size < 200 or emg_exp.size < 200:
            continue
        h = ccg(st).astype(float)
        sur = np.array([ccg(np.sort(st + rng.uniform(-0.02, 0.02, st.size))) for _ in range(100)])
        z = (h - sur.mean(0)) / np.maximum(sur.std(0), 1e-9)
        k = int(np.argmax(z))
        # lag axis: unit spike time minus EMG spike time; negative = unit spike first
        ccg_rows.append({"unit_id": u, "n_unit_spikes_exp": st.size, "peak_z": z[k],
                         "peak_lag_ms": (lags[k] + 0.0005) * 1e3, "trough_z": z.min()})
        ccgs[u] = z
    C = pd.DataFrame(ccg_rows)
    if len(C):
        U = U.merge(C, on="unit_id", how="left")

    # 5. sighs
    sg = np.flatnonzero(d["sigh"].to_numpy() & (ons >= t0) & (ons < t1))
    tb = np.arange(-3, 5.0001, 0.1)
    sig_rows = []
    peth_s, peth_n = {}, {}
    normal = np.flatnonzero(insp & ~d["sigh"].to_numpy())
    for u in ids:
        s = spikes[u]
        hs = np.mean([np.histogram(s - ons[i], tb)[0] for i in sg], axis=0) / 0.1 if sg.size else np.full(tb.size - 1, np.nan)
        # matched normal breaths: up to 5 normal breaths within 60 s of each sigh
        ctrl = []
        for i in sg:
            near = normal[(np.abs(ons[normal] - ons[i]) < 60) & (np.abs(ons[normal] - ons[i]) > 3)]
            ctrl.extend(rng.choice(near, min(5, near.size), replace=False) if near.size else [])
        hn = np.array([np.histogram(s - ons[i], tb)[0] for i in ctrl]) / 0.1 if ctrl else np.full((1, tb.size - 1), np.nan)
        peth_s[u], peth_n[u] = hs, hn.mean(0)
        sd = hn.std(0) / np.sqrt(max(sg.size, 1)) + 1e-9
        pre = (tb[:-1] >= -1.5) & (tb[:-1] < 0)
        dur_ = (tb[:-1] >= 0) & (tb[:-1] < 0.5)
        sig_rows.append({"unit_id": u, "n_sighs": int(sg.size),
                         "pre_sigh_z": float(np.mean((hs[pre] - hn.mean(0)[pre]) / sd[pre])),
                         "sigh_z": float(np.mean((hs[dur_] - hn.mean(0)[dur_]) / sd[dur_])),
                         "pre_sigh_rate_ratio": float((hs[pre].mean() + 0.1) / (hn.mean(0)[pre].mean() + 0.1)),
                         "sigh_rate_ratio": float((hs[dur_].mean() + 0.1) / (hn.mean(0)[dur_].mean() + 0.1))})
    U = U.merge(pd.DataFrame(sig_rows), on="unit_id", how="left")
    U.to_csv(out / "unit_tonic_sigh.tsv", sep="\t", index=False, float_format="%.4f")
    pd.DataFrame({"bin_start_raw_s": edges[:-1] + off, "tonic_hz": emg_bin, "sighs_per_min": sigh_bin,
                  "summed_unit_rate_hz": pop_bin}).to_csv(out / "bins_30s.tsv", sep="\t", index=False, float_format="%.4f")

    # figures
    xt = edges[:-1] + off + BIN_S / 2
    top = U.reindex(U["rho_exp_tonic"].abs().sort_values(ascending=False).index).head(6)
    fig, ax = plt.subplots(4 + len(top), 1, figsize=(14, 2.0 * (4 + len(top))), sharex=True)
    ax[0].plot(xt, emg_bin, color="#c27a00"); ax[0].set_ylabel("tonic dia\nspikes/s (exp.)")
    ax[1].step(xt, sigh_bin, where="mid", color="m"); ax[1].set_ylabel("sighs/min")
    ax[2].plot(B["raw_s"], 60 / B["ibi_s"], ",", color="0.3"); ax[2].set_ylabel("breaths/min")
    ax[3].plot(xt, pop_bin, color="k"); ax[3].set_ylabel("summed unit\nrate (Hz)")
    for x, r in zip(ax[4:], top.itertuples()):
        x.plot(xt, rexp_bin[r.unit_id], color="#2a78d6")
        x.set_ylabel(f"#{r.unit_id} exp.\nrate (Hz)")
        x.text(0.01, 0.85, f"rho tonic {r.rho_exp_tonic:+.2f} (q {r.q_exp_tonic:.3f}); sighs {r.rho_exp_sighs:+.2f}",
               transform=x.transAxes, fontsize=7)
    for x in ax:
        if a.cno is not None:
            x.axvline(a.cno, color="g", lw=0.8)
        x.spines[["top", "right"]].set_visible(False)
    ax[-1].set_xlabel("run time (s)")
    fig.suptitle(f"{c.get('title', '')}: tonic diaphragm activity, sighs and the units that track tonic activity most "
                 f"(30 s bins; green: CNO)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "tonic_sigh_timecourse.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_, col, lab in ((axes[0], "rho_exp_tonic", "expiratory rate vs tonic EMG"),
                          (axes[1], "rho_rate_sighs", "cycle rate vs sigh rate")):
        o = U.sort_values(col)
        colr = np.where(o["q_" + col[4:]] < 0.05, "#d93636", "0.6")
        ax_.bar(np.arange(len(o)), o[col], color=colr)
        ax_.set_title(f"Spearman rho, {lab}\n(red: q < 0.05, circular-shift test)", fontsize=8)
        ax_.set_xlabel("units (sorted)")
    if len(C):
        for u, z in ccgs.items():
            axes[2].plot((lags[:-1] + 0.0005) * 1e3, z, lw=0.5, color="0.7")
        best = C.sort_values("peak_z", ascending=False).head(3)
        for r in best.itertuples():
            axes[2].plot((lags[:-1] + 0.0005) * 1e3, ccgs[r.unit_id], lw=1.2, label=f"#{r.unit_id} z {r.peak_z:.1f}")
        axes[2].axvline(0, color="k", lw=0.5)
        axes[2].legend(fontsize=7)
        axes[2].set_xlabel("unit spike - EMG spike (ms); negative: unit first")
        axes[2].set_ylabel("z vs +-20 ms jitter")
        axes[2].set_title("cross-correlogram with tonic EMG spikes (expiration)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tonic_sigh_units.png", dpi=130)
    plt.close(fig)

    if sg.size:
        o = U.sort_values("sigh_z", ascending=False)["unit_id"].tolist()
        M = np.array([np.sqrt(peth_s[u]) - np.sqrt(peth_n[u]) for u in o])
        fig, ax_ = plt.subplots(figsize=(7, 8))
        v = np.nanpercentile(np.abs(M), 98)
        ax_.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=(tb[0], tb[-1], len(o), 0))
        ax_.axvline(0, color="k", lw=0.6)
        ax_.set_xlabel("s from sigh onset")
        ax_.set_ylabel("unit (sorted by sigh-cycle change)")
        ax_.set_title(f"sigh - matched normal breath, sqrt Hz ({sg.size} sighs)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "sigh_peth_difference.png", dpi=130)
        plt.close(fig)
    summ = {"n_units": len(ids), "n_breaths": int(sel.size), "n_sighs": int(sg.size),
            "emg_noise_v": float(ev["emg_noise_v"]), "cno_tonic_hz_pre_gap_after": cno_emg,
            "units_q05_exp_tonic": int((U["q_exp_tonic"] < 0.05).sum()),
            "units_q05_rate_sighs": int((U["q_rate_sighs"] < 0.05).sum()),
            "units_following_cno_step": int(U.get("follows_cno_step", pd.Series(dtype=bool)).sum()),
            "units_ccg_peak_z_gt_5": int((U.get("peak_z", pd.Series(dtype=float)) > 5).sum())}
    (out / "summary.json").write_text(pd.Series(summ).to_json(indent=1))
    print(summ)


if __name__ == "__main__":
    main()
