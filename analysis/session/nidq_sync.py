"""Map NIDQ samples to imec seconds from the shared 1 Hz sync square wave, for long recordings.

The population analyses index the NIDQ file at the nominal 10 kHz. Over e097's 420 s window that is ~1 ms off the
imec clock, but over hours it is not: on e038 / e040 (2.4 / 2.8 h, PXIe-6363) the NIDQ runs at ~9999.912 Hz, so
nominal NIDQ time falls 71 / 90 ms behind imec time by the end of the run, a third of a breath. This fits
imec_s = a * nidq_sample + b on the rising sync edges (NIDQ: bit 0 of the digital word; imec: bit 6 of the last
channel of the LF file, read instead of the AP file because it is 12x smaller) and writes it as JSON. A sort config
that names the JSON in `nidq.sync_map` makes population_dynamics.py read the NIDQ on the imec clock.

    python nidq_sync.py <run>/<run>_t0.nidq.bin <run>/<run>_imec0/<run>_t0.imec0.lf.bin --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _meta(binfile: Path) -> dict:
    return dict(l.split("=", 1) for l in binfile.with_suffix(".meta").read_text().splitlines() if "=" in l)


def rising_edges(binfile: Path, bit: int) -> np.ndarray:
    """Sample indices of the rising edges of one bit of a SpikeGLX file's last saved channel."""
    n = int(_meta(binfile)["nSavedChans"])
    word = np.memmap(binfile, dtype="int16", mode="r").reshape(-1, n)[:, n - 1]
    b = ((np.asarray(word) >> bit) & 1).astype(np.int8)
    return np.flatnonzero(np.diff(b) > 0) + 1


def fit_sync_map(nidq_bin: Path, imec_bin: Path, nidq_bit: int = 0, imec_bit: int = 6) -> dict:
    ne = rising_edges(nidq_bin, nidq_bit)
    le = rising_edges(imec_bin, imec_bit)
    fs_imec = float(_meta(imec_bin)["imSampRate"])
    nt, lt = ne / 1e4, le / fs_imec
    off = np.median(lt[:20] - nt[:20])
    j = np.clip(np.searchsorted(lt, nt + off), 1, lt.size - 1)
    jj = np.where(np.abs(lt[j - 1] - nt - off) < np.abs(lt[j] - nt - off), j - 1, j)
    ok = np.abs(lt[jj] - nt - off) < 0.1
    A = np.vstack([ne[ok], np.ones(ok.sum())]).T
    (a, b), *_ = np.linalg.lstsq(A, lt[jj[ok]], rcond=None)
    res = lt[jj[ok]] - A @ (a, b)
    last = ne[ok][-1]
    return {"nidq_bin": str(nidq_bin), "imec_bin": str(imec_bin),
            "imec_s_per_nidq_sample": float(a), "imec_s_at_nidq_sample0": float(b),
            "implied_nidq_rate_hz": float(1 / a), "meta_nidq_rate_hz": float(_meta(nidq_bin)["niSampRate"]),
            "n_edges_paired": int(ok.sum()), "n_edges_nidq": int(ne.size), "n_edges_imec": int(le.size),
            "residual_ms_p99": float(np.percentile(np.abs(res), 99) * 1e3),
            "nominal_10k_lag_ms_at_last_edge": float((last * a + b - last / 1e4) * 1e3)}


def nidq_rows(sync_map: dict, start_s: float, n: int, fs: float = 1e4) -> np.ndarray:
    """NIDQ sample index (nearest) at imec times start_s + i / fs, i = 0..n-1."""
    t = start_s + np.arange(n) / fs
    return np.rint((t - sync_map["imec_s_at_nidq_sample0"]) / sync_map["imec_s_per_nidq_sample"]).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nidq_bin")
    p.add_argument("imec_bin", help="the run's imec .lf.bin (or .ap.bin; any stream that saves the sync channel)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = fit_sync_map(Path(a.nidq_bin), Path(a.imec_bin))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
