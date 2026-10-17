"""Check the environment before sorting real data. Every check prints PASS or FAIL.

    python check_env.py            # expects a CUDA GPU (Kilosort 4)
    python check_env.py --no-gpu   # skip the GPU requirement

The old pipeline hid crashes in the math libraries by replacing NumPy functions with PyTorch
ones. This script exercises the same operations so a broken install fails here, loudly, instead.
"""
import argparse
import sys
import tempfile
import traceback
from pathlib import Path

RESULTS = []


def check(name):
    def wrap(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k)
                RESULTS.append((name, True, detail or ""))
            except Exception as e:  # report every failure, keep checking the rest
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        return run
    return wrap


@check("package versions")
def versions():
    import numpy, scipy, sklearn, numba, spikeinterface, kilosort, torch, yaml, hdbscan  # noqa: F401
    return (f"spikeinterface {spikeinterface.__version__}, kilosort {kilosort.__version__}, "
            f"torch {torch.__version__}, numpy {numpy.__version__}, numba {numba.__version__}")


@check("torch sees a CUDA GPU")
def gpu(required: bool):
    import torch
    if not torch.cuda.is_available():
        if required:
            raise RuntimeError("torch.cuda.is_available() is False (CPU-only torch build or driver problem)")
        return "no GPU (allowed with --no-gpu)"
    x = torch.randn(4096, 4096, device="cuda")
    (x @ x).sum().item()
    return torch.cuda.get_device_name(0)


@check("numpy / scipy / sklearn linear algebra")
def linalg():
    import numpy as np
    from scipy import linalg as sla
    from sklearn.decomposition import TruncatedSVD, PCA
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3000, 384)).astype("float32")
    np.linalg.svd(a, full_matrices=False)
    cov = a.T @ a / len(a)
    np.linalg.eigh(cov)
    np.linalg.solve(cov + np.eye(384), rng.standard_normal(384))
    sla.cho_factor(cov + np.eye(384))
    TruncatedSVD(5).fit(a)
    PCA(5).fit(a)
    return "SVD, eigh, solve, Cholesky, TruncatedSVD, PCA ok"


@check("SpikeInterface parallel jobs (process pool, spawn)")
def parallel_process():
    import spikeinterface.core as si
    rec = si.generate_recording(num_channels=64, durations=[20.0], seed=0)
    with tempfile.TemporaryDirectory() as d:
        rec.save(folder=Path(d) / "r", n_jobs=4, pool_engine="process", mp_context="spawn", chunk_duration="1s", progress_bar=False)
        si.get_noise_levels(rec, n_jobs=4, pool_engine="process", mp_context="spawn")
    return "ok"


@check("SpikeInterface parallel jobs (thread pool)")
def parallel_thread():
    import spikeinterface.core as si
    rec = si.generate_recording(num_channels=64, durations=[20.0], seed=0)
    with tempfile.TemporaryDirectory() as d:
        rec.save(folder=Path(d) / "r", n_jobs=4, pool_engine="thread", chunk_duration="1s", progress_bar=False)
    return "ok"


@check("whitening + peak detection + clustering features")
def components():
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    rec, _ = si.generate_ground_truth_recording(durations=[20.0], num_channels=64, num_units=10, seed=0)
    w = spre.whiten(spre.bandpass_filter(rec, dtype="float32"), mode="local", radius_um=100.0, dtype="float32")
    peaks = detect_peaks(w, method="locally_exclusive", method_kwargs={"detect_threshold": 5}, job_kwargs={"n_jobs": 2, "progress_bar": False})
    return f"{len(peaks)} peaks"


@check("sorters installed")
def sorters():
    import spikeinterface.sorters as ss
    installed = ss.installed_sorters()
    missing = [s for s in ("kilosort4", "lupin") if s not in installed]
    if missing:
        raise RuntimeError(f"not installed: {missing}")
    return "kilosort4, lupin"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()
    versions()
    gpu(required=not args.no_gpu)
    linalg()
    parallel_process()
    parallel_thread()
    components()
    sorters()
    print("\n" + "=" * 70)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:50s} {detail}")
    failed = [r for r in RESULTS if not r[1]]
    print("=" * 70)
    if failed:
        print(f"{len(failed)} check(s) failed. Fix the environment before sorting (see README troubleshooting).")
        return 1
    print("Environment OK.")
    return 0


if __name__ == "__main__":  # required on Windows for spawn-based multiprocessing
    sys.exit(main())
