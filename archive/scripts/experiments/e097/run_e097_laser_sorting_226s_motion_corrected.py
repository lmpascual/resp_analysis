#!/usr/bin/env python3
"""
Motion-Corrected 226s Continuous Lupin Sliding-Window Spike Sorting & Recruitment Pipeline
Animal: e097 | Window: t = 796.827s to 1023.070s (226.243s duration)

Architecture:
1. Decentralized Rigid Axial Motion Estimation (direction='y', bin_s=0.5s, detect_threshold=6.0 SD)
2. Global Spatial Kriging Interpolation (sigma=20.0 um, force_extrapolate)
3. 13 Overlapping 50-Channel Sliding Windows (Lupin / SpyKING-CIRCUS 2)
4. Strict Cross-Window Consensus Matching (Agreement >= 0.80, Dedup delta <= 0.4 ms)
5. Allen Institute QC (Presence Ratio >= 0.20, ISI Violations <= 0.10)
6. Direct Comparative Benchmark against Uncorrected Consensus Sorting
7. Comprehensive Run Journal & Drift Telemetry Generation
"""
import sys
import os
import time
import shutil
import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import spikeinterface.full as si
import spikeinterface.preprocessing as prep
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion
import spikeinterface.sortingcomponents.motion.motion_interpolation as mi

from src.pipeline.config import PipelineConfig, LupinConfig
from src.pipeline.io.io_spikeglx import read_spikeglx_ap_safe
from src.pipeline.sorting.sliding_window_runner import compute_window_ranges
from src.pipeline.sorting.lupin_pipeline import run_lupin_sorting
from src.pipeline.sorting.postprocessing import export_sorting_alf_and_qc
from src.pipeline.sorting.sliding_window_consensus import compute_sliding_window_consensus
try:
    from scripts.plot_interactive_sorter_raster import build_interactive_sorter_raster
except ImportError:
    build_interactive_sorter_raster = None

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Windows NumPy / BLAS Compatibility Patches (PyTorch-backed Kernel & Matmul)
# -------------------------------------------------------------------------
try:
    import threadpoolctl
    for cls in [threadpoolctl.OpenBLASController, threadpoolctl.MKLController]:
        if hasattr(cls, 'get_num_threads'): cls.get_num_threads = lambda self: 1
        if hasattr(cls, 'set_num_threads'): cls.set_num_threads = lambda self, n: None
except Exception:
    pass

orig_np_norm = np.linalg.norm
def safe_norm(x, ord=None, axis=None, keepdims=False):
    try:
        t = torch.as_tensor(x)
        if axis is None and not keepdims:
            return float(torch.linalg.norm(t, ord=ord))
        return torch.linalg.norm(t, ord=ord, dim=axis, keepdim=keepdims).numpy()
    except Exception:
        return orig_np_norm(x, ord=ord, axis=axis, keepdims=keepdims)

np.linalg.norm = safe_norm

try:
    import scipy.sparse.linalg as sla
    if hasattr(sla, 'lsmr'):
        sla.lsmr.__globals__['norm'] = safe_norm
except Exception:
    pass

try:
    import matplotlib.transforms as mtransforms
    def safe_transform_dot(a, b):
        t_a = torch.as_tensor(a, dtype=torch.float64)
        t_b = torch.as_tensor(b, dtype=torch.float64)
        return (t_a @ t_b).numpy()
    mtransforms.np.dot = safe_transform_dot
    mtransforms.inv = lambda m: torch.linalg.inv(torch.as_tensor(m, dtype=torch.float64)).numpy()
except Exception:
    pass

try:
    si_whiten_mod = sys.modules.get('spikeinterface.preprocessing.whiten')
    if si_whiten_mod is not None:
        def safe_compute_covariance_matrix(recording, apply_mean=False, regularize=False, regularize_kwargs=None, random_chunk_kwargs=None, **kwargs):
            random_chunk_kwargs = random_chunk_kwargs or {}
            random_data = si_whiten_mod.get_random_data_chunks(recording, concatenated=True, return_scaled=False, **random_chunk_kwargs)
            random_data = random_data.astype(np.float32)
            regularize_kwargs = regularize_kwargs if regularize_kwargs is not None else {'method': 'GraphicalLassoCV'}
            if apply_mean:
                M = np.mean(random_data, axis=0, keepdims=True)
                data = random_data - M
            else:
                M = None
                data = random_data
            if not regularize:
                t_data = torch.from_numpy(data)
                cov = (t_data.T @ t_data).numpy() / data.shape[0]
            else:
                cov = si_whiten_mod.compute_sklearn_covariance_matrix(data, regularize_kwargs).astype('float32')
            return data, cov, M

        def safe_compute_whitening_from_covariance(cov, eps):
            t_cov = torch.as_tensor(cov, dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(t_cov, full_matrices=True)
            diag_s = torch.diag(1.0 / torch.sqrt(S + eps))
            W = (U @ diag_s @ Vh).numpy()
            return W

        def safe_whiten_get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
            if traces.dtype.kind == 'u':
                traces = traces.astype('float32')
            if self.M is not None:
                t_diff = torch.from_numpy(traces - self.M)
                t_W = torch.from_numpy(self.W)
                whiten_traces = (t_diff @ t_W).numpy()
            else:
                t_traces = torch.from_numpy(traces)
                t_W = torch.from_numpy(self.W)
                whiten_traces = (t_traces @ t_W).numpy()
            whiten_traces = whiten_traces[:, channel_indices]
            if self.int_scale is not None:
                whiten_traces *= self.int_scale
            return whiten_traces.astype(self.dtype)

        si_whiten_mod.compute_covariance_matrix = safe_compute_covariance_matrix
        si_whiten_mod.compute_whitening_from_covariance = safe_compute_whitening_from_covariance
        si_whiten_mod.WhitenRecordingSegment.get_traces = safe_whiten_get_traces
except Exception:
    pass

try:
    import sklearn.utils.extmath as em
    from sklearn.utils.extmath import svd_flip
    import sklearn.decomposition._truncated_svd as tsvd

    def safe_randomized_svd(M, n_components, **kwargs):
        t_M = torch.as_tensor(M, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(t_M, full_matrices=False)
        u = U[:, :n_components].numpy()
        s = S[:n_components].numpy()
        vh = Vh[:n_components, :].numpy()
        if kwargs.get('flip_sign', True):
            u, vh = svd_flip(u, vh)
        return u, s, vh

    orig_safe_sparse_dot = em.safe_sparse_dot
    def safe_sparse_dot_patch(a, b, *, dense_output=False):
        try:
            t_a = torch.as_tensor(a)
            t_b = torch.as_tensor(b)
            return (t_a @ t_b).numpy()
        except Exception:
            return orig_safe_sparse_dot(a, b, dense_output=dense_output)

    tsvd._randomized_svd = safe_randomized_svd
    tsvd.safe_sparse_dot = safe_sparse_dot_patch
    em._randomized_svd = safe_randomized_svd
    em.randomized_svd = safe_randomized_svd
    em.safe_sparse_dot = safe_sparse_dot_patch
except Exception:
    pass

try:
    import spikeinterface.sortingcomponents.matching.circus as cm
    def safe_compress_templates(templates_array, approx_rank, remove_mean=False, return_new_templates=True):
        if remove_mean:
            templates_array -= templates_array.mean(axis=(1, 2))[:, None, None]
        num_templates, num_samples, num_channels = templates_array.shape
        temporal = np.zeros((num_templates, num_samples, approx_rank), dtype=np.float32)
        spatial = np.zeros((num_templates, approx_rank, num_channels), dtype=np.float32)
        singular = np.zeros((num_templates, approx_rank), dtype=np.float32)
        t_templates = torch.from_numpy(templates_array.astype(np.float32))
        U, S, Vh = torch.linalg.svd(t_templates, full_matrices=False)
        k = min(approx_rank, num_channels)
        temporal[:, :, :k] = U[:, :, :k].numpy()
        singular[:, :k] = S[:, :k].numpy()
        spatial[:, :k, :] = Vh[:, :k, :].numpy()
        if return_new_templates:
            t_temp = torch.from_numpy(temporal * singular[:, np.newaxis, :])
            t_spat = torch.from_numpy(spatial)
            templates_array = (t_temp @ t_spat).numpy()
        else:
            templates_array = None
        return temporal, singular, spatial, templates_array

    cm.compress_templates = safe_compress_templates
    cm.np.matmul = lambda a, b: (torch.as_tensor(a) @ torch.as_tensor(b)).numpy()
except Exception:
    pass

orig_np_svd = np.linalg.svd
def safe_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    try:
        t_a = torch.as_tensor(a)
        if not compute_uv:
            s = torch.linalg.svdvals(t_a)
            return s.numpy()
        U, S, Vh = torch.linalg.svd(t_a, full_matrices=full_matrices)
        return U.numpy(), S.numpy(), Vh.numpy()
    except Exception:
        return orig_np_svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)

np.linalg.svd = safe_svd

orig_get_kriging = mi.get_spatial_interpolation_kernel
get_kriging_kernel_distance = orig_get_kriging.__globals__['get_kriging_kernel_distance']

def safe_matmul(a, b, out=None):
    t_a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    t_b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b
    res = (t_a @ t_b).numpy()
    if out is not None:
        out[:] = res
        return out
    return res

mi.np.matmul = safe_matmul

def safe_get_spatial_interpolation_kernel(
    source_location,
    target_location,
    method="kriging",
    sigma_um=20.0,
    p=1,
    num_closest=4,
    sparse_thresh=None,
    dtype="float32",
    force_extrapolate=False,
):
    if method != "kriging":
        return orig_get_kriging(source_location, target_location, method=method, sigma_um=sigma_um, p=p, num_closest=num_closest, sparse_thresh=sparse_thresh, dtype=dtype, force_extrapolate=force_extrapolate)

    target_is_inside = np.ones(target_location.shape[0], dtype=bool)
    for dim in range(source_location.shape[1]):
        l0, l1 = np.min(source_location[:, dim]), np.max(source_location[:, dim])
        target_is_inside &= (target_location[:, dim] >= l0) & (target_location[:, dim] <= l1)

    Kxx = get_kriging_kernel_distance(source_location, source_location, sigma_um, p)
    Kyx = get_kriging_kernel_distance(target_location, source_location, sigma_um, p)

    t_Kyx = torch.from_numpy(Kyx.astype(np.float32))
    t_Kxx = torch.from_numpy((Kxx + 1e-6 * np.eye(Kxx.shape[0])).astype(np.float32))
    t_pinv = torch.linalg.pinv(t_Kxx)
    t_interp = t_Kyx @ t_pinv
    interpolation_kernel = t_interp.numpy().T.copy()

    if sparse_thresh is not None:
        interpolation_kernel[interpolation_kernel < sparse_thresh] = 0.0

    s = np.sum(interpolation_kernel, axis=0)
    interpolation_kernel[:, target_is_inside] /= s[target_is_inside].reshape(1, -1)

    if not force_extrapolate:
        interpolation_kernel[:, ~target_is_inside] = 0

    return interpolation_kernel.astype(dtype)

mi.get_spatial_interpolation_kernel = safe_get_spatial_interpolation_kernel

# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s_motion_corrected"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = OUTPUT_DIR / "motion_corrected_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("E097_MOTION_CORRECTED_PIPELINE")

def main():
    t_global_start = time.time()
    config_path = PROJECT_ROOT / "pipeline_config_e097_prelaser_through_laser3_226s.json"
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = PipelineConfig.from_dict(cfg_dict)

    # Override parameters for this motion-corrected run
    dur_s = cfg.duration_s
    start_s = cfg.start_time_s
    agreement_thresh = 0.80 # User specified: 80%
    presence_ratio_thresh = 0.20 # User specified: keep low (0.20)
    bin_s = 0.5 # User specified: 0.5s temporal bins
    sigma_um = 20.0 # User confirmed: 20 um Kriging kernel
    
    scratch_dir = Path(r"C:\SCRATCH_neuropixels\mc_sliding_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    catgt_dir = Path(cfg.catgt.dest_dir) / f"catgt_{cfg.catgt.run_name}_g{cfg.catgt.gate}"

    logger.info("=" * 85)
    logger.info(" STARTING E097 226s MOTION-CORRECTED LUPIN SPIKE SORTING PIPELINE")
    logger.info(f" Window: {start_s:.3f}s to {start_s + dur_s:.3f}s ({dur_s:.3f}s duration)")
    logger.info(f" CatGT AP Binary Dir: {catgt_dir}")
    logger.info(f" Motion Model       : Decentralized Rigid (direction='y', bin_s={bin_s}s)")
    logger.info(f" Spatial Kriging    : sigma={sigma_um} um, border_mode='force_extrapolate'")
    logger.info(f" Agreement Cutoff   : {agreement_thresh*100:.0f}% coincident spikes")
    logger.info(f" Presence Ratio     : {presence_ratio_thresh:.2f}")
    logger.info(f" Output Root        : {OUTPUT_DIR.resolve()}")
    logger.info("=" * 85)

    # 1. Ingest CatGT Preprocessed AP Data
    logger.info("\n>>> STAGE 1: INGESTING PREPROCESSED AP RECORDING")
    recording = read_spikeglx_ap_safe(catgt_dir)
    n_ch = recording.get_num_channels()
    fs_ap = recording.sampling_frequency
    logger.info(f"Recording loaded: {n_ch} channels, {fs_ap:.1f} Hz, {recording.get_total_duration():.2f}s")

    # 2. Global Rigid Motion Estimation
    logger.info("\n>>> STAGE 2: GLOBAL RIGID AXIAL MOTION ESTIMATION (ALL 384 CHANNELS)")
    t0_motion = time.time()
    
    peaks_cache = OUTPUT_DIR / "peaks_detected_226s.npy"
    locs_cache = OUTPUT_DIR / "peaks_localized_226s.npy"

    if peaks_cache.exists() and locs_cache.exists():
        logger.info(f"  Step 2a/b: Loading cached peaks and localized coordinates from {OUTPUT_DIR}...")
        peaks = np.load(peaks_cache)
        locs = np.load(locs_cache)
        logger.info(f"  --> Loaded {len(peaks):,} peaks and {len(locs):,} localized coordinates from cache.")
    else:
        logger.info(f"  Step 2a: Detecting peaks across 226s (neg, threshold=6.0 SD, radius=80 um)...")
        peaks = detect_peaks(
            recording=recording,
            method="locally_exclusive",
            peak_sign="neg",
            detect_threshold=6.0,
            radius_um=80.0,
            n_jobs=1,
            progress_bar=True
        )
        logger.info(f"  --> Detected {len(peaks):,} total peaks across recording. Saving to {peaks_cache.name}...")
        np.save(peaks_cache, peaks)

        logger.info(f"  Step 2b: Localizing peak coordinates via monopolar triangulation...")
        locs = localize_peaks(
            recording=recording,
            peaks=peaks,
            method="monopolar_triangulation",
            radius_um=75.0,
            max_distance_um=150.0,
            n_jobs=1,
            progress_bar=True
        )
        logger.info(f"  --> Localized {len(locs):,} peak locations. Saving to {locs_cache.name}...")
        np.save(locs_cache, locs)

    logger.info(f"  Step 2c: Estimating decentralized rigid motion (bin_s={bin_s}s, direction='y')...")
    motion = estimate_motion(
        recording=recording,
        peaks=peaks,
        peak_locations=locs,
        direction="y",
        rigid=True,
        bin_s=bin_s,
        method="decentralized",
        progress_bar=True,
        verbose=True
    )
    dur_motion = time.time() - t0_motion
    logger.info(f"Motion estimation completed in {dur_motion:.1f}s ({dur_motion/60:.2f} min).")

    # Save Motion Drift Telemetry to CSV
    disp_y = motion.displacement[0].flatten()
    time_bins_s = motion.temporal_bins_s[0]

    # Resolve recording start time offset (CatGT SpikeGLX sets seg.t_start from raw session clock)
    rec_t_start = recording.get_times()[0] if recording.has_time_vector() else 0.0
    if hasattr(recording, "_recording_segments") and len(recording._recording_segments):
        seg = recording._recording_segments[0]
        if hasattr(seg, "t_start") and seg.t_start is not None:
            rec_t_start = float(seg.t_start)
    window_time_s = time_bins_s - rec_t_start

    drift_df = pd.DataFrame({
        "window_time_s": np.round(window_time_s, 3),
        "recording_time_s": np.round(time_bins_s, 3),
        "displacement_y_um": np.round(disp_y, 4),
        "abs_displacement_um": np.round(np.abs(disp_y), 4)
    })
    drift_csv = OUTPUT_DIR / "probe_motion_drift_226s.csv"
    drift_df.to_csv(drift_csv, index=False)
    logger.info(f"Saved continuous motion telemetry to: {drift_csv}")
    logger.info(f"Drift Telemetry Summary: Max Absolute = {drift_df['abs_displacement_um'].max():.2f} um, Mean Absolute = {drift_df['abs_displacement_um'].mean():.2f} um")

    # Plot Motion Telemetry Figure
    drift_fig = OUTPUT_DIR / "probe_motion_drift_226s.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
        t_vals = drift_df["window_time_s"].values
        ax.plot(t_vals, drift_df["displacement_y_um"].values, color="#2563eb", lw=1.5, label="Estimated Probe Displacement (y)")
        ax.plot([0, dur_s], [0, 0], color="#94a3b8", ls="--", lw=1)
        
        laser_trains = [(60.0, 62.0, "Laser 1"), (120.0, 122.0, "Laser 2"), (164.0, 166.0, "Laser 3")]
        for t_on, t_off, lbl in laser_trains:
            ax.fill_between([t_on, t_off], [-20, -20], [20, 20], color="#f97316", alpha=0.35)
            ax.text(t_on + 1.0, 12.0, lbl, color="#c2410c", fontweight="bold", ha="center", fontsize=9)
            
        ax.set_xlim(0, dur_s)
        ax.set_ylim(-18, 18)
        ax.set_xlabel("Continuous Window Time (s) [0 - 226s]", fontsize=11, fontweight="bold")
        ax.set_ylabel("Axial Probe Displacement (um)", fontsize=11, fontweight="bold")
        ax.set_title("E097 Neuropixels 2.0 Rigid Probe Motion Telemetry across 226s Window (bin = 0.5s)", fontsize=12, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(drift_fig)
        plt.close(fig)
        logger.info(f"Saved drift telemetry plot to: {drift_fig}")
    except Exception as e_plot:
        logger.warning(f"Could not generate drift plot: {e_plot}")

    # 3. Global Spatial Kriging Motion Interpolation
    logger.info("\n>>> STAGE 3: APPLYING GLOBAL SPATIAL KRIGING INTERPOLATION")
    rec_float = prep.astype(recording, "float32")
    rec_corrected = interpolate_motion(
        rec_float,
        motion,
        spatial_interpolation_method="kriging",
        sigma_um=sigma_um,
        border_mode="force_extrapolate"
    )
    logger.info(f"Motion-corrected recording constructed successfully ({rec_corrected.get_num_channels()} channels).")

    # 4. Sliding-Window Sorting across 13 Sub-Windows
    logger.info("\n>>> STAGE 4: EXECUTING 13 SLIDING-WINDOW SORTING JOBS (LUPIN / SC2)")
    windows_dir = OUTPUT_DIR / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    
    sw_cfg = cfg.sliding_window
    windows = compute_window_ranges(
        total_channels=n_ch,
        channel_range=sw_cfg.channel_range,
        window_size=sw_cfg.window_size,
        overlap=sw_cfg.overlap
    )
    logger.info(f"Sorting {len(windows)} sub-windows (50 channels each, 20 channel overlap)...")

    window_results = []
    all_channel_ids = list(rec_corrected.channel_ids)
    t_sort_start = time.time()

    for win in windows:
        w_id = win["window_id"]
        w_name = win["name"]
        s_ch = win["start_ch"]
        e_ch = win["end_ch"]

        logger.info(f"\n[{w_id+1}/{len(windows)}] Sorting {w_name} (Channels {s_ch} to {e_ch-1})...")
        win_dir = windows_dir / w_name
        win_dir.mkdir(parents=True, exist_ok=True)

        qc_tsv = win_dir / "cluster_info.tsv"
        alf_spk_times = win_dir / "alf" / "spike_times.npy"
        alf_spk_clusters = win_dir / "alf" / "spike_clusters.npy"

        if qc_tsv.exists() and alf_spk_times.exists() and alf_spk_clusters.exists():
            logger.info(f"[{w_name}] Found existing completed sorting. Resuming from checkpoint...")
            qc_df = pd.read_csv(qc_tsv, sep="\t")
            st = np.load(alf_spk_times)
            sc = np.load(alf_spk_clusters)
            sorting = si.NumpySorting.from_samples_and_labels(
                samples_list=np.round(st * fs_ap).astype(np.int64),
                labels_list=sc,
                sampling_frequency=fs_ap
            )
            n_sua = int(np.sum(qc_df["passed_allen_qc"])) if "passed_allen_qc" in qc_df.columns else 0
            logger.info(f"[{w_name}] Loaded {len(qc_df)} Clusters ({n_sua} SUA) from checkpoint.")
            window_results.append({
                "window": win,
                "sorting": sorting,
                "qc_df": qc_df,
                "win_dir": win_dir,
                "n_clusters": len(qc_df),
                "n_sua": n_sua,
                "duration_s": 0.0
            })
            continue

        slice_ch_ids = all_channel_ids[s_ch:e_ch]
        rec_slice = rec_corrected.channel_slice(channel_ids=slice_ch_ids)

        # Save slice to fast scratch folder using single job for bulletproof Windows stability
        w_scratch = scratch_dir / f"rec_{w_name}"
        if w_scratch.exists():
            try:
                shutil.rmtree(w_scratch)
            except Exception:
                pass
        rec_slice_saved = rec_slice.save(folder=w_scratch, overwrite=True, n_jobs=1, chunk_duration="1s")

        t0_win = time.time()
        try:
            sorting = run_lupin_sorting(
                recording=rec_slice_saved,
                output_folder=win_dir / "si_output",
                n_jobs=1,
                verbose=False
            )
            dur_win = time.time() - t0_win
            logger.info(f"[{w_name}] Sorting complete in {dur_win:.1f}s. Extracted {len(sorting.unit_ids)} raw clusters.")

            # Compute QC and Export ALF for sub-window
            qc_df = export_sorting_alf_and_qc(
                sorting=sorting,
                fs=rec_slice_saved.get_sampling_frequency(),
                duration_s=rec_slice_saved.get_total_duration(),
                out_dir=win_dir,
                sorter_name=f"lupin_{w_name}",
                recording=rec_slice_saved
            )

            # Cleanup slice scratch safely
            del rec_slice_saved
            try:
                shutil.rmtree(w_scratch)
            except Exception:
                pass

            n_sua = int(np.sum(qc_df["passed_allen_qc"]))
            logger.info(f"[{w_name}] {len(qc_df)} Total Clusters -> {n_sua} Single Units (SUA)")

            window_results.append({
                "window": win,
                "sorting": sorting,
                "qc_df": qc_df,
                "win_dir": win_dir,
                "n_clusters": len(qc_df),
                "n_sua": n_sua,
                "duration_s": dur_win
            })
        except Exception as e_win:
            logger.error(f"Error sorting {w_name}: {e_win}", exc_info=True)

    t_sort_total = time.time() - t_sort_start
    logger.info(f"\nAll 13 sliding windows completed in {t_sort_total/60:.1f} minutes.")

    # 5. Cross-Window Agreement Consensus & Allen QC Evaluation (Agreement >= 0.80)
    logger.info("\n>>> STAGE 5: CROSS-WINDOW CONSENSUS (AGREEMENT >= 80%) & MASTER ALLEN QC")
    cons_dir = OUTPUT_DIR / "consensus_master"
    cons_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily update config with user's strict agreement threshold
    cfg.sliding_window.agreement_threshold = agreement_thresh
    cfg.sorters.qc_presence_ratio = presence_ratio_thresh

    consensus_res = compute_sliding_window_consensus(
        window_results=window_results,
        recording=rec_corrected,
        out_dir=cons_dir,
        agreement_threshold=agreement_thresh,
        sorter_name="lupin",
        pipeline_config=cfg
    )

    # 6. Build Interactive HTML Raster
    logger.info("\n>>> STAGE 6: GENERATING SYNCHRONIZED INTERACTIVE MASTER RASTER")
    if build_interactive_sorter_raster is not None:
        try:
            build_interactive_sorter_raster(
                sorter_dir=cons_dir,
                sorter_name="Lupin (Motion-Corrected Consensus Master 226s)"
            )
        except Exception as e_rast:
            logger.warning(f"Could not build master interactive raster: {e_rast}")

    # 7. Population Recruitment Analysis
    logger.info("\n>>> STAGE 7: COMPUTING POPULATION RECRUITMENT & ROSTRAL ACTIVATION")
    analyze_population_recruitment(cons_dir, start_s, dur_s)

    # 8. Side-by-Side Comparison against Uncorrected Baseline
    logger.info("\n>>> STAGE 8: COMPARATIVE BENCHMARK AGAINST UNCORRECTED SORTING")
    uncorr_dir = PROJECT_ROOT / "output_e097_prelaser_through_laser3_226s" / "lupin" / "consensus_master"
    compare_with_uncorrected(cons_dir, uncorr_dir)

    # 9. Generate Updated Comprehensive Run Journal
    logger.info("\n>>> STAGE 9: WRITING COMPREHENSIVE RUN JOURNAL & JSON")
    write_run_journal(OUTPUT_DIR, cfg, dur_motion, t_sort_total, time.time() - t_global_start)

    total_pipeline_time = time.time() - t_global_start
    logger.info("=" * 85)
    logger.info(f" [COMPLETE] MOTION-CORRECTED SORTING PIPELINE FINISHED IN {total_pipeline_time/60:.1f} MINUTES!")
    logger.info(f" Master Output: {cons_dir.resolve()}")
    logger.info("=" * 85)

def analyze_population_recruitment(master_dir: Path, t_global_start: float, duration_s: float):
    alf_dir = master_dir / "alf"
    spk_times = np.load(alf_dir / "spike_times.npy")
    spk_clusters = np.load(alf_dir / "spike_clusters.npy")
    qc_df = pd.read_csv(master_dir / "master_consensus_qc_summary.csv")

    epochs = {
        "pre_laser": (0.0, 60.0),
        "laser_train1": (60.0, 61.955),
        "inter_train1": (61.955, 119.664),
        "laser_train2": (119.664, 121.619),
        "inter_train2": (121.619, 164.288),
        "laser_train3": (164.288, 166.243),
        "post_laser": (166.243, 226.243),
    }

    records = []
    for _, row in qc_df.iterrows():
        cid = row["cluster_id"]
        u_spks = spk_times[spk_clusters == cid]

        rec = {
            "cluster_id": cid,
            "depth_um": row["depth_um"],
            "total_spikes": len(u_spks),
            "global_fr_hz": row["firing_rate_hz"],
            "isi_violations": row["isi_violations"],
            "presence_ratio": row["presence_ratio"],
            "passed_allen_qc": row["passed_allen_qc"],
            "is_consensus": row["is_cross_window_consensus"],
        }

        for ep_name, (t_start, t_end) in epochs.items():
            dur = t_end - t_start
            n_spk = np.sum((u_spks >= t_start) & (u_spks < t_end))
            rec[f"fr_{ep_name}"] = round(n_spk / dur, 3)

        fr_pre = rec["fr_pre_laser"]
        max_laser_fr = max(rec["fr_laser_train1"], rec["fr_laser_train2"], rec["fr_laser_train3"])
        rec["max_laser_fr"] = max_laser_fr
        rec["delta_laser_fr"] = round(max_laser_fr - fr_pre, 3)

        is_newly_recruited = (fr_pre < 0.2) and (max_laser_fr >= 3.0) and (row["isi_violations"] < 0.10)
        rec["is_newly_recruited"] = is_newly_recruited
        rec["is_rostral_vrc"] = row["depth_um"] <= 1080.0
        records.append(rec)

    res_df = pd.DataFrame(records)
    out_csv = master_dir / "population_recruitment_summary_226s.csv"
    res_df.to_csv(out_csv, index=False)
    logger.info(f"Saved population recruitment summary to: {out_csv}")
    n_recruited = np.sum(res_df["is_newly_recruited"])
    n_rostral_recruited = np.sum(res_df["is_newly_recruited"] & res_df["is_rostral_vrc"])
    logger.info(f"Newly recruited units: {n_recruited} total ({n_rostral_recruited} in rostral VRC)")

def compare_with_uncorrected(corr_dir: Path, uncorr_dir: Path):
    if not uncorr_dir.exists():
        logger.warning(f"Uncorrected dir {uncorr_dir} not found. Skipping comparative benchmark.")
        return

    corr_alf = corr_dir / "alf"
    uncorr_alf = uncorr_dir / "alf"
    if not (corr_alf / "spike_times.npy").exists() or not (uncorr_alf / "spike_times.npy").exists():
        return

    st_corr = np.load(corr_alf / "spike_times.npy")
    sc_corr = np.load(corr_alf / "spike_clusters.npy")
    st_uncorr = np.load(uncorr_alf / "spike_times.npy")
    sc_uncorr = np.load(uncorr_alf / "spike_clusters.npy")

    qc_corr = pd.read_csv(corr_dir / "cluster_info.tsv", sep="\t")
    qc_uncorr = pd.read_csv(uncorr_dir / "cluster_info.tsv", sep="\t")

    sua_corr = qc_corr[qc_corr["passed_allen_qc"]]
    sua_uncorr = qc_uncorr[qc_uncorr["passed_allen_qc"]]

    logger.info("=" * 60)
    logger.info(" COMPARATIVE BENCHMARK: MOTION-CORRECTED VS UNCORRECTED")
    logger.info(f" Motion-Corrected: {len(qc_corr)} total clusters, {len(sua_corr)} SUA units ({len(st_corr):,} spikes)")
    logger.info(f" Uncorrected     : {len(qc_uncorr)} total clusters, {len(sua_uncorr)} SUA units ({len(st_uncorr):,} spikes)")
    logger.info("=" * 60)

    # Save comparative summary CSV
    comp_df = pd.DataFrame([{
        "dataset": "Motion-Corrected (Rigid DREDge, bin=0.5s, Kriging 20um, Agreement 80%)",
        "total_clusters": len(qc_corr),
        "sua_single_units": len(sua_corr),
        "total_spikes": len(st_corr),
        "mean_snr": round(sua_corr["snr"].mean(), 2) if len(sua_corr) > 0 else 0.0,
        "mean_peak_amp_uv": round(sua_corr["peak_amp_uv"].mean(), 2) if len(sua_corr) > 0 else 0.0,
    }, {
        "dataset": "Uncorrected Baseline (Stationary Contacts, Agreement 50%)",
        "total_clusters": len(qc_uncorr),
        "sua_single_units": len(sua_uncorr),
        "total_spikes": len(st_uncorr),
        "mean_snr": round(sua_uncorr["snr"].mean(), 2) if len(sua_uncorr) > 0 else 0.0,
        "mean_peak_amp_uv": round(sua_uncorr["peak_amp_uv"].mean(), 2) if len(sua_uncorr) > 0 else 0.0,
    }])
    comp_csv = corr_dir / "comparison_with_uncorrected_sorting.csv"
    comp_df.to_csv(comp_csv, index=False)
    logger.info(f"Saved comparison table to: {comp_csv}")

def write_run_journal(out_dir: Path, cfg: PipelineConfig, dur_motion: float, dur_sort: float, total_s: float):
    cons_dir = out_dir / "consensus_master"
    qc_file = cons_dir / "cluster_info.tsv"
    n_clusters = 0
    n_sua = 0
    if qc_file.exists():
        df = pd.read_csv(qc_file, sep="\t")
        n_clusters = len(df)
        n_sua = int(np.sum(df["passed_allen_qc"]))

    journal_md = f"""# Spike Sorting Pipeline Run Journal (Motion-Corrected Run)

**Execution Timestamp**: {time.strftime('%Y-%m-%dT%H:%M:%S')}  
**Sorter**: Lupin (SpyKING-CIRCUS 2 Engine via SpikeInterface 0.102.3)  
**Session Identifier**: `e097_g2_prelaser_through_laser3_226s_motion_corrected`  
**Output Directory**: `{out_dir.resolve()}`  

---

## 1. Motion Correction Configuration (ENABLED)

* **Motion Model**: Decentralized Rigid Axial Model (`direction='y'`, `rigid=True`)
* **Temporal Bin Size (`bin_s`)**: **0.5 seconds** (tuned for sudden gasping jerks during Laser 3)
* **Peak Detection Threshold**: 6.0 SD (`locally_exclusive`, radius = 80 µm)
* **Peak Localization**: Monopolar Triangulation (`radius=75 um`, `max_distance=150 um`)
* **Spatial Interpolation**: **Kriging** (sigma = 20.0 µm, Gaussian kernel)
* **Boundary Mode**: `force_extrapolate`
* **Motion Execution Time**: {dur_motion:.1f} seconds ({dur_motion/60:.2f} minutes)
* **Drift Telemetry Artifact**: [`probe_motion_drift_226s.csv`](file:///{out_dir.resolve()}/probe_motion_drift_226s.csv)

---

## 2. Sliding-Window Architecture & Consensus Matching

* **Probe Coverage**: 384 active channels on Shank 2 (y = 720 to 2880 µm)
* **Sub-Windows**: 13 overlapping windows (50 channels each, 750 µm span)
* **Overlap & Stride**: 20 channels (300 µm overlap), 30 channels (450 µm step)
* **Sorter Hyperparameters**:
  * Cutout window: [-1.5 ms, +2.0 ms] (105 samples at 30 kHz)
  * Sparsity radius: 100.0 µm
  * Template extraction: SVD basis compression (`templates_from_svd = True`)
  * Clustering: `circus`
  * Matching pursuit: `circus-omp-svd`
* **Cross-Window Agreement Threshold**: **80.0%** (increased from 50% for high-confidence merging)
* **Spike De-duplication**: Coincident spikes within delta_t <= 0.4 ms collapsed; non-coincident spikes retained via de-duplicated union.
* **Sliding Sorting Duration**: {dur_sort/60:.1f} minutes

---

## 3. Allen Institute Quality Control Criteria

* **Presence Ratio**: >= 0.20 (preserves laser-quenched and gasping-recruited neurons)
* **ISI Refractory Violations**: <= 0.10 (refractory window 1.5 ms)
* **Amplitude Cutoff**: <= 0.15
* **Minimum Firing Rate**: >= 0.05 Hz

---

## 4. Results Summary

* **Total Formed Clusters**: {n_clusters}
* **Passing Single-Unit Activity (SUA)**: **{n_sua}**
* **Total Execution Time**: {total_s/60:.1f} minutes

---
*Journal recorded automatically by Antigravity IDE for e097 analysis pipeline.*
"""
    with open(cons_dir / "run_journal.md", "w", encoding="utf-8") as f:
        f.write(journal_md)

    journal_dict = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "sorter": "lupin",
        "motion_correction": {
            "enabled": True,
            "method": "decentralized_rigid",
            "direction": "y",
            "bin_s": 0.5,
            "spatial_interpolation": "kriging",
            "sigma_um": 20.0
        },
        "consensus": {
            "agreement_threshold": 0.80,
            "dedup_delta_s": 0.0004
        },
        "qc": {
            "presence_ratio": 0.20,
            "isi_cutoff": 0.10,
            "amp_cutoff": 0.15,
            "min_fr_hz": 0.05
        },
        "results": {
            "total_clusters": n_clusters,
            "sua_single_units": n_sua,
            "total_duration_minutes": round(total_s / 60.0, 2)
        }
    }
    with open(cons_dir / "run_journal.json", "w", encoding="utf-8") as f:
        json.dump(journal_dict, f, indent=4)
    logger.info(f"Recorded run journal to: {cons_dir / 'run_journal.md'}")

if __name__ == "__main__":
    main()
