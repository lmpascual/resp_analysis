"""
Modular Motion Correction Engine for Neuropixels recordings via SpikeInterface.
Supports:
1. Step-by-step peak detection, localization, motion estimation, and kriging interpolation.
2. Checkpointed caching to NVMe scratch with parameter inspection.
3. Multi-method support (kilosort_like, dredge, nonrigid_fast_and_accurate).
"""
import time
import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion
from spikeinterface.core.core_tools import SIJsonEncoder

from src.pipeline.config import MotionConfig
from .motion_presets import get_custom_motion_params

logger = logging.getLogger("MOTION_CORRECTOR")

def compute_and_cache_noise_levels(
    recording: si.BaseRecording,
    cache_folder: Path
) -> np.ndarray:
    """Computes or loads per-channel noise levels (MAD-based)."""
    cache_folder = Path(cache_folder)
    noise_file = cache_folder / "noise_levels.npy"
    if noise_file.exists():
        logger.info(f"Loading cached noise levels from {noise_file}")
        return np.load(noise_file)
    
    logger.info("Computing per-channel noise levels (MAD)...")
    noise_levels = si.get_noise_levels(recording, return_scaled=False)
    cache_folder.mkdir(parents=True, exist_ok=True)
    np.save(noise_file, noise_levels)
    return noise_levels

def run_stepwise_motion_pipeline(
    recording: si.BaseRecording,
    config: MotionConfig,
    working_dir: Path,
    n_jobs: int = 12,
    chunk_duration: str = "1s"
) -> Tuple[si.BaseRecording, Dict[str, Any]]:
    """
    Executes the modular motion pipeline step-by-step with checkpointed disk caching.
    """
    working_dir = Path(working_dir)
    peaks_dir = working_dir / "peaks_and_locs"
    motion_dir = working_dir / "motion_estimation"
    interp_dir = working_dir / "interpolated_recording"

    peaks_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration=chunk_duration, progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)

    params = get_custom_motion_params(
        preset=config.preset,
        bin_duration_s=config.bin_duration_s,
        peak_threshold=config.peak_detect_threshold,
        spatial_bin_um=config.spatial_bin_um,
        detect_method=config.peak_detect_method,
        localize_method=config.peak_localize_method,
        interpolate_method=config.interpolate_method
    )

    # Save parameter manifest
    (working_dir / "motion_params.json").write_text(
        json.dumps(params, indent=4, cls=SIJsonEncoder), encoding="utf8"
    )

    # 1. Noise levels
    noise_levels = compute_and_cache_noise_levels(recording, peaks_dir)
    params["detect_kwargs"]["noise_levels"] = noise_levels

    # 2. Peak Detection
    peaks_file = peaks_dir / "peaks.npy"
    if peaks_file.exists():
        logger.info(f"Loading cached peaks from {peaks_file}")
        peaks = np.load(peaks_file)
    else:
        logger.info(f"Detecting peaks (method={config.peak_detect_method}, threshold={config.peak_detect_threshold})...")
        t0 = time.time()
        peaks = detect_peaks(recording=recording, **params["detect_kwargs"], **job_kwargs)
        t_detect = time.time() - t0
        np.save(peaks_file, peaks)
        logger.info(f"Detected {len(peaks):,} peaks in {t_detect:.1f}s.")

    # 3. Peak Localization
    locs_file = peaks_dir / "peak_locations.npy"
    if locs_file.exists():
        logger.info(f"Loading cached peak locations from {locs_file}")
        peak_locations = np.load(locs_file)
    else:
        logger.info(f"Localizing peaks (method={config.peak_localize_method})...")
        t0 = time.time()
        peak_locations = localize_peaks(
            recording=recording,
            peaks=peaks,
            **params["localize_peaks_kwargs"],
            **job_kwargs
        )
        t_loc = time.time() - t0
        np.save(locs_file, peak_locations)
        logger.info(f"Localized {len(peak_locations):,} peak locations in {t_loc:.1f}s.")

    # 4. Motion Estimation
    results_pkl = motion_dir / "motion_info.pkl"
    if results_pkl.exists():
        logger.info(f"Loading cached motion estimate from {results_pkl}")
        with open(results_pkl, "rb") as f:
            motion_info = pickle.load(f)
        motion = motion_info["motion"]
    else:
        logger.info(f"Estimating motion (method={config.estimate_motion_method}, bin_duration={config.bin_duration_s}s)...")
        t0 = time.time()
        
        # In SpikeInterface 0.102+, estimate_motion returns motion (or motion, motion_info)
        est_result = estimate_motion(
            recording=recording,
            peaks=peaks,
            peak_locations=peak_locations,
            extra_outputs=True,
            **params["estimate_motion_kwargs"]
        )
        t_motion = time.time() - t0
        
        if isinstance(est_result, tuple):
            motion, motion_info = est_result
        else:
            motion = est_result
            motion_info = {}

        motion_info.update({
            "parameters": params,
            "runtime_s": t_motion,
            "peaks": peaks,
            "peak_locations": peak_locations,
            "motion": motion,
        })

        with open(results_pkl, "wb") as f:
            pickle.dump(motion_info, f)
        logger.info(f"Motion estimation finished in {t_motion:.1f}s.")

    # 5. Motion Interpolation (Kriging to drift-corrected binary)
    if interp_dir.exists() and (interp_dir / "traces_cached_seg0.raw").exists():
        logger.info(f"Loading cached interpolated recording from {interp_dir}")
        corrected_rec = si.load(interp_dir)
    else:
        logger.info(f"Interpolating recording with {config.interpolate_method}...")
        t0 = time.time()
        
        # interpolate_motion requires floating point traces
        if np.issubdtype(recording.dtype, np.integer):
            rec_for_interp = spre.astype(recording, dtype="float32")
        else:
            rec_for_interp = recording

        interp_kwargs = params["interpolate_motion_kwargs"].copy()
        spatial_method = interp_kwargs.pop("spatial_interpolation_method", "kriging")
        interp_kwargs.pop("method", None)
        
        corrected_rec_lazy = interpolate_motion(
            recording=rec_for_interp,
            motion=motion,
            spatial_interpolation_method=spatial_method,
            **interp_kwargs
        )

        logger.info(f"Caching drift-corrected binary to NVMe scratch at {interp_dir}...")
        if interp_dir.exists():
            import shutil
            shutil.rmtree(interp_dir, ignore_errors=True)
            
        corrected_rec = corrected_rec_lazy.save(
            folder=interp_dir,
            format="binary",
            **job_kwargs,
            overwrite=True
        )
        t_interp = time.time() - t0
        logger.info(f"Drift-corrected recording saved in {t_interp:.1f}s.")

    return corrected_rec, motion_info
