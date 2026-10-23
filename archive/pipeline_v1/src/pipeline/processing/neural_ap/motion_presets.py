"""
Preset configurations and parameter factories for SpikeInterface motion estimation and correction.
"""
from typing import Dict, Any

def get_custom_motion_params(
    preset: str = "kilosort_like",
    bin_duration_s: float = 2.0,
    peak_threshold: float = 5.5,
    spatial_bin_um: float = 50.0,
    detect_method: str = "locally_exclusive",
    localize_method: str = "monopolar_triangulation",
    interpolate_method: str = "kriging"
) -> Dict[str, Any]:
    """
    Constructs a cohesive dictionary of parameters for modular motion correction.
    """
    detect_kwargs = {
        "method": detect_method,
        "peak_sign": "neg",
        "detect_threshold": float(peak_threshold),
        "radius_um": 100.0,
        "exclude_sweep_ms": 0.5,
    }

    localize_peaks_kwargs = {
        "method": localize_method,
        "radius_um": 100.0,
    }

    estimate_motion_kwargs = {
        "progress_bar": True,
        "verbose": True
    }

    if preset == "kilosort_like":
        estimate_motion_kwargs["method"] = "iterative_template"
        estimate_motion_kwargs["bin_s"] = float(bin_duration_s)
        estimate_motion_kwargs["rigid"] = False
        estimate_motion_kwargs["win_step_um"] = 200.0
        estimate_motion_kwargs["win_scale_um"] = 400.0
        estimate_motion_kwargs["hist_margin_um"] = 0
        estimate_motion_kwargs["win_shape"] = "rect"
    elif preset == "dredge":
        estimate_motion_kwargs["method"] = "dredge_ap"
        estimate_motion_kwargs["bin_s"] = float(bin_duration_s)
        estimate_motion_kwargs["spatial_bin_um"] = spatial_bin_um
    elif preset in ["nonrigid_accurate", "nonrigid_fast_and_accurate", "decentralized"]:
        estimate_motion_kwargs["method"] = "decentralized"
        estimate_motion_kwargs["bin_duration_s"] = float(bin_duration_s)
        estimate_motion_kwargs["spatial_bin_um"] = spatial_bin_um
        estimate_motion_kwargs["convergence_method"] = "lsqr_robust"
        estimate_motion_kwargs["weight_with_amplitude"] = True
    else:
        estimate_motion_kwargs["method"] = preset
        estimate_motion_kwargs["bin_duration_s"] = float(bin_duration_s)

    interpolate_motion_kwargs = {
        "border_mode": "force_extrapolate",
        "spatial_interpolation_method": "kriging" if interpolate_method == "kriging" else "nearest",
        "sigma_um": 20.0,
        "p": 1,
    }

    return {
        "preset": preset,
        "detect_kwargs": detect_kwargs,
        "localize_peaks_kwargs": localize_peaks_kwargs,
        "estimate_motion_kwargs": estimate_motion_kwargs,
        "interpolate_motion_kwargs": interpolate_motion_kwargs,
    }
