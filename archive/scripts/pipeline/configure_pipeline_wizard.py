#!/usr/bin/env python3
"""
Interactive Configuration Wizard for Neuropixels Pipeline
Prompts the user for all pipeline parameters with clear default values,
validates user input, and writes the verified settings to pipeline_config.json.
"""
import sys
import json
import re
import math
from pathlib import Path
from typing import Tuple, List, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.config import PipelineConfig

def prompt_bool(message: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    val = input(f"{message} [{default_str}]: ").strip().lower()
    if not val or val == "y" or val == "yes":
        return True if default else False
    if val == "n" or val == "no":
        return False
    return default

def prompt_string(message: str, default: str) -> str:
    val = input(f"{message} [y for '{default}' or type custom]: ").strip()
    if not val or val.lower() == "y" or val.lower() == "yes":
        return default
    return val

def prompt_float_tuple(message: str, default: Tuple[float, float]) -> Tuple[float, float]:
    val = input(f"{message} [y for {list(default)} or type custom e.g. [50, 200]]: ").strip()
    if not val or val.lower() == "y" or val.lower() == "yes":
        return default
    try:
        # Clean brackets/spaces and parse
        cleaned = val.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [float(p.strip()) for p in cleaned.split(",") if p.strip()]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except Exception:
        pass
    print(f"  --> Could not parse '{val}', using default {list(default)}")
    return default

def prompt_int_range(message: str, default: Tuple[int, int]) -> Tuple[int, int]:
    val = input(f"{message} [y for '{default[0]}-{default[1]}' or type custom e.g. 0-330]: ").strip()
    if not val or val.lower() == "y" or val.lower() == "yes":
        return default
    try:
        cleaned = val.replace("[", "").replace("]", "").replace(" ", "")
        if "-" in cleaned:
            parts = [int(p) for p in cleaned.split("-") if p]
            if len(parts) == 2:
                return (parts[0], parts[1])
        elif "," in cleaned:
            parts = [int(p) for p in cleaned.split(",") if p]
            if len(parts) == 2:
                return (parts[0], parts[1])
    except Exception:
        pass
    print(f"  --> Could not parse '{val}', using default {default[0]}-{default[1]}")
    return default

def prompt_int(message: str, default: int) -> int:
    val = input(f"{message} [y for {default} or type integer]: ").strip()
    if not val or val.lower() == "y" or val.lower() == "yes":
        return default
    try:
        return int(val)
    except Exception:
        return default

def prompt_float_or_none(message: str, default: Any) -> Any:
    default_repr = "None (Full)" if default is None else str(default)
    val = input(f"{message} [y for {default_repr} or type seconds]: ").strip()
    if not val or val.lower() == "y" or val.lower() == "yes":
        return default
    if val.lower() == "none":
        return None
    try:
        return float(val)
    except Exception:
        return default

def run_wizard(config_path: Path = Path("pipeline_config.json")) -> PipelineConfig:
    print("\n" + "=" * 80)
    print(" [CONFIG WIZARD] INTERACTIVE NEUROPIXELS PIPELINE CONFIGURATION")
    print("=" * 80)
    print("Press [Enter] or type 'y' to accept recommended defaults, or type custom values.\n")

    # Load existing config if available
    cfg = PipelineConfig()
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = PipelineConfig.from_dict(json.load(f))
            print(f"Loaded existing configuration from: {config_path}")
        except Exception:
            pass

    # 1. Recording Path & Output Directory
    print("\n" + "-" * 70)
    print(" 1. Recording Ingestion & Output Paths")
    print("-" * 70)
    cfg.raw_data_dir = Path(prompt_string("Raw / Preprocessed Ephys Path", str(cfg.raw_data_dir)))
    cfg.output_dir = Path(prompt_string("Output Directory Base", str(cfg.output_dir)))

    # 2. CatGT Preprocessing Parameters
    print("\n" + "-" * 70)
    print(" 2. CatGT Filter & Artifact Removal Settings")
    print("-" * 70)
    cfg.catgt.enabled = prompt_bool("Enable CatGT Preprocessing?", cfg.catgt.enabled)
    if cfg.catgt.enabled:
        cfg.catgt.apfilter = prompt_string("CatGT AP Bandpass Filter", cfg.catgt.apfilter)
        cfg.catgt.loccar_um = prompt_float_tuple("Local CAR (loccar_um: exclude, include in µm)", cfg.catgt.loccar_um)
        cfg.catgt.start_secs = prompt_float_or_none("CatGT Start Time (start_secs)", cfg.catgt.start_secs or 0.0)
        cfg.catgt.max_secs = prompt_float_or_none("CatGT Max Duration (max_secs)", cfg.catgt.max_secs or 60.0)
        cfg.catgt.t_shift = prompt_bool("Apply SpikeGLX ADC Channel Time-Shift (t_shift)?", cfg.catgt.t_shift)
        cfg.start_time_s = cfg.catgt.start_secs or 0.0
        cfg.duration_s = cfg.catgt.max_secs

    # 3. Motion Correction Parameters
    print("\n" + "-" * 70)
    print(" 3. SpikeInterface Motion Drift Correction")
    print("-" * 70)
    cfg.motion.enabled = prompt_bool("Enable Motion Drift Estimation & Interpolation?", cfg.motion.enabled)
    if cfg.motion.enabled:
        print("\n  [Default Motion Parameters]:")
        print(f"    • Preset:               '{cfg.motion.preset}'")
        print(f"    • Peak Detection:       '{cfg.motion.peak_detect_method}' (Threshold: {cfg.motion.peak_detect_threshold} SD, Radius: {cfg.motion.peak_detect_radius_um} µm)")
        print(f"    • Peak Localization:    '{cfg.motion.peak_localize_method}'")
        print(f"    • Motion Estimation:    '{cfg.motion.estimate_motion_method}' (Bin Duration: {cfg.motion.bin_duration_s}s, Spatial Bin: {cfg.motion.spatial_bin_um} µm)")
        print(f"    • Interpolation Method: '{cfg.motion.interpolate_method}' (Kriging)\n")

        keep_motion_defaults = prompt_bool("Keep all recommended default motion parameters?", True)
        if not keep_motion_defaults:
            cfg.motion.bin_duration_s = float(prompt_string("Motion temporal bin duration (seconds)", str(cfg.motion.bin_duration_s)))
            cfg.motion.spatial_bin_um = float(prompt_string("Motion spatial bin resolution (µm)", str(cfg.motion.spatial_bin_um)))
            cfg.motion.interpolate_method = prompt_string("Interpolation method ('kriging' or 'nearest')", cfg.motion.interpolate_method)

    # 4. Sliding Window Spike Sorting
    print("\n" + "-" * 70)
    print(" 4. Iterative Sliding-Window Spike Sorting & Consensus Settings")
    print("-" * 70)
    cfg.sliding_window.enabled = prompt_bool("Enable Sliding-Window Sub-Channel Spike Sorting?", cfg.sliding_window.enabled)
    if cfg.sliding_window.enabled:
        print("  [Channel Range Verification]:")
        print("  Default is all 384 channels (0-384). You can verify or enter a custom subset (e.g. 0-330).")
        cfg.sliding_window.channel_range = prompt_int_range("Channel Selection Range (min-max)", cfg.sliding_window.channel_range)
        cfg.sliding_window.window_size = prompt_int("Sliding Window Channel Size (e.g. 50 channels)", cfg.sliding_window.window_size)
        cfg.sliding_window.overlap = prompt_int("Sliding Window Channel Overlap (e.g. 20 channels)", cfg.sliding_window.overlap)
        step = max(1, cfg.sliding_window.window_size - cfg.sliding_window.overlap)
        n_windows = int(math.ceil((cfg.sliding_window.channel_range[1] - cfg.sliding_window.channel_range[0] - cfg.sliding_window.overlap) / step))
        print(f"  --> Stride/Step = {step} channels. Total estimated windows across channels {cfg.sliding_window.channel_range[0]} to {cfg.sliding_window.channel_range[1]}: ~{n_windows} windows.")
        
        agr_val = input(f"Cross-Window Unit Agreement Threshold [y for {cfg.sliding_window.agreement_threshold:.2f} (50%) or type custom e.g. 0.90]: ").strip()
        if agr_val and agr_val.lower() not in ["y", "yes"]:
            try:
                cfg.sliding_window.agreement_threshold = float(agr_val)
            except Exception:
                pass

    # 5. Allen Single-Unit Quality Control Metrics
    print("\n" + "-" * 70)
    print(" 5. Allen Institute Single-Unit (SUA) Quality Metrics Criteria")
    print("-" * 70)
    print("  [Current Allen QC Thresholds]:")
    print(f"    • Max ISI Violation Rate (qc_isi_cutoff):    {cfg.sorters.qc_isi_cutoff:.2f} (Refractory violations < {cfg.sorters.qc_isi_cutoff*100:.0f}%)")
    print(f"    • Min Presence Ratio (qc_presence_ratio):    {cfg.sorters.qc_presence_ratio:.2f} (Active during >= {cfg.sorters.qc_presence_ratio*100:.0f}% of recording)")
    print(f"    • Max Amplitude Cutoff (qc_amp_cutoff):      {cfg.sorters.qc_amp_cutoff:.2f} (Spike amplitude distribution cutoff <= {cfg.sorters.qc_amp_cutoff*100:.0f}%)")
    print(f"    • Min Firing Rate (qc_min_fr_hz):            {cfg.sorters.qc_min_fr_hz:.2f} Hz")
    print(f"    • Refractory Period Window (isi_threshold): {cfg.sorters.isi_threshold_s*1000.0:.1f} ms\n")

    keep_qc_defaults = prompt_bool("Keep these recommended Allen Single-Unit QC criteria?", True)
    if not keep_qc_defaults:
        cfg.sorters.qc_isi_cutoff = float(prompt_string("Max ISI Violation Rate (qc_isi_cutoff)", str(cfg.sorters.qc_isi_cutoff)))
        cfg.sorters.qc_presence_ratio = float(prompt_string("Min Presence Ratio (qc_presence_ratio)", str(cfg.sorters.qc_presence_ratio)))
        cfg.sorters.qc_amp_cutoff = float(prompt_string("Max Amplitude Cutoff (qc_amp_cutoff)", str(cfg.sorters.qc_amp_cutoff)))
        cfg.sorters.qc_min_fr_hz = float(prompt_string("Min Firing Rate in Hz (qc_min_fr_hz)", str(cfg.sorters.qc_min_fr_hz)))
        cfg.sorters.isi_threshold_s = float(prompt_string("Refractory Period in seconds (isi_threshold_s)", str(cfg.sorters.isi_threshold_s)))

    # 6. Spike Sorter Selection
    print("\n" + "-" * 70)
    print(" 6. Spike Sorter Algorithm Selection")
    print("-" * 70)
    print("Available Sorters:")
    print("  [1] Lupin (Garcia et al. 2026 - Recommended)")
    print("  [2] MountainSort 5")
    print("  [3] Kilosort 4")
    print("  [4] Multi-Sorter Benchmark (All 3)")
    sorter_choice = input("Select sorter [1=Lupin (default), 2=MountainSort5, 3=Kilosort4, 4=All]: ").strip()
    if sorter_choice == "2":
        cfg.sorters.enabled_sorters = ["mountainsort5"]
    elif sorter_choice == "3":
        cfg.sorters.enabled_sorters = ["kilosort4"]
    elif sorter_choice == "4":
        cfg.sorters.enabled_sorters = ["lupin", "mountainsort5", "kilosort4"]
    else:
        cfg.sorters.enabled_sorters = ["lupin"]
    print(f"  --> Selected Sorter(s): {cfg.sorters.enabled_sorters}")

    # Save to JSON
    cfg.save_json(config_path)
    print("\n" + "=" * 80)
    print(f" [SUCCESS] CONFIGURATION SAVED TO: {config_path.resolve()}")
    print("=" * 80)
    return cfg

def main():
    cfg_file = Path("pipeline_config.json")
    if len(sys.argv) > 1:
        cfg_file = Path(sys.argv[1])
    run_wizard(cfg_file)

if __name__ == "__main__":
    main()
