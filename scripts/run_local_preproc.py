#!/usr/bin/env python3
"""
CLI entrypoint for running the modular neurophysiology and cardiorespiratory analysis pipeline.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.config import PipelineConfig, EMGConfig, ECGConfig, SyncOptoConfig, EphysConfig
from src.pipeline.pipeline_runner import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Modular Neurophysiology Pipeline: Neuropixel & Analog NIDQ Analysis (EMG, ECG, Sync, Opto)"
    )
    parser.add_argument("--animal-id", type=str, default="p05", help="Subject / Animal ID")
    parser.add_argument("--session-id", type=str, default="session_01", help="Session ID")
    parser.add_argument("--raw-dir", type=str, default="./raw_data", help="Path to raw data directory")
    parser.add_argument("--out-dir", type=str, default="./output", help="Path to output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to pre-existing JSON config/journal for exact replication")
    parser.add_argument("--non-interactive", action="store_true", help="Run without interactive parameter prompts")
    
    # Optional parameter flags
    parser.add_argument("--emg-highpass", type=float, default=300.0, help="EMG highpass cutoff (Hz)")
    parser.add_argument("--emg-lowpass", type=float, default=3000.0, help="EMG lowpass cutoff (Hz)")
    parser.add_argument("--emg-window-ms", type=float, default=50.0, help="EMG smoothing window (ms)")
    parser.add_argument("--emg-peak-sd", type=float, default=2.5, help="EMG peak threshold (SD)")
    parser.add_argument("--ecg-peak-sd", type=float, default=3.0, help="ECG R-peak threshold (SD)")
    parser.add_argument("--pulse-threshold", type=float, default=2.5, help="Opto/Sync pulse threshold (V)")

    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        import json
        with open(args.config, 'r') as f:
            cfg_dict = json.load(f)
            if "config" in cfg_dict:
                cfg_dict = cfg_dict["config"]
        config = PipelineConfig.from_dict(cfg_dict)
    else:
        config = PipelineConfig(
            animal_id=args.animal_id,
            session_id=args.session_id,
            raw_data_dir=Path(args.raw_dir),
            output_dir=Path(args.out_dir),
            emg=EMGConfig(
                highpass_cutoff=args.emg_highpass,
                lowpass_cutoff=args.emg_lowpass,
                smoothing_window_ms=args.emg_window_ms,
                peak_threshold_sd=args.emg_peak_sd
            ),
            ecg=ECGConfig(
                r_peak_threshold_sd=args.ecg_peak_sd
            ),
            sync_opto=SyncOptoConfig(
                pulse_threshold_v=args.pulse_threshold
            )
        )

    interactive = not args.non_interactive
    run_pipeline(config, interactive=interactive)

if __name__ == "__main__":
    main()
