"""
Pipeline Runner Orchestration Engine.
Integrates parameter checkpoints, NIDQ + Neuropixel processors, diagnostic plotting, and provenance journaling.
"""
from pathlib import Path
import logging
import numpy as np
from typing import Optional, Dict, Any

from .config import PipelineConfig
from .models import SessionResults, AnalogData, EphysData
from .io.io_nidq import read_nidq_bin, generate_synthetic_nidq
from .io.io_ephys import read_alf_spikes, generate_synthetic_ephys
from .processors.proc_emg import process_emg
from .processors.proc_ecg import process_ecg
from .processors.proc_sync_opto import extract_pulse_events
from .processors.proc_ephys import compute_respiratory_modulation
from .journaler import checkpoint_approve_parameters, write_run_journal
from .visualizer import generate_diagnostic_dashboard

logger = logging.getLogger(__name__)

def run_pipeline(config: PipelineConfig, interactive: bool = True) -> Optional[SessionResults]:
    """
    Run full modular analysis pipeline with parameter checkpoints, visual stationarity inspection,
    and provenance journaling.
    """
    print(f"\n[*] Starting Modular Pipeline Execution: {config.animal_id} / {config.session_id}")

    # Step 1: Interactive Parameter Approval Checkpoint
    config, approved = checkpoint_approve_parameters(config, interactive=interactive)
    if not approved:
        print(" [!] Pipeline execution cancelled by user.")
        return None

    # Step 2: Load Data (NIDQ & Ephys)
    nidq_path = Path(config.raw_data_dir) / f"{config.animal_id}_{config.session_id}.nidq.bin"
    if nidq_path.exists():
        print(f" -> Reading NIDQ binary from {nidq_path}")
        analog_data = read_nidq_bin(nidq_path, fs=config.emg.fs)
    else:
        print(f" -> Raw NIDQ binary not found at {nidq_path}. Generating synthetic demo dataset.")
        analog_data = generate_synthetic_nidq(duration_s=60.0, fs=config.emg.fs)

    ephys_path = Path(config.raw_data_dir)
    ephys_data = read_alf_spikes(ephys_path)
    if ephys_data is None:
        print(f" -> Raw ALF ephys spikes not found. Generating synthetic demo Neuropixel units.")
        ephys_data = generate_synthetic_ephys(duration_s=60.0, num_units=8)

    # Step 3: Run Modality Processors
    print(" -> Processing Diaphragm EMG (Filtering, Envelope, Breath Peak Detection, SNR Stationarity)...")
    emg_results = process_emg(analog_data.emg_raw, analog_data.time_vector, config.emg)

    print(" -> Processing ECG (Filtering, R-Peaks, Instantaneous HR, HRV)...")
    ecg_results = process_ecg(analog_data.ecg_raw, analog_data.time_vector, config.ecg)

    print(" -> Extracting Sync & Optic Fiber Laser Pulse TTLs...")
    sync_pulses = extract_pulse_events(analog_data.sync_raw, analog_data.time_vector, config.sync_opto)
    opto_pulses = extract_pulse_events(analog_data.opto_raw, analog_data.time_vector, config.sync_opto)

    ephys_mod = {}
    if ephys_data is not None:
        print(" -> Computing Neuropixel Unit Respiratory Modulation & Depth Rasters...")
        ephys_mod = compute_respiratory_modulation(ephys_data, emg_results, config.ephys)

    session_results = SessionResults(
        animal_id=config.animal_id,
        session_id=config.session_id,
        analog_raw=analog_data,
        emg=emg_results,
        ecg=ecg_results,
        sync=sync_pulses,
        opto=opto_pulses,
        ephys=ephys_data
    )

    # Step 4: Generate Interactive Visual Diagnostic Plots
    print(" -> Rendering visual diagnostic plots for parameter confirmation & stationarity check...")
    generated_plots = generate_diagnostic_dashboard(session_results, config.output_dir)

    # Step 5: Write Provenance Journal & Summary
    exec_summary = {
        "duration_seconds": float(analog_data.time_vector[-1]),
        "breaths_detected": len(emg_results.breath_onsets),
        "mean_respiratory_rate_bpm": float(np.mean(emg_results.instantaneous_rr)) if len(emg_results.instantaneous_rr) > 0 else 0.0,
        "emg_mean_snr_db": float(np.mean(emg_results.snr_db)) if len(emg_results.snr_db) > 0 else 0.0,
        "apnea_events_count": len(emg_results.apnea_events),
        "r_peaks_detected": len(ecg_results.r_peaks),
        "mean_heart_rate_bpm": float(np.mean(ecg_results.instantaneous_hr)) if len(ecg_results.instantaneous_hr) > 0 else 0.0,
        "hrv_sdnn_ms": ecg_results.hrv_sdnn,
        "sync_pulses_count": len(sync_pulses.onsets),
        "opto_pulses_count": len(opto_pulses.onsets),
        "neuropixel_units_count": len(ephys_data.unit_ids) if ephys_data else 0,
        "diagnostic_plots": {k: str(v) for k, v in generated_plots.items()}
    }

    journal_path = write_run_journal(config, exec_summary)
    print(f"\n [SUCCESS] Analysis Pipeline Completed Successfully for {config.animal_id}!")
    return session_results
