"""
Unit and integration tests for the modular analysis pipeline.
"""
import pytest
from pathlib import Path
import numpy as np

from src.pipeline.config import PipelineConfig, EMGConfig, ECGConfig, SyncOptoConfig, EphysConfig
from src.pipeline.io.io_nidq import generate_synthetic_nidq
from src.pipeline.io.io_ephys import generate_synthetic_ephys
from src.pipeline.processors.proc_emg import process_emg
from src.pipeline.processors.proc_ecg import process_ecg
from src.pipeline.processors.proc_sync_opto import extract_pulse_events
from src.pipeline.processors.proc_ephys import compute_respiratory_modulation
from src.pipeline.visualizer import generate_diagnostic_dashboard
from src.pipeline.journaler import write_run_journal, checkpoint_approve_parameters
from src.pipeline.pipeline_runner import run_pipeline

def test_config_serialization(tmp_path):
    cfg = PipelineConfig(animal_id="test_mouse", output_dir=tmp_path)
    json_file = tmp_path / "test_config.json"
    cfg.save_json(json_file)
    assert json_file.exists()

    reloaded = PipelineConfig.from_dict(cfg.to_dict())
    assert reloaded.animal_id == "test_mouse"
    assert reloaded.emg.highpass_cutoff == 300.0

def test_synthetic_nidq_generation():
    analog = generate_synthetic_nidq(duration_s=10.0, fs=10000.0)
    assert len(analog.time_vector) == 100000
    assert len(analog.emg_raw) == 100000
    assert len(analog.ecg_raw) == 100000

def test_emg_processing():
    analog = generate_synthetic_nidq(duration_s=10.0, fs=10000.0)
    cfg = EMGConfig(fs=10000.0)
    res = process_emg(analog.emg_raw, analog.time_vector, cfg)
    assert len(res.filtered) == 100000
    assert len(res.envelope) == 100000
    assert len(res.snr_db) > 0

def test_ecg_processing():
    analog = generate_synthetic_nidq(duration_s=10.0, fs=10000.0)
    cfg = ECGConfig(fs=10000.0)
    res = process_ecg(analog.ecg_raw, analog.time_vector, cfg)
    assert len(res.filtered) == 100000
    assert len(res.r_peaks) > 0
    assert res.hrv_sdnn >= 0.0

def test_pulse_extraction():
    analog = generate_synthetic_nidq(duration_s=10.0, fs=10000.0)
    cfg = SyncOptoConfig(fs=10000.0)
    pulses = extract_pulse_events(analog.sync_raw, analog.time_vector, cfg)
    assert len(pulses.onsets) > 0

def test_pipeline_non_interactive(tmp_path):
    cfg = PipelineConfig(
        animal_id="test_animal",
        session_id="s1",
        raw_data_dir=tmp_path,
        output_dir=tmp_path / "output"
    )
    res = run_pipeline(cfg, interactive=False)
    assert res is not None
    assert (tmp_path / "output" / "test_animal_s1_provenance.md").exists()
    assert (tmp_path / "output" / "test_animal_s1_run_journal.json").exists()
    assert (tmp_path / "output" / "test_animal_s1_emg_stationarity.png").exists()
