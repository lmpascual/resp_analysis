"""
Configuration settings and dataclasses for the analysis pipeline.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

@dataclass
class EMGConfig:
    channel_index: int = 0
    fs: float = 10000.0  # Hz
    highpass_cutoff: float = 300.0  # Hz
    lowpass_cutoff: float = 3000.0  # Hz
    notch_freq: float = 60.0  # Hz
    rectify: bool = True
    smoothing_window_ms: float = 50.0  # ms
    peak_threshold_sd: float = 2.5
    min_breath_duration_ms: float = 100.0  # ms
    min_inter_breath_interval_ms: float = 150.0  # ms
    apnea_threshold_s: float = 2.0  # seconds

@dataclass
class ECGConfig:
    channel_index: int = 1
    fs: float = 10000.0  # Hz
    highpass_cutoff: float = 5.0  # Hz
    lowpass_cutoff: float = 100.0  # Hz
    notch_freq: float = 60.0  # Hz
    r_peak_threshold_sd: float = 3.0
    min_rr_interval_ms: float = 80.0  # ms for mouse/rat heart rates

@dataclass
class SyncOptoConfig:
    sync_channel_index: int = 2
    opto_channel_index: int = 3
    fs: float = 10000.0  # Hz
    pulse_threshold_v: float = 2.5  # Volts for digital TTL detection
    min_pulse_width_ms: float = 1.0  # ms

@dataclass
class EphysConfig:
    fs_ap: float = 30000.0  # Hz
    bin_size_ms: float = 20.0  # ms for PSTH/rasters
    depth_bin_um: float = 100.0  # um for depth heatmaps
    min_firing_rate_hz: float = 0.1

@dataclass
class PipelineConfig:
    animal_id: str = "subject_01"
    session_id: str = "session_01"
    raw_data_dir: Path = Path("./raw_data")
    output_dir: Path = Path("./output")
    interactive_checkpoint: bool = True
    emg: EMGConfig = field(default_factory=EMGConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    sync_opto: SyncOptoConfig = field(default_factory=SyncOptoConfig)
    ephys: EphysConfig = field(default_factory=EphysConfig)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['raw_data_dir'] = str(self.raw_data_dir)
        d['output_dir'] = str(self.output_dir)
        return d

    def save_json(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        raw_dir = Path(data.get('raw_data_dir', './raw_data'))
        out_dir = Path(data.get('output_dir', './output'))
        emg_cfg = EMGConfig(**data.get('emg', {}))
        ecg_cfg = ECGConfig(**data.get('ecg', {}))
        sync_cfg = SyncOptoConfig(**data.get('sync_opto', {}))
        ephys_cfg = EphysConfig(**data.get('ephys', {}))
        return cls(
            animal_id=data.get('animal_id', 'subject_01'),
            session_id=data.get('session_id', 'session_01'),
            raw_data_dir=raw_dir,
            output_dir=out_dir,
            interactive_checkpoint=data.get('interactive_checkpoint', True),
            emg=emg_cfg,
            ecg=ecg_cfg,
            sync_opto=sync_cfg,
            ephys=ephys_cfg
        )
