"""
Configuration settings and dataclasses for the modular Neuropixels spike sorting,
motion correction, and cardiorespiratory analysis pipeline.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
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
    enabled: bool = True
    channel_index: int = 1  # Default to channel 1 (XA1;1)
    fs: float = 10000.0  # Hz
    highpass_cutoff: float = 5.0  # Hz
    lowpass_cutoff: float = 100.0  # Hz
    notch_freq: float = 60.0  # Hz
    r_peak_threshold_sd: float = 3.0
    min_rr_interval_ms: float = 80.0  # ms for mouse/rat heart rates

@dataclass
class SyncOptoConfig:
    sync_channel_index: int = 7  # Default sync channel index (XA7;7)
    opto_channel_index: int = 6  # Default opto laser channel index (XA6;6)
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
class CatGTConfig:
    enabled: bool = True
    catgt_path: str = r"C:\Users\lpascu\SpikeGLX\CatGT-win\CatGT.exe"
    run_name: Optional[str] = None
    gate: int = 1
    trigger: int = 0
    probe: int = 0
    t_shift: bool = True  # SpikeGLX ADC channel time shift correction
    apfilter: str = "butter,12,300,9000"  # Bandpass 300-9000 Hz, order 12
    loccar_um: Tuple[float, float] = (75.0, 300.0)  # Exclude 75um, include 300um
    start_secs: Optional[float] = None
    max_secs: Optional[float] = None
    dest_dir: Optional[str] = None

@dataclass
class MotionConfig:
    enabled: bool = True
    preset: str = "kilosort_like"  # 'kilosort_like', 'dredge', 'nonrigid_accurate', 'nonrigid_fast_and_accurate'
    peak_detect_method: str = "locally_exclusive"  # 'locally_exclusive' or 'locally_exclusive_torch'
    peak_detect_threshold: float = 5.5  # SD threshold or direct amplitude
    peak_detect_radius_um: float = 100.0
    peak_localize_method: str = "monopolar_triangulation"  # 'monopolar_triangulation', 'grid_convolution', 'center_of_mass'
    estimate_motion_method: str = "kilosort_like"  # 'kilosort_like', 'decentralized', 'iterative_template'
    bin_duration_s: float = 2.0  # 2.0s or 4.0s for motion temporal bins
    spatial_bin_um: float = 50.0  # Spatial resolution in um
    interpolate_method: str = "kriging"  # 'kriging', 'nearest'
    scratch_dir: str = r"C:\SCRATCH_neuropixels"

@dataclass
class LupinConfig:
    enabled: bool = True
    sparsity_radius_um: float = 100.0
    clustering_method: str = "circus"  # 'circus', 'hdbscan'
    matching_method: str = "circus-omp-svd"  # 'circus-omp-svd', 'wobble'
    ms_before: float = 1.5
    ms_after: float = 2.0
    templates_from_svd: bool = True
    matching_amplitudes: List[float] = field(default_factory=lambda: [0.6, 2.0])

@dataclass
class SlidingWindowConfig:
    enabled: bool = True
    channel_range: Tuple[int, int] = (0, 384)  # min_channel, max_channel ROI (defaults to all 384)
    window_size: int = 50  # Number of channels per window
    overlap: int = 20  # Overlap channels between adjacent windows (e.g. 20)
    agreement_threshold: float = 0.50  # Minimum spike match coincidence agreement (e.g. 0.50 or 0.90)
    min_spikes_to_evaluate: int = 30

@dataclass
class SorterConfig:
    enabled_sorters: List[str] = field(default_factory=lambda: ["lupin"])
    do_CAR: bool = False  # Set False if CAR / loccar was applied in CatGT or Preproc
    save_phy: bool = True
    export_alf: bool = True
    isi_threshold_s: float = 0.0010
    qc_isi_cutoff: float = 0.10
    qc_presence_ratio: float = 0.90
    qc_amp_cutoff: float = 0.10
    qc_min_fr_hz: float = 0.10
    n_jobs: int = 12

@dataclass
class PipelineConfig:
    animal_id: str = "subject_01"
    session_id: str = "session_01"
    raw_data_dir: Path = Path("./raw_data")
    output_dir: Path = Path("./output")
    interactive_checkpoint: bool = True
    duration_s: Optional[float] = None
    start_time_s: float = 0.0
    emg: EMGConfig = field(default_factory=EMGConfig)
    ecg: ECGConfig = field(default_factory=ECGConfig)
    sync_opto: SyncOptoConfig = field(default_factory=SyncOptoConfig)
    ephys: EphysConfig = field(default_factory=EphysConfig)
    catgt: CatGTConfig = field(default_factory=CatGTConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    sorters: SorterConfig = field(default_factory=SorterConfig)
    lupin: LupinConfig = field(default_factory=LupinConfig)

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
        if "config" in data:
            data = data["config"]
        raw_dir = Path(data.get('raw_data_dir', './raw_data'))
        out_dir = Path(data.get('output_dir', './output'))
        emg_cfg = EMGConfig(**data.get('emg', {}))
        ecg_cfg = ECGConfig(**data.get('ecg', {}))
        sync_cfg = SyncOptoConfig(**data.get('sync_opto', {}))
        ephys_cfg = EphysConfig(**data.get('ephys', {}))
        catgt_cfg = CatGTConfig(**data.get('catgt', {}))
        motion_cfg = MotionConfig(**data.get('motion', {}))
        sliding_cfg = SlidingWindowConfig(**data.get('sliding_window', {}))
        sorters_cfg = SorterConfig(**data.get('sorters', {}))
        lupin_cfg = LupinConfig(**data.get('lupin', {}))
        return cls(
            animal_id=data.get('animal_id', 'subject_01'),
            session_id=data.get('session_id', 'session_01'),
            raw_data_dir=raw_dir,
            output_dir=out_dir,
            interactive_checkpoint=data.get('interactive_checkpoint', True),
            duration_s=data.get('duration_s', None),
            start_time_s=data.get('start_time_s', 0.0),
            emg=emg_cfg,
            ecg=ecg_cfg,
            sync_opto=sync_cfg,
            ephys=ephys_cfg,
            catgt=catgt_cfg,
            motion=motion_cfg,
            sliding_window=sliding_cfg,
            sorters=sorters_cfg,
            lupin=lupin_cfg
        )
