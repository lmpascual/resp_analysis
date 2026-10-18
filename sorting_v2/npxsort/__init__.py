"""npxsort: Neuropixels spike sorting with Kilosort 4 + Lupin, real quality metrics and bombcell labels."""
from .config import load_config, ConfigError
from .pipeline import run_pipeline, STAGES

__all__ = ["load_config", "run_pipeline", "ConfigError", "STAGES"]
