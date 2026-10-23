"""npxsort: Neuropixels spike sorting with Kilosort 4 + Lupin, real quality metrics and bombcell labels."""
from .config import load_config, ConfigError

__all__ = ["load_config", "run_pipeline", "ConfigError", "STAGES"]


def __getattr__(name):
    # The pipeline imports SpikeInterface: load it only when asked for, so the light modules (npxsort.paths,
    # npxsort.spikefeatures) also import in the cibrrig analysis environment.
    if name in ("run_pipeline", "STAGES"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module 'npxsort' has no attribute {name!r}")
