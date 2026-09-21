"""
Neuropixel ephys data loader.
Reads ALF format spike sorting results (spikes.times.npy, spikes.clusters.npy) or synthetic ephys data.
"""
from pathlib import Path
import numpy as np
import logging
from typing import Optional, Dict
from src.pipeline.models import EphysData

logger = logging.getLogger(__name__)

def read_alf_spikes(session_path: Path) -> Optional[EphysData]:
    """
    Read ALF formatted spike sorting outputs from session_path/alf directory.
    Expects spikes.times.npy, spikes.clusters.npy, clusters.depths.npy.
    """
    session_path = Path(session_path)
    alf_dir = session_path / "alf"
    if not alf_dir.exists():
        alf_dir = session_path

    spikes_times_fn = alf_dir / "spikes.times.npy"
    spikes_clusters_fn = alf_dir / "spikes.clusters.npy"

    if not (spikes_times_fn.exists() and spikes_clusters_fn.exists()):
        logger.info(f"No ALF spike files found at {alf_dir}")
        return None

    spike_times = np.load(spikes_times_fn)
    spike_clusters = np.load(spikes_clusters_fn)

    unit_ids = np.unique(spike_clusters)
    unit_spiketimes = {}
    unit_depths = {}
    unit_quality = {}

    depths_fn = alf_dir / "clusters.depths.npy"
    cluster_depths = np.load(depths_fn) if depths_fn.exists() else None

    for uid in unit_ids:
        unit_spiketimes[int(uid)] = spike_times[spike_clusters == uid]
        if cluster_depths is not None and uid < len(cluster_depths):
            unit_depths[int(uid)] = float(cluster_depths[uid])
        else:
            unit_depths[int(uid)] = float(uid * 50.0)
        unit_quality[int(uid)] = "good"

    return EphysData(
        unit_ids=unit_ids,
        unit_spiketimes=unit_spiketimes,
        unit_depths=unit_depths,
        unit_quality=unit_quality
    )

def generate_synthetic_ephys(duration_s: float = 60.0, num_units: int = 12) -> EphysData:
    """
    Generate synthetic Neuropixel units with varied firing rates and probe depths.
    """
    unit_ids = np.arange(num_units)
    unit_spiketimes = {}
    unit_depths = {}
    unit_quality = {}

    for uid in unit_ids:
        # Firing rates between 2 Hz and 25 Hz
        rate = np.random.uniform(2.0, 25.0)
        n_spikes = int(duration_s * rate)
        spikes = np.sort(np.random.uniform(0, duration_s, n_spikes))
        
        unit_spiketimes[int(uid)] = spikes
        unit_depths[int(uid)] = float(200.0 + uid * 250.0)  # µm along probe
        unit_quality[int(uid)] = "good" if uid % 2 == 0 else "mua"

    return EphysData(
        unit_ids=unit_ids,
        unit_spiketimes=unit_spiketimes,
        unit_depths=unit_depths,
        unit_quality=unit_quality
    )
