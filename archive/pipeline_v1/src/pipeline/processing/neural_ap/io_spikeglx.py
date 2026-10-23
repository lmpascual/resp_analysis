"""
SpikeGLX stream isolation and robust raw/processed recording reader.
Avoids Neo stream enumeration errors when NIDAQ and IMEC probe streams co-occur.
"""
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import spikeinterface.full as si
import spikeinterface.extractors as se

logger = logging.getLogger("IO_SPIKEGLX")

def find_spikeglx_ap_files(dataset_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Scans the given folder or subfolders for .ap.bin and .ap.meta files.
    """
    dataset_path = Path(dataset_path)
    if dataset_path.is_file() and dataset_path.name.endswith(".ap.bin"):
        bin_file = dataset_path
        meta_file = dataset_path.with_suffix(".meta")
        if not meta_file.exists():
            meta_file = Path(str(dataset_path).replace(".ap.bin", ".ap.meta"))
        return bin_file, meta_file if meta_file.exists() else None

    # Search directly in folder
    bin_files = list(dataset_path.glob("*.ap.bin"))
    if bin_files:
        bin_file = bin_files[0]
        meta_files = list(dataset_path.glob("*.ap.meta"))
        meta_file = meta_files[0] if meta_files else None
        return bin_file, meta_file

    # Search in probe subfolders (e.g. *_imec0)
    bin_files = list(dataset_path.glob("*_imec*/*.ap.bin"))
    if bin_files:
        bin_file = bin_files[0]
        meta_file = bin_file.with_suffix(".meta")
        return bin_file, meta_file if meta_file.exists() else None

    # Search recursively
    bin_files = list(dataset_path.rglob("*.ap.bin"))
    if bin_files:
        bin_file = bin_files[0]
        meta_file = bin_file.with_suffix(".meta")
        return bin_file, meta_file if meta_file.exists() else None

    return None, None

def read_spikeglx_ap_safe(
    dataset_path: Path,
    stream_id: Optional[str] = None,
    load_sync_channel: bool = False
) -> si.BaseRecording:
    """
    Safely loads a SpikeGLX AP stream without tripping Neo stream ambiguity on co-occurring NIDAQ files.
    """
    dataset_path = Path(dataset_path)
    logger.info(f"Resolving SpikeGLX AP stream from: {dataset_path}")

    # Case 1: Pointing directly to an imec directory with ap.bin
    bin_file, meta_file = find_spikeglx_ap_files(dataset_path)
    if bin_file is not None and bin_file.exists():
        target_dir = bin_file.parent
        logger.info(f"Found AP binary at: {bin_file}")
        try:
            # Try read_spikeglx directly on the probe directory
            # If stream_id not specified, check stream name
            rec = se.read_spikeglx(
                folder_path=target_dir,
                stream_id=stream_id,
                load_sync_channel=load_sync_channel
            )
            logger.info(f"Loaded SpikeGLX recording: {rec.get_num_channels()} channels, {rec.sampling_frequency:.1f} Hz, {rec.get_total_duration():.2f}s")
            return rec
        except Exception as e:
            logger.warning(f"Standard se.read_spikeglx failed ({e}). Falling back to stream discovery...")

    # Case 2: Discover neo streams explicitly
    try:
        stream_names, stream_ids = si.get_neo_streams('spikeglx', str(dataset_path))
        logger.info(f"Available Neo streams: names={stream_names}, ids={stream_ids}")
        ap_stream = None
        for s_id, s_name in zip(stream_ids, stream_names):
            if "ap" in s_id.lower() or "ap" in s_name.lower():
                ap_stream = s_id
                break
        if ap_stream is None and len(stream_ids) > 0:
            ap_stream = stream_ids[0]
            
        rec = se.read_spikeglx(
            folder_path=dataset_path,
            stream_id=ap_stream,
            load_sync_channel=load_sync_channel
        )
        return rec
    except Exception as e2:
        logger.error(f"Failed to load via Neo streams: {e2}")

    # Case 3: Load directly via binary extractor if meta is available
    if bin_file is not None and meta_file is not None and meta_file.exists():
        logger.info("Attempting direct binary read from parsed SpikeGLX metadata...")
        from spikeinterface.extractors.neoextractors.spikeglx import SpikeGLXRecordingExtractor
        rec = SpikeGLXRecordingExtractor(
            folder_path=bin_file.parent,
            stream_id=stream_id,
            load_sync_channel=load_sync_channel
        )
        return rec

    raise FileNotFoundError(f"Could not locate or load valid SpikeGLX AP stream from {dataset_path}")
