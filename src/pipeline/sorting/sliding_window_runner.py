"""
Sliding-Window Spike Sorting Engine
Runs spike sorting (Lupin, MountainSort 5, or Kilosort 4) across overlapping
channel sub-bands to eliminate full-probe over-splitting and template collisions.
"""
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import spikeinterface.full as si

from .lupin_pipeline import run_lupin_sorting
from .postprocessing import export_sorting_alf_and_qc
from ..config import PipelineConfig, SlidingWindowConfig

logger = logging.getLogger("SLIDING_WINDOW_RUNNER")

def compute_window_ranges(
    total_channels: int,
    channel_range: Tuple[int, int],
    window_size: int = 50,
    overlap: int = 20
) -> List[Dict[str, Any]]:
    """
    Computes list of overlapping channel sub-windows.
    Example: range=(0, 330), size=50, overlap=20 -> step=30
    [0:50], [30:80], [60:110], ..., [280:330]
    """
    min_ch, max_ch = channel_range
    min_ch = max(0, min_ch)
    max_ch = min(total_channels, max_ch)
    step = max(1, window_size - overlap)

    windows = []
    w_idx = 0
    start_ch = min_ch
    while start_ch < max_ch:
        end_ch = min(max_ch, start_ch + window_size)
        if end_ch - start_ch < 10 and len(windows) > 0:
            # If the last remaining window is too small, merge with previous
            break

        # Compute overlap with previous window
        prev_overlap_range = None
        if len(windows) > 0:
            prev_start = windows[-1]["start_ch"]
            prev_end = windows[-1]["end_ch"]
            overlap_start = max(prev_start, start_ch)
            overlap_end = min(prev_end, end_ch)
            if overlap_end > overlap_start:
                prev_overlap_range = (overlap_start, overlap_end)

        windows.append({
            "window_id": w_idx,
            "name": f"win_{w_idx:02d}_ch{start_ch}_{end_ch}",
            "start_ch": start_ch,
            "end_ch": end_ch,
            "num_channels": end_ch - start_ch,
            "prev_overlap_range": prev_overlap_range
        })

        if end_ch >= max_ch:
            break
        start_ch += step
        w_idx += 1

    return windows

def run_sliding_window_sorting(
    recording: si.BaseRecording,
    config: PipelineConfig,
    out_dir: Path,
    sorter_name: str = "lupin",
    scratch_dir: Optional[Path] = None,
    max_windows: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Executes sliding window spike sorting across the recording.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir or r"C:\SCRATCH_neuropixels_test\sliding_scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    sw_cfg = config.sliding_window
    total_ch = recording.get_num_channels()
    windows = compute_window_ranges(
        total_channels=total_ch,
        channel_range=sw_cfg.channel_range,
        window_size=sw_cfg.window_size,
        overlap=sw_cfg.overlap
    )

    if max_windows is not None:
        windows = windows[:max_windows]

    logger.info("=" * 80)
    logger.info(f" SLIDING-WINDOW SPIKE SORTING INITIALIZED ({sorter_name.upper()})")
    logger.info(f" Total Probe Channels: {total_ch} | ROI: {sw_cfg.channel_range[0]} to {sw_cfg.channel_range[1]}")
    logger.info(f" Window Size: {sw_cfg.window_size} channels | Overlap: {sw_cfg.overlap} channels | Step: {sw_cfg.window_size - sw_cfg.overlap} channels")
    logger.info(f" Total Slices to Sort: {len(windows)} sub-windows")
    logger.info("=" * 80)

    window_results = []
    all_channel_ids = list(recording.channel_ids)

    for win in windows:
        w_id = win["window_id"]
        w_name = win["name"]
        s_ch = win["start_ch"]
        e_ch = win["end_ch"]

        logger.info(f"\n[{w_id+1}/{len(windows)}] Sorting {w_name} (Channels {s_ch} to {e_ch-1})...")
        win_dir = out_dir / w_name
        win_dir.mkdir(parents=True, exist_ok=True)

        slice_ch_ids = all_channel_ids[s_ch:e_ch]
        rec_slice = recording.channel_slice(channel_ids=slice_ch_ids)

        # Save slice to fast scratch folder
        w_scratch = scratch_dir / f"rec_{w_name}"
        if w_scratch.exists():
            shutil.rmtree(w_scratch)
        rec_slice_saved = rec_slice.save(folder=w_scratch, n_jobs=4, chunk_duration="1s")

        t0 = time.time()
        sorting = None
        try:
            if sorter_name == "lupin":
                sorting = run_lupin_sorting(
                    recording=rec_slice_saved,
                    output_folder=win_dir / "si_output",
                    n_jobs=4,
                    verbose=False
                )
            elif sorter_name == "mountainsort5":
                sorting = si.run_sorter(
                    sorter_name="mountainsort5",
                    recording=rec_slice_saved,
                    folder=win_dir / "si_output",
                    scheme="2",
                    detect_threshold=5.0,
                    verbose=False
                )
            elif sorter_name == "kilosort4":
                ks_cfg = getattr(config, "kilosort4", getattr(config, "kilosort", None))
                th_u = getattr(ks_cfg, "Th_universal", 9) if ks_cfg else 9
                th_l = getattr(ks_cfg, "Th_learned", 8) if ks_cfg else 8
                do_car = getattr(ks_cfg, "do_CAR", config.sorters.do_CAR) if ks_cfg else config.sorters.do_CAR
                nblocks = getattr(ks_cfg, "nblocks", 1) if ks_cfg else 1
                batch_size = getattr(ks_cfg, "batch_size", 60000) if ks_cfg else 60000
                sorting = si.run_sorter(
                    sorter_name="kilosort4",
                    recording=rec_slice_saved,
                    folder=win_dir / "si_output",
                    Th_universal=th_u,
                    Th_learned=th_l,
                    do_CAR=do_car,
                    nblocks=nblocks,
                    batch_size=batch_size,
                    verbose=False
                )

            dur = time.time() - t0
            logger.info(f"[{w_name}] Sorting complete in {dur:.1f}s. Extracted {len(sorting.unit_ids)} raw clusters.")

            # Compute QC and Export ALF
            qc_df = export_sorting_alf_and_qc(
                sorting=sorting,
                fs=rec_slice_saved.get_sampling_frequency(),
                duration_s=rec_slice_saved.get_total_duration(),
                out_dir=win_dir,
                sorter_name=f"{sorter_name}_{w_name}",
                recording=rec_slice_saved
            )

            # Cleanup slice scratch
            try:
                shutil.rmtree(w_scratch)
            except Exception:
                pass

            n_sua = int(np.sum(qc_df["passed_allen_qc"]))
            logger.info(f"[{w_name}] {len(qc_df)} Total Clusters -> {n_sua} Single Units (SUA)")

            window_results.append({
                "window": win,
                "sorting": sorting,
                "qc_df": qc_df,
                "win_dir": win_dir,
                "n_clusters": len(qc_df),
                "n_sua": n_sua,
                "duration_s": dur
            })

        except Exception as e:
            logger.error(f"Error sorting {w_name}: {e}", exc_info=True)

    logger.info(f"\n[SLIDING WINDOW SORTING COMPLETE] Finished {len(window_results)}/{len(windows)} windows.")
    return window_results
