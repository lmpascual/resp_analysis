#!/usr/bin/env python3
"""
CLI Runner: Interactive Phy-Like Cluster Validation & Quality Assurance Suite.

Generates:
1. Unified Multi-Cluster Browser: <dest_dir>/cluster_validation_browser.html
2. Individual Unit HTML Deep-Dives: <dest_dir>/units/unit_xxxx.html
3. Detailed Provenance Logging in RUN_JOURNAL.md

Usage:
  python scripts/visualization/generate_phy_like_cluster_validator.py \
      --output-dir outputs/LP21/LP21_fent_EXPT_g1_60s_120s_lupin_run \
      --ap-bin "Z:/ramirezlab/pascual/rec_data/LP21_fent_EXPT_g1/LP21_fent_EXPT_g1_imec0/LP21_fent_EXPT_g1_t0.imec0.ap.bin" \
      --nidq-bin "Z:/ramirezlab/pascual/rec_data/LP21_fent_EXPT_g1/LP21_fent_EXPT_g1_t0.nidq.bin" \
      --insp-offset-source flow \
      --n-channels 8 \
      --max-units 30
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.io.io_nidq import read_nidq_bin
from src.pipeline.config import EMGConfig
from src.pipeline.diagnostics.cluster_validator import (
    UnitValidationBundle,
    detect_respiratory_landmarks,
    compute_respiratory_phase_vector,
    compute_isi_and_violations,
    compute_amplitude_stationarity,
    compute_time_resolved_isi_pdfs,
    compute_autocorrelogram,
    compute_amplitude_cutoff,
    find_unit_channel_footprint,
    compute_waveform_kinetics,
    extract_representative_trace_snippet
)
from src.pipeline.diagnostics.cluster_html_report import (
    generate_single_unit_html,
    generate_multi_cluster_browser_html
)


def parse_geom_depths_from_meta(meta_path: Optional[Path]) -> Dict[int, float]:
    """
    Parses channel depth (y coordinate in um) from Neuropixels metadata.
    Defaults to standard NP2004 staggered layout (y = (ch // 2) * 15 um).
    """
    depths = {}
    if meta_path is not None and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("~snsGeomMap="):
                        parts = line.strip().split(")(")
                        for ch_idx, p in enumerate(parts[1:]):
                            p_clean = p.rstrip(")")
                            tokens = p_clean.split(":")
                            if len(tokens) >= 3:
                                y_um = float(tokens[2])
                                depths[ch_idx] = y_um
                        break
        except Exception:
            pass

    for ch in range(385):
        if ch not in depths:
            depths[ch] = float((ch // 2) * 15.0)
    return depths


def main():
    parser = argparse.ArgumentParser(
        description="Generate Interactive Phy-Like Validation Suite for Spike-Sorted Clusters."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Path to sorting output directory containing alf/ and cluster_info.tsv."
    )
    parser.add_argument(
        "--ap-bin", type=Path, required=True,
        help="Path to raw or filtered AP binary (.ap.bin)."
    )
    parser.add_argument(
        "--nidq-bin", type=Path, required=True,
        help="Path to raw NIDQ binary (.nidq.bin)."
    )
    parser.add_argument(
        "--meta-path", type=Path, default=None,
        help="Path to AP .meta file (auto-detected if omitted)."
    )
    parser.add_argument(
        "--insp-offset-source", type=str, choices=["flow", "emg"], default="flow",
        help="Landmark for inspiration offset: 'flow' (default: pneumotach ZC) or 'emg' (diaphragm EMG drop)."
    )
    parser.add_argument(
        "--n-channels", type=int, default=8,
        help="Number of footprint channels to extract (default: 8)."
    )
    parser.add_argument(
        "--max-units", type=int, default=30,
        help="Maximum units to process (default: 30, sorted by firing rate)."
    )
    parser.add_argument(
        "--dest-dir", type=Path, default=None,
        help="Destination directory for validation dashboards."
    )
    parser.add_argument(
        "--time-window", type=float, default=1.5,
        help="Duration of representative multi-channel trace snippet (s, default: 1.5s)."
    )
    parser.add_argument(
        "--individual-pages", action="store_true", default=True,
        help="Whether to also generate individual unit HTML pages."
    )

    args = parser.parse_args()

    out_dir = args.output_dir
    dest_dir = args.dest_dir or (out_dir / "sorting_validation")
    dest_dir.mkdir(parents=True, exist_ok=True)
    units_dir = dest_dir / "units"
    if args.individual_pages:
        units_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" INTERACTIVE SPIKE SORTING VALIDATION SUITE")
    print(f" Source Directory: {out_dir}")
    print(f" Output Directory: {dest_dir}")
    print(f" Inspiration Offset Landmark: {args.insp_offset_source.upper()}")
    print("=" * 80)

    # 1. Locate ALF arrays & cluster_info
    alf_dir = out_dir / "00_spike_sorting_data" / "alf"
    if not alf_dir.exists():
        alf_dir = out_dir / "alf"

    spike_times_path = alf_dir / "spike_times.npy"
    spike_clusters_path = alf_dir / "spike_clusters.npy"
    if not (spike_times_path.exists() and spike_clusters_path.exists()):
        print(f"Error: Missing spike_times.npy or spike_clusters.npy in {alf_dir}")
        sys.exit(1)

    spk_times = np.load(spike_times_path)
    spk_clusts = np.load(spike_clusters_path)

    tsv_path = out_dir / "00_spike_sorting_data" / "cluster_info.tsv"
    if not tsv_path.exists():
        tsv_path = out_dir / "cluster_info.tsv"
    
    if tsv_path.exists():
        df_clusters = pd.read_csv(tsv_path, sep="	")
    else:
        unique_uids = np.unique(spk_clusts)
        df_clusters = pd.DataFrame({
            "cluster_id": unique_uids,
            "firing_rate_hz": [np.sum(spk_clusts == u) / max(1.0, float(np.ptp(spk_times))) for u in unique_uids],
            "n_spikes": [np.sum(spk_clusts == u) for u in unique_uids],
            "depth_um": [float(u * 15.0) for u in unique_uids],
            "passed_qc": [True for _ in unique_uids]
        })

    if "firing_rate_hz" in df_clusters.columns:
        df_clusters = df_clusters.sort_values(by="firing_rate_hz", ascending=False).reset_index(drop=True)
    elif "firing_rate" in df_clusters.columns:
        df_clusters = df_clusters.sort_values(by="firing_rate", ascending=False).reset_index(drop=True)

    selected_df = df_clusters.head(args.max_units).copy()
    print(f"Loaded {len(df_clusters)} total clusters. Processing top {len(selected_df)} units.")

    # 2. Parse Channel Geometry & Meta
    meta_path = args.meta_path
    if meta_path is None:
        cand_meta = args.ap_bin.with_suffix(".meta")
        if cand_meta.exists():
            meta_path = cand_meta
    channel_depths = parse_geom_depths_from_meta(meta_path)

    # 3. Load NIDQ Analog Signals & Detect Respiratory Landmarks
    nidq_meta = args.nidq_bin.with_suffix(".meta")
    nidq_data = read_nidq_bin(args.nidq_bin, nidq_meta)
    t_nidq = nidq_data.time_vector
    fs_nidq = nidq_data.fs

    emg_raw = nidq_data.all_channels[0]
    flow_raw = nidq_data.all_channels[5] if len(nidq_data.all_channels) > 5 else None

    t_min_spk = float(np.min(spk_times))
    t_max_spk = float(np.max(spk_times))
    time_offset_s = 0.0
    cfg_snapshot = out_dir / "00_spike_sorting_data" / "pipeline_config_snapshot.json"
    if cfg_snapshot.exists():
        import json
        with open(cfg_snapshot, "r") as f:
            cfg_dict = json.load(f)
            time_offset_s = float(cfg_dict.get("start_time_s", 0.0))

    print(f"Spike train relative time: {t_min_spk:.2f}s to {t_max_spk:.2f}s (offset: +{time_offset_s:.1f}s)")

    nidq_window_start = time_offset_s + t_min_spk
    nidq_window_end = time_offset_s + t_max_spk + 2.0
    nidq_mask = (t_nidq >= nidq_window_start) & (t_nidq <= nidq_window_end)
    if np.sum(nidq_mask) > 1000:
        t_nidq_w = t_nidq[nidq_mask] - time_offset_s
        emg_raw_w = emg_raw[nidq_mask]
        flow_raw_w = flow_raw[nidq_mask] if flow_raw is not None else None
    else:
        t_nidq_w = t_nidq
        emg_raw_w = emg_raw
        flow_raw_w = flow_raw

    breath_onsets, breath_offsets, landmark_prov, _ = detect_respiratory_landmarks(
        emg_raw=emg_raw_w,
        flow_raw=flow_raw_w,
        t_nidq=t_nidq_w,
        fs_nidq=fs_nidq,
        offset_source=args.insp_offset_source
    )
    print(f"Detected {len(breath_onsets)} breaths.")
    print(f"Landmark Provenance: {landmark_prov}")

    # 4. Memory-Map Raw AP Binary
    fs_ap = 30000.0
    n_chans_ap = 385
    ap_file_bytes = args.ap_bin.stat().st_size
    n_ap_samples = ap_file_bytes // (2 * n_chans_ap)
    mm_ap = np.memmap(args.ap_bin, dtype=np.int16, mode="r", shape=(n_ap_samples, n_chans_ap))

    # 5. Process Each Cluster
    bundles = []
    t0_start = time.time()

    for idx, row in selected_df.iterrows():
        uid = int(row["cluster_id"])
        fr = float(row.get("firing_rate_hz", row.get("firing_rate", 1.0)))
        n_spk = int(row.get("n_spikes", 0))
        shank = int(row.get("shank", 0))
        passed_qc = bool(row.get("passed_qc", row.get("passed_allen_qc", True)))

        u_spk_times = np.sort(spk_times[spk_clusts == uid])
        if len(u_spk_times) < 5:
            continue

        u_abs_times = time_offset_s + u_spk_times

        # TRUE global peak channel discovery across all 384 channels
        peak_chan, footprint_chans, channel_ptps, template_wfs = find_unit_channel_footprint(
            mm_ap=mm_ap,
            spike_times_s=u_abs_times,
            fs=fs_ap,
            candidate_channels=None,  # Scan all 384 channels!
            n_footprint_channels=args.n_channels,
            max_sample_spikes=100
        )

        true_depth_um = float(channel_depths.get(peak_chan, (peak_chan // 2) * 15.0))

        # Extract representative multi-channel trace snippet (1.5s native 30 kHz burst snippet, zero downsampling)
        # If time_window <= 5.0s, pass t_start_s=None to automatically find the highest firing rate burst window!
        snip_t_start = None if args.time_window <= 5.0 else time_offset_s
        snippet = extract_representative_trace_snippet(
            mm_ap=mm_ap,
            channels=footprint_chans,
            channel_depths=channel_depths,
            spike_times_s=u_abs_times,
            peak_channel=peak_chan,
            t_start_s=snip_t_start,
            window_duration_s=args.time_window,
            fs=fs_ap,
            refractory_ms=1.0,
            max_display_pts_per_chan=100000
        )
        # Shift snippet time axis to relative coordinates for display
        snippet.t_axis_s -= time_offset_s
        snippet.spike_times_in_window -= time_offset_s
        snippet.aligned_trough_times -= time_offset_s
        snippet.t_start_s -= time_offset_s
        snippet.t_end_s -= time_offset_s
        if len(snippet.violation_trough_times) > 0:
            snippet.violation_trough_times -= time_offset_s

        # Compute respiratory phase vector
        rp_res = compute_respiratory_phase_vector(
            spike_times_s=u_spk_times,
            breath_onsets=breath_onsets,
            breath_offsets=breath_offsets,
            landmark_provenance=landmark_prov
        )

        # Compute ISI distribution
        isi_res = compute_isi_and_violations(
            spike_times_s=u_spk_times,
            refractory_ms=1.0
        )

        # Estimate spike amplitudes on TRUE peak channel
        peak_sample_idxs = (u_abs_times * fs_ap).astype(np.int64)
        valid_p_idxs = peak_sample_idxs[(peak_sample_idxs >= 15) & (peak_sample_idxs < n_ap_samples - 30)]
        if len(valid_p_idxs) > 0:
            samp_subset = valid_p_idxs[:min(2000, len(valid_p_idxs))]
            amps = [float(np.ptp(mm_ap[s - 15 : s + 30, peak_chan])) for s in samp_subset]
            spk_amps_uv = np.array(amps, dtype=np.float32)
            st_times = u_spk_times[:len(spk_amps_uv)]
        else:
            spk_amps_uv = np.array([50.0] * len(u_spk_times), dtype=np.float32)
            st_times = u_spk_times

        st_res = compute_amplitude_stationarity(st_times, spk_amps_uv)
        tr_isi_res = compute_time_resolved_isi_pdfs(u_spk_times, n_epochs=5)
        acg_res = compute_autocorrelogram(u_spk_times)
        cutoff_res = compute_amplitude_cutoff(spk_amps_uv)

        peak_template = template_wfs.get(peak_chan, np.zeros(75, dtype=np.float32))
        wf_res = compute_waveform_kinetics(
            template_wf=peak_template,
            peak_chan=peak_chan,
            footprint_chans=footprint_chans,
            channel_depths=channel_depths,
            all_template_wfs=template_wfs,
            fs=fs_ap,
            noise_sigma=12.0
        )

        bundle = UnitValidationBundle(
            cluster_id=uid,
            firing_rate_hz=fr,
            total_spikes=len(u_spk_times),
            depth_um=true_depth_um,
            shank=shank,
            peak_channel=peak_chan,
            footprint_channels=footprint_chans,
            passed_allen_qc=passed_qc,
            respiratory_phase=rp_res,
            isi=isi_res,
            stationarity=st_res,
            time_resolved_isi=tr_isi_res,
            acg=acg_res,
            waveform=wf_res,
            amplitude_cutoff=cutoff_res,
            trace_snippet=snippet
        )
        bundles.append(bundle)

        if args.individual_pages:
            unit_file = units_dir / f"unit_{uid:04d}_ch{peak_chan}.html"
            generate_single_unit_html(bundle, unit_file)

        print(f"[{len(bundles):02d}/{len(selected_df):02d}] Unit {uid:04d} | Peak Ch {peak_chan:03d} ({true_depth_um:.0f}um) | PTP: {channel_ptps.get(peak_chan, 0.0):.1f}uV | FR: {fr:.1f}Hz | Phase: {rp_res.subtype_class} (R={rp_res.mean_vector_r})")

    # 6. Render Unified Multi-Cluster Browser HTML
    browser_html_path = dest_dir / "cluster_validation_browser.html"
    generate_multi_cluster_browser_html(bundles, browser_html_path)

    elapsed = time.time() - t0_start
    print("-" * 80)
    print(f" Validation suite successfully regenerated for {len(bundles)} units in {elapsed:.1f}s.")
    print(f" Unified Browser: file:///{browser_html_path.resolve().as_posix()}")
    if args.individual_pages:
        print(f" Individual Unit Pages: file:///{units_dir.resolve().as_posix()}")
    print("-" * 80)

    # 7. Append Execution Record to RUN_JOURNAL.md
    journal_path = out_dir / "RUN_JOURNAL.md"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    journal_entry = f"""
## [{timestamp_str}] Phy-Like Single-Cluster Validation Suite Execution (V2 - Global Footprint & Trough Snapping)
- **Units Validated**: {len(bundles)} clusters
- **Peak Channel Discovery**: Global 384-channel scan (unbiased by local window depth)
- **Spike Alignment**: Exact negative trough snapping ($dt \le 0.5\,\text{{ms}}$) on peak physical contact
- **Respiratory Landmark Architecture**:
  * **Inspiration Onset ($0^\circ$)**: Diaphragm EMG Burst Onset (NIDQ Ch 0)
  * **Inspiration Offset ($180^\circ$)**: {landmark_prov}
- **Outputs**:
  * Unified Interactive Browser: `sorting_validation/cluster_validation_browser.html`
  * Individual Deep-Dive Pages: `sorting_validation/units/`
"""
    if journal_path.exists():
        with open(journal_path, "a", encoding="utf-8") as jf:
            jf.write(journal_entry)


if __name__ == "__main__":
    main()
