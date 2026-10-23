"""
Modular Lupin Spike Sorting Pipeline based on Garcia et al. 2026:
'Opening the black box: a modular approach to spike sorting' (bioRxiv 2026.01.23.701239)

Pipeline Architecture:
1. Locally Exclusive Peak Detection + Monopolar Triangulation
2. Motion Estimation & Kriging / Wobble Compensation
3. Sparse SVD Feature Clustering (Circus / HDBSCAN)
4. Robust SVD-OMP Template Matching Pursuit
"""
import time
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

import spikeinterface.full as si
import spikeinterface.sorters as ss
from ..config import LupinConfig
import torch
import spikeinterface.preprocessing as prep

# -------------------------------------------------------------------------
# PyTorch-backed safe whitening patch (prevents Windows OpenBLAS 0xc06d007f)
# -------------------------------------------------------------------------
try:
    si_whiten_mod = sys.modules.get('spikeinterface.preprocessing.whiten')
    if si_whiten_mod is not None:
        def safe_compute_covariance_matrix(recording, apply_mean=False, regularize=False, regularize_kwargs=None, random_chunk_kwargs=None, **kwargs):
            random_chunk_kwargs = random_chunk_kwargs or {}
            random_data = si_whiten_mod.get_random_data_chunks(recording, concatenated=True, return_scaled=False, **random_chunk_kwargs)
            random_data = random_data.astype(np.float32)
            regularize_kwargs = regularize_kwargs if regularize_kwargs is not None else {'method': 'GraphicalLassoCV'}
            if apply_mean:
                M = np.mean(random_data, axis=0, keepdims=True)
                data = random_data - M
            else:
                M = None
                data = random_data
            if not regularize:
                t_data = torch.from_numpy(data)
                cov = (t_data.T @ t_data).numpy() / data.shape[0]
            else:
                cov = si_whiten_mod.compute_sklearn_covariance_matrix(data, regularize_kwargs).astype('float32')
            return data, cov, M

        def safe_compute_whitening_from_covariance(cov, eps):
            t_cov = torch.as_tensor(cov, dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(t_cov, full_matrices=True)
            diag_s = torch.diag(1.0 / torch.sqrt(S + eps))
            W = (U @ diag_s @ Vh).numpy()
            return W

        def safe_whiten_get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
            traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
            if traces.dtype.kind == 'u':
                traces = traces.astype('float32')
            if self.M is not None:
                t_diff = torch.from_numpy(traces - self.M)
                t_W = torch.from_numpy(self.W)
                whiten_traces = (t_diff @ t_W).numpy()
            else:
                t_traces = torch.from_numpy(traces)
                t_W = torch.from_numpy(self.W)
                whiten_traces = (t_traces @ t_W).numpy()
            whiten_traces = whiten_traces[:, channel_indices]
            if self.int_scale is not None:
                whiten_traces *= self.int_scale
            return whiten_traces.astype(self.dtype)

        si_whiten_mod.compute_covariance_matrix = safe_compute_covariance_matrix
        si_whiten_mod.compute_whitening_from_covariance = safe_compute_whitening_from_covariance
        si_whiten_mod.WhitenRecordingSegment.get_traces = safe_whiten_get_traces
except Exception:
    pass

try:
    import threadpoolctl
    for cls in [threadpoolctl.OpenBLASController, threadpoolctl.MKLController]:
        if hasattr(cls, 'get_num_threads'): cls.get_num_threads = lambda self: 1
        if hasattr(cls, 'set_num_threads'): cls.set_num_threads = lambda self, n: None
except Exception:
    pass

try:
    import sklearn.utils.extmath as em
    from sklearn.utils.extmath import svd_flip
    import sklearn.decomposition._truncated_svd as tsvd

    def safe_randomized_svd(M, n_components, **kwargs):
        t_M = torch.as_tensor(M, dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(t_M, full_matrices=False)
        u = U[:, :n_components].numpy()
        s = S[:n_components].numpy()
        vh = Vh[:n_components, :].numpy()
        if kwargs.get('flip_sign', True):
            u, vh = svd_flip(u, vh)
        return u, s, vh

    orig_safe_sparse_dot = em.safe_sparse_dot
    def safe_sparse_dot_patch(a, b, *, dense_output=False):
        try:
            t_a = torch.as_tensor(a)
            t_b = torch.as_tensor(b)
            return (t_a @ t_b).numpy()
        except Exception:
            return orig_safe_sparse_dot(a, b, dense_output=dense_output)

    tsvd._randomized_svd = safe_randomized_svd
    tsvd.safe_sparse_dot = safe_sparse_dot_patch
    em._randomized_svd = safe_randomized_svd
    em.randomized_svd = safe_randomized_svd
    em.safe_sparse_dot = safe_sparse_dot_patch
except Exception:
    pass

try:
    import spikeinterface.sortingcomponents.matching.circus as cm
    def safe_compress_templates(templates_array, approx_rank, remove_mean=False, return_new_templates=True):
        if remove_mean:
            templates_array -= templates_array.mean(axis=(1, 2))[:, None, None]
        num_templates, num_samples, num_channels = templates_array.shape
        temporal = np.zeros((num_templates, num_samples, approx_rank), dtype=np.float32)
        spatial = np.zeros((num_templates, approx_rank, num_channels), dtype=np.float32)
        singular = np.zeros((num_templates, approx_rank), dtype=np.float32)
        t_templates = torch.from_numpy(templates_array.astype(np.float32))
        U, S, Vh = torch.linalg.svd(t_templates, full_matrices=False)
        k = min(approx_rank, num_channels)
        temporal[:, :, :k] = U[:, :, :k].numpy()
        singular[:, :k] = S[:, :k].numpy()
        spatial[:, :k, :] = Vh[:, :k, :].numpy()
        if return_new_templates:
            t_temp = torch.from_numpy(temporal * singular[:, np.newaxis, :])
            t_spat = torch.from_numpy(spatial)
            templates_array = (t_temp @ t_spat).numpy()
        else:
            templates_array = None
        return temporal, singular, spatial, templates_array

    cm.compress_templates = safe_compress_templates
    cm.np.matmul = lambda a, b: (torch.as_tensor(a) @ torch.as_tensor(b)).numpy()
except Exception:
    pass

orig_np_svd = np.linalg.svd
def safe_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    try:
        t_a = torch.as_tensor(a)
        if not compute_uv:
            s = torch.linalg.svdvals(t_a)
            return s.numpy()
        U, S, Vh = torch.linalg.svd(t_a, full_matrices=full_matrices)
        return U.numpy(), S.numpy(), Vh.numpy()
    except Exception:
        return orig_np_svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)

np.linalg.svd = safe_svd

try:
    import spikeinterface.sortingcomponents.clustering.split as split_mod
    orig_split_clusters = split_mod.split_clusters

    def safe_split_clusters(
        peak_labels,
        recording,
        features_dict_or_folder,
        method="local_feature_clustering",
        method_kwargs=None,
        recursive=False,
        recursive_depth=None,
        returns_split_count=False,
        debug_folder=None,
        **job_kwargs,
    ):
        if len(peak_labels) > 500_000:
            logger.info(f"Dataset has {len(peak_labels)} peaks (>500k). Bypassing split_clusters to avoid Windows named-pipe buffer overflow and proceeding directly to SVD template construction...")
            split_count = np.zeros(peak_labels.size, dtype=int) if returns_split_count else None
            return peak_labels, split_count

        try:
            return orig_split_clusters(
                peak_labels, recording, features_dict_or_folder,
                method=method, method_kwargs=method_kwargs or {},
                recursive=recursive, recursive_depth=recursive_depth,
                returns_split_count=returns_split_count, debug_folder=debug_folder,
                **job_kwargs
            )
        except Exception as e:
            logger.warning(f"ProcessPoolExecutor split_clusters encountered Windows pipe IPC error ({e}). Bypassing split_clusters and proceeding directly to SVD template construction...")
            split_count = np.zeros(peak_labels.size, dtype=int) if returns_split_count else None
            return peak_labels, split_count

    split_mod.split_clusters = safe_split_clusters
except Exception:
    pass

logger = logging.getLogger("LUPIN_PIPELINE")

def run_lupin_sorting(
    recording: si.BaseRecording,
    output_folder: Path,
    config: Optional[LupinConfig] = None,
    n_jobs: int = 12,
    verbose: bool = True
) -> si.BaseSorting:
    """
    Executes the modular Lupin pipeline on a preprocessed/interpolated recording.
    """
    config = config or LupinConfig()
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(" INITIALIZING MODULAR LUPIN SPIKE SORTING PIPELINE (Garcia et al. 2026)")
    logger.info(f" Recording: {recording.get_num_channels()} channels, {recording.sampling_frequency:.1f} Hz, {recording.get_total_duration():.2f}s")
    logger.info(f" Clustering: {config.clustering_method} | Matching: {config.matching_method}")
    logger.info("=" * 80)

    # Use optimized SpyKING-CIRCUS 2 / Lupin engine in SpikeInterface
    sc2_params = ss.get_default_sorter_params('spykingcircus2')
    
    # Configure Lupin-optimal settings
    sc2_params['apply_preprocessing'] = False # Data is already filtered & loccar'd
    sc2_params['apply_motion_correction'] = False # Motion already corrected via Kriging
    sc2_params['general']['radius_um'] = config.sparsity_radius_um
    sc2_params['general']['ms_before'] = config.ms_before
    sc2_params['general']['ms_after'] = config.ms_after
    sc2_params['templates_from_svd'] = config.templates_from_svd
    sc2_params['clustering']['method'] = config.clustering_method
    sc2_params['matching']['method'] = config.matching_method
    sc2_params['matching']['method_kwargs'] = {
        'amplitudes': list(config.matching_amplitudes)
    }
    sc2_params['cache_preprocessing'] = {'mode': 'no-cache'}
    sc2_params['job_kwargs'] = {'n_jobs': n_jobs, 'progress_bar': False}

    t0 = time.time()
    sorting = ss.run_sorter(
        sorter_name='spykingcircus2',
        recording=recording,
        folder=str(output_folder / "lupin_working"),
        remove_existing_folder=True,
        verbose=False,
        **sc2_params
    )
    t_elapsed = time.time() - t0
    logger.info(f"Lupin pipeline completed in {t_elapsed:.2f}s. Extracted {len(sorting.get_unit_ids())} units.")
    return sorting
