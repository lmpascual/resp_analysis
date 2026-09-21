"""
Cross-Sorter Unit Matching & Agreement Comparison Matrix:
Computes bipartite spike coincidence matching, Jaccard agreement, and consensus
across Kilosort 4, MountainSort 5, SpyKING-CIRCUS 2, Tridesclous 2, and Lupin.
"""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger("SORTER_COMPARATOR")

def compute_spike_coincidences(
    spikes1: np.ndarray,
    spikes2: np.ndarray,
    delta_s: float = 0.0015
) -> Tuple[int, float, float, float]:
    """Computes bipartite spike coincidences with tolerance delta_s."""
    if len(spikes1) == 0 or len(spikes2) == 0:
        return 0, 0.0, 0.0, 0.0
    idx = np.searchsorted(spikes2, spikes1)
    n_coinc = 0
    used2 = set()
    for i, t in enumerate(spikes1):
        c_idx = idx[i]
        best_cand = None
        min_dt = delta_s
        for c in (c_idx - 1, c_idx, c_idx + 1):
            if 0 <= c < len(spikes2) and c not in used2:
                dt = abs(spikes2[c] - t)
                if dt < min_dt:
                    min_dt = dt
                    best_cand = c
        if best_cand is not None:
            n_coinc += 1
            used2.add(best_cand)
    n1 = len(spikes1)
    n2 = len(spikes2)
    jaccard = n_coinc / (n1 + n2 - n_coinc) if (n1 + n2 - n_coinc) > 0 else 0.0
    recall1 = n_coinc / n1 if n1 > 0 else 0.0
    precision2 = n_coinc / n2 if n2 > 0 else 0.0
    return n_coinc, jaccard, recall1, precision2

def match_two_sorters(
    times1: np.ndarray,
    clusters1: np.ndarray,
    times2: np.ndarray,
    clusters2: np.ndarray,
    sorter1_name: str = "Sorter1",
    sorter2_name: str = "Sorter2",
    delta_s: float = 0.0015
) -> pd.DataFrame:
    """
    Computes best unit matches and Jaccard agreement between two sorters.
    """
    u1_ids = np.unique(clusters1)
    u2_ids = np.unique(clusters2)

    spks1_dict = {u: times1[clusters1 == u] for u in u1_ids}
    spks2_dict = {u: times2[clusters2 == u] for u in u2_ids}

    records = []
    for u1 in u1_ids:
        s1 = spks1_dict[u1]
        best_u2 = None
        best_jacc = 0.0
        best_coinc = 0
        best_rec = 0.0
        best_prec = 0.0

        for u2 in u2_ids:
            s2 = spks2_dict[u2]
            n_coinc, jacc, rec, prec = compute_spike_coincidences(s1, s2, delta_s=delta_s)
            if jacc > best_jacc:
                best_jacc = jacc
                best_u2 = u2
                best_coinc = n_coinc
                best_rec = rec
                best_prec = prec

        if best_jacc >= 0.50:
            m_class = "High-Confidence Match"
        elif best_jacc >= 0.30 or best_rec >= 0.60:
            m_class = "Moderate Match"
        elif best_jacc >= 0.15:
            m_class = "Partial / Split Match"
        else:
            m_class = "Unmatched / Sorter-Unique"

        records.append({
            f'{sorter1_name.lower()}_unit_id': int(u1),
            f'{sorter1_name.lower()}_n_spikes': len(s1),
            f'best_{sorter2_name.lower()}_unit_id': int(best_u2) if best_u2 is not None else -1,
            'jaccard_agreement': round(best_jacc, 4),
            'spike_recall': round(best_rec, 4),
            'spike_precision': round(best_prec, 4),
            'match_classification': m_class
        })

    return pd.DataFrame(records)

def compute_pairwise_agreement_matrix(
    sorter_alf_dirs: Dict[str, Path],
    delta_s: float = 0.0015
) -> pd.DataFrame:
    """
    Builds a summary matrix of mean Jaccard agreement and matched unit proportions across all pairs of sorters.
    """
    sorter_names = list(sorter_alf_dirs.keys())
    matrix = np.zeros((len(sorter_names), len(sorter_names)))

    loaded_data = {}
    for name, alf_dir in sorter_alf_dirs.items():
        alf_dir = Path(alf_dir)
        t_file = alf_dir / "spike_times.npy"
        c_file = alf_dir / "spike_clusters.npy"
        if t_file.exists() and c_file.exists():
            loaded_data[name] = (np.load(t_file), np.load(c_file))

    for i, name1 in enumerate(sorter_names):
        for j, name2 in enumerate(sorter_names):
            if i == j:
                matrix[i, j] = 1.0
            elif name1 in loaded_data and name2 in loaded_data:
                t1, c1 = loaded_data[name1]
                t2, c2 = loaded_data[name2]
                df_m = match_two_sorters(t1, c1, t2, c2, name1, name2, delta_s=delta_s)
                if len(df_m) > 0:
                    matrix[i, j] = round(df_m['jaccard_agreement'].mean(), 3)

    return pd.DataFrame(matrix, index=sorter_names, columns=sorter_names)
