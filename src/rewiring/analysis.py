import copy
from typing import List, Dict, Any
import json

import numpy as np
import pandas as pd
import scipy as sp


def read_experiment(dim: int):
    path = f'/Users/andrea/Desktop/PhD/Projects/Current/Experiment/src/rewiring/experiment_attempts/dim/experiment_{dim}.json'
    data = json.load(open(path, 'r'))
    data_ = {}
    for d in data:
        if d['experiment'] not in data_:
            data_[d['experiment']] = [{'m': np.array(d['m']), 'error': float(d['error']), 'estimate': float(d['estimate'])}]
        else:
            data_[d['experiment']].append({'m': np.array(d['m']), 'error': float(d['error']), 'estimate': float(d['estimate'])})
    return data_


def peak_statistics(dim: int, vision: int):
    experiments = experiment_with_vision(dim=dim, vision=vision)
    stats = []
    for e in experiments:
        exp = experiments[e]

        flat_peaks = [p['m'].flatten() for p in exp]
        if len(flat_peaks) == 1:
            stats.append({'experiment': e, 'mean_hamming': 0, 'std_hamming': 0, 'n_peaks': 1, 'mean_error': exp[0]['error'], 'std_error': 0, 'diff_error': 0})
            continue

        hamming = sp.spatial.distance.pdist(flat_peaks, 'hamming')
        mean_hamming, std_hamming = np.mean(hamming), np.std(hamming)

        n_peaks = len(flat_peaks)

        err = [p['error'] for p in exp]
        mean_error, std_error = np.mean(err), np.std(err)
        diff_error = np.max(err) - np.min(err)

        stats.append({'experiment': e, 'mean_hamming': mean_hamming, 'std_hamming': std_hamming, 'n_peaks': n_peaks, 'mean_error': mean_error, 'std_error': std_error, 'diff_error': diff_error})

    stats = pd.DataFrame(stats)
    return stats


def experiment_with_vision(dim: int, vision: int):
    experiments = read_experiment(dim=dim)
    if vision == 1:
        return experiments

    experiments_with_vision = {}
    for e in experiments:
        exp = experiments[e]
        universe = {i: p for i, p in enumerate(exp)}
        refined_peaks = {}
        for p in exp:
            res = _hill_climb(m=p['m'], error=p['error'], estimate=p['estimate'], vision=vision, universe=universe)
            peak = res['m'].flatten().astype(int).tobytes()
            if peak not in refined_peaks:
                refined_peaks[peak] = res

        refined_peaks = list(refined_peaks.values())
        experiments_with_vision[e] = refined_peaks

    return experiments_with_vision


def _get_neighbors_within_distance_k(m: np.ndarray, k: int, universe: Dict) -> List[int]:
    flat_universe = [(id, u['m'].flatten()) for id, u in universe.items()]
    flat_m = m.flatten()
    hamming = 1/2 * (m.shape[0]**2) * sp.spatial.distance.cdist([flat_m], [u[1] for u in flat_universe], 'hamming')[0]
    neighbors = [u[0] for i, u in enumerate(flat_universe) if hamming[i] <= k]
    return neighbors


def _hill_climb(m: np.ndarray, error: float, estimate: float, vision: int, universe: Dict) -> Dict[str, Any]:
    current_error = copy.deepcopy(error)
    current_estimate = copy.deepcopy(estimate)

    m_ = m.copy()
    converged = False

    while not converged:
        neighbors = _get_neighbors_within_distance_k(m=m_, k=vision, universe=universe)
        improved = False

        for neighbor in neighbors:
            error = universe[neighbor]['error']

            if error < current_error:
                m_ = universe[neighbor]['m']
                current_error = error
                current_estimate = universe[neighbor]['estimate']
                improved = True

        if not improved:
            converged = True

    return {'error': current_error, 'estimate': current_estimate, 'm': m_}


if __name__ == '__main__':
    print(peak_statistics(dim=8, vision=3))
