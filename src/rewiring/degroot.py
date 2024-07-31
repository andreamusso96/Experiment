from typing import Union, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def degroot_err(m: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter: int) -> Tuple[float, Union[None, np.ndarray]]:
    m_ = m.astype(float)
    div = m_.sum(axis=1).reshape(-1, 1)
    m_ = np.divide(m_, div, out=np.zeros_like(m_), where=div != 0)
    mt = np.linalg.matrix_power(m_, niter)
    pt = mt @ initial_belief
    error = np.abs(np.mean(pt - correct_belief))
    return error, pt


def get_rewiring_sample(m0: np.ndarray, n_rewiring_attempts: int):
    rewiring = []

    n = m0.shape[0]
    for _ in range(n_rewiring_attempts):
        m = m0.copy()
        for i in range(n):
            np.random.shuffle(m[i])

            # Rewire self loops
            if m[i, i] == 1:
                j = np.random.randint(0, n)
                while m[i, j] == 1:
                    j = np.random.randint(0, n)

                m[i, j] = 1
                m[i, i] = 0

        rewiring.append(m)

    unique_rewiring = np.unique([s.flatten() for s in rewiring], axis=0)
    unique_rewiring = [s.reshape(n, n) for s in unique_rewiring]
    return unique_rewiring


def find_local_peaks(sample: List[np.ndarray], initial_belief: np.ndarray, correct_belief: float, niter_degroot: int, vision: int) -> List[Dict[str, Any]]:
    peaks = Parallel(n_jobs=-1)(delayed(_hill_climb)(m=s, initial_belief=initial_belief, correct_belief=correct_belief, niter_degroot=niter_degroot, vision=vision) for s in tqdm(sample))
    return peaks


def extract_local_peaks(peaks: List[Dict[str, Any]]) -> List[np.ndarray]:
    unique_peaks = np.unique([p['m'].flatten() for p in peaks], axis=0)
    shape = peaks[0]['m'].shape
    unique_peaks = [p.reshape(shape[0], shape[1]) for p in unique_peaks]
    return unique_peaks


def extract_local_peak_stats(peaks: List[Dict[str, Any]]) -> pd.DataFrame:
    unique_peaks = np.unique([p['m'].flatten() for p in peaks], axis=0)
    unique_peaks_labels = {p.astype(int).tobytes(): i for i, p in enumerate(unique_peaks)}

    res = []
    for p in peaks:
        peak = p['m'].flatten().astype(int).tobytes()
        peak_label = unique_peaks_labels[peak]
        error = p['error']
        estimate = p['estimate']
        res.append({'peak_label': peak_label, 'error': error, 'estimate': estimate})

    res = pd.DataFrame(res)
    res = res.groupby('peak_label').agg({'error': 'mean', 'estimate': 'mean', 'peak_label': 'count'}).rename(columns={'peak_label': 'count'}).reset_index()
    return res


def _get_neighbors_at_distance_1(m: np.ndarray) -> List[np.ndarray]:
    neighbors = []
    m_ = m.copy()
    edges = np.argwhere(m_ > 0)

    for e in edges:
        s = e[0]
        t = e[1]
        new_targets = np.argwhere(m_[s] == 0).flatten()
        new_targets = new_targets[new_targets != s]

        for new_target in new_targets:
            m_rewire = m_.copy()
            m_rewire[s, t] = 0
            m_rewire[s, new_target] = 1
            neighbors.append(m_rewire)

    return neighbors


def _get_neighbors_within_distance_k(m: np.ndarray, k: int) -> List[np.ndarray]:
    if k == 0:
        return [m]
    else:
        neighbors_within_distance_k_min_1 = _get_neighbors_within_distance_k(m=m, k=k - 1)
        neighbors = [m_ for m_ in neighbors_within_distance_k_min_1]
        for m_ in neighbors_within_distance_k_min_1:
            neighbors.extend(_get_neighbors_at_distance_1(m=m_))

        unique_neighbors = np.unique([s.flatten() for s in neighbors], axis=0)
        unique_neighbors = [s.reshape(m.shape[0], m.shape[1]) for s in unique_neighbors]
        return unique_neighbors


def _hill_climb(m: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int, vision: int) -> Dict[str, Any]:
    current_error, current_belief = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)
    m_ = m.copy()
    converged = False

    while not converged:
        neighbors = _get_neighbors_within_distance_k(m=m_, k=vision)
        improved = False

        for neighbor in neighbors:
            error, final_belief = degroot_err(m=neighbor, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

            if error < current_error:
                m_ = neighbor
                current_error = error
                current_belief = final_belief
                improved = True

        if not improved:
            converged = True

    return {'error': current_error, 'estimate': np.mean(current_belief), 'm': m_}


if __name__ == '__main__':
    pass
