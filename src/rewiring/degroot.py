from typing import Union, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import copy
import networkx as nx
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


class Result:
    def __init__(self, m: np.ndarray, m0: np.ndarray, error: float, initial_error: float, converged: bool, iterations: int, trajectory: Dict):
        self.m = m
        self.m0 = m0
        self.error = error
        self.initial_error = initial_error
        self.converged = converged
        self.iterations = iterations
        self.trajectory = trajectory


def myopic_search(m0: np.ndarray, niter_search: int, max_edge_value: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int) -> Result:
    m = m0.copy()
    initial_error, final_belief = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

    iteration = 1
    n_improvements = 0
    n_rounds_no_improvement = 0
    current_error = copy.deepcopy(initial_error)
    converged = False
    trajectory = {0: {'m': m0, 'final_belief': final_belief}}

    for i in range(niter_search):
        n_rewire = np.random.randint(0, m.shape[0])
        vs = np.random.choice(m.shape[0], n_rewire, replace=False)

        m_rewire = m.copy()
        for v in vs:
            sources = np.argwhere(m[v] > 0).flatten()
            targets = np.argwhere(m[v] == 0).flatten()
            targets = targets[targets != v]

            s = np.random.choice(sources)
            t = np.random.choice(targets)
            ev = np.random.randint(1, max_edge_value + 1)

            m_rewire[v, s] = 0
            m_rewire[v, t] = ev

        error_rewire, final_belief = degroot_err(m=m_rewire, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

        if error_rewire < current_error:
            m = m_rewire.copy()
            current_error = error_rewire
            trajectory[iteration] = {'m': m.copy(), 'final_belief': final_belief}
            n_improvements += 1
            n_rounds_no_improvement = 0

        iteration += 1
        n_rounds_no_improvement += 1

        if n_rounds_no_improvement > 1000:
            converged = True
            break

    result = Result(m=m, m0=m0, error=current_error, initial_error=initial_error, converged=converged, iterations=iteration, trajectory=trajectory)
    return result


def get_optimal_rewire(m: np.ndarray, v: int, sources: np.ndarray, targets: np.ndarray, max_edge_value: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int, symmetric: bool) -> Tuple[np.ndarray, float, np.ndarray]:
    best_m = None
    best_error = np.inf
    best_belief = None
    for s in sources:
        for t in targets:
            for ev in range(1, max_edge_value + 1):

                if symmetric:
                    error_rewire, final_belief, m_rewire = _evaluate_symmetric_rewire(m=m, v=v, s=s, t=t, ev=ev, initial_belief=initial_belief, correct_belief=correct_belief, niter_degroot=niter_degroot)
                else:
                    error_rewire, final_belief, m_rewire = _evaluate_rewire(m=m, v=v, s=s, t=t, ev=ev, initial_belief=initial_belief, correct_belief=correct_belief, niter_degroot=niter_degroot)

                if error_rewire < best_error:
                    best_m = m_rewire
                    best_error = error_rewire
                    best_belief = final_belief

    return best_m, best_error, best_belief


def _evaluate_rewire(m: np.ndarray, v: int, s: int, t: int, ev: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int) -> Tuple[float, np.ndarray, np.ndarray]:
    m_rewire = m.copy()
    m_rewire[v, s] = 0
    m_rewire[v, t] = ev
    error_rewire, final_belief = degroot_err(m=m_rewire, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)
    return error_rewire, final_belief, m_rewire


def _evaluate_symmetric_rewire(m: np.ndarray, v: int, s: int, t: int, ev: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int) -> Tuple[float, np.ndarray, np.ndarray]:
    m_rewire = m.copy()
    m_rewire[v, s] = 0
    m_rewire[s, v] = 0
    m_rewire[v, t] = ev
    m_rewire[t, v] = ev
    error_rewire, final_belief = degroot_err(m=m_rewire, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)
    return error_rewire, final_belief, m_rewire


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


def find_local_peaks(sample: List[np.ndarray], initial_belief: np.ndarray, correct_belief: float, niter_degroot: int):
    peaks = Parallel(n_jobs=-1)(delayed(_hill_climb)(m=s, initial_belief=initial_belief, correct_belief=correct_belief, niter_degroot=niter_degroot) for s in tqdm(sample))
    return peaks


def _hill_climb(m: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int) -> Dict[str, Any]:
    current_error, current_belief = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)
    m_ = m.copy()
    converged = False

    while not converged:
        edges = np.argwhere(m_ > 0)
        np.random.permutation(edges)  # Randomize the order of edges
        improved = False
        for e in edges:
            s = e[0]
            t = e[1]
            new_targets = np.argwhere(m_[s] == 0).flatten()
            new_targets = new_targets[new_targets != s]
            np.random.permutation(new_targets)

            for new_target in new_targets:
                m_rewire = m_.copy()
                m_rewire[s, t] = 0
                m_rewire[s, new_target] = 1
                error_rewire, final_belief = degroot_err(m=m_rewire, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

                if error_rewire < current_error:
                    m_ = m_rewire
                    current_error = error_rewire
                    current_belief = final_belief
                    improved = True

        if not improved:
            converged = True

    return {'error': current_error, 'estimate': np.mean(current_belief), 'm': m_}


def compute_histogram_of_solution_quality(m0: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int, n_samples: int):
    rewiring = get_rewiring_sample(m0=m0, n_rewiring_attempts=n_samples)
    errors = []

    for m in tqdm(rewiring):
        error, final_belief = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)
        errors.append({'error': error, 'estimate': np.mean(final_belief)})

    errors = pd.DataFrame(errors)
    return errors


if __name__ == '__main__':
   pass
