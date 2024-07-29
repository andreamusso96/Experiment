from typing import Union, Tuple, List, Dict
import numpy as np
import copy
import networkx as nx
from tqdm import tqdm


def degroot_err(m: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter: int) -> Tuple[float, Union[None, np.ndarray]]:
    m_ = m.astype(float)
    div = m_.sum(axis=1).reshape(-1, 1)
    m_ = np.divide(m_, div, out=np.zeros_like(m_), where=div != 0)
    mt = np.linalg.matrix_power(m_, niter)
    pt = mt @ initial_belief
    error = np.abs(np.mean(pt) - correct_belief)
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


def myopic_search(m0: np.ndarray, niter_search: int, max_edge_value: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int, symmetric: bool) -> Result:
    m = m0.copy()
    initial_error, final_belief = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

    iteration = 1
    n_iterations_no_improvement = 0
    current_error = copy.deepcopy(initial_error)
    converged = False
    trajectory = {0: {'m': m0, 'final_belief': final_belief}}

    for i in range(niter_search):
        nodes_random_order = np.random.permutation(m0.shape[0])
        improvement_made = False

        for v in nodes_random_order:
            sources = np.argwhere(m[v] > 0).flatten()
            targets = np.argwhere(m[v] == 0).flatten()
            targets = targets[targets != v]

            best_m, best_error, best_belief = get_optimal_rewire(m=m, v=v, sources=sources, targets=targets, max_edge_value=max_edge_value, initial_belief=initial_belief, correct_belief=correct_belief, niter_degroot=niter_degroot, symmetric=symmetric)

            iteration += 1
            n_iterations_no_improvement += 1

            if best_error < current_error:
                m = best_m.copy()
                current_error = best_error
                improvement_made = True
                trajectory[iteration] = {'m': m.copy(), 'final_belief': best_belief}
                n_iterations_no_improvement = 0

        if not improvement_made or current_error < 0.01:
            converged = True
            break

    result = Result(m=m, m0=m0, error=current_error, initial_error=initial_error, converged=converged, iterations=iteration - n_iterations_no_improvement, trajectory=trajectory)
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


if __name__ == '__main__':
   pass
