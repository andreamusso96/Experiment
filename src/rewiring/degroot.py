from typing import Union, Tuple, List, Dict
import numpy as np
import copy
import networkx as nx
from tqdm import tqdm


def degroot_err(m: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter: int) -> Tuple[float, Union[None, np.ndarray]]:
    m_ = m.copy()
    div = m_.sum(axis=1)
    m_ = m_ * (1 / div).reshape(-1, 1)
    mt = np.linalg.matrix_power(m_, niter)
    if np.sum(np.abs(mt - np.linalg.matrix_power(mt, 10))) > 1e-6:
        return np.inf, None
    pt = mt @ initial_belief
    error = 1/m.shape[0] * np.sum(np.abs(pt - correct_belief))
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


def myopic_search(m0: np.ndarray, niter_search: int, max_edge_value: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int):
    m = m0.copy()
    initial_error, final_belief = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

    iteration = 1
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

            best_m, best_error, best_belief = get_optimal_rewire(m=m, v=v, sources=sources, targets=targets, max_edge_value=max_edge_value, initial_belief=initial_belief, correct_belief=correct_belief, niter_degroot=niter_degroot)

            if best_error < current_error:
                m = best_m.copy()
                current_error = best_error
                improvement_made = True
                trajectory[iteration] = {'m': m.copy(), 'final_belief': best_belief}

            iteration += 1

        if not improvement_made or current_error < 1e-3:
            converged = True

    result = Result(m=m, m0=m0, error=current_error, initial_error=initial_error, converged=converged, iterations=iteration, trajectory=trajectory)
    return result


def get_optimal_rewire(m: np.ndarray, v: int, sources: np.ndarray, targets: np.ndarray, max_edge_value: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int) -> Tuple[np.ndarray, float, np.ndarray]:
    best_m = None
    best_error = np.inf
    best_belief = None
    for s in sources:
        for t in targets:
            for ev in range(1, max_edge_value + 1):
                m_rewire = m.copy()
                m_rewire[v, s] = 0
                m_rewire[v, t] = ev
                error_rewire, final_belief = degroot_err(m=m_rewire, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

                if error_rewire < best_error:
                    best_m = m_rewire
                    best_error = error_rewire
                    best_belief = final_belief

    return best_m, best_error, best_belief


if __name__ == '__main__':
    dims = 10
    erdos_reni_p = 0.5
    n_initial_conditions_attempts = 100
    max_edge_value = 4
    init_belief = np.random.rand(dims)
    correct_belief = 0.5
    niter_degroot = 1000
    niter_search = 10_000
    convergence_bar = 100


    def generate_initial_conditions(n: int, p: float, n_attempts: int):
        initial_conditions = []
        for i in range(n_attempts):
            m = nx.adjacency_matrix(nx.erdos_renyi_graph(n=n, p=p)).todense()
            if np.min(np.sum(m, axis=1)) == 0 or np.max(np.sum(m, axis=1)) == n - 1:
                continue

            m = np.random.randint(1, max_edge_value + 1, size=(n, n)) * m
            initial_conditions.append(m)

        unique_initial_conditions = np.unique([s.flatten() for s in initial_conditions], axis=0)
        unique_initial_conditions = [s.reshape(dims, dims) for s in unique_initial_conditions]
        print('Number of unique initial conditions:', len(unique_initial_conditions))
        return unique_initial_conditions

    init_cond = generate_initial_conditions(n=dims, p=erdos_reni_p, n_attempts=n_initial_conditions_attempts)

    results = []
    for x in tqdm(init_cond):
        res = myopic_search(m0=x, niter_search=niter_search, convergence_bar=convergence_bar, max_edge_value=max_edge_value, initial_belief=init_belief, correct_belief=correct_belief, niter_degroot=niter_degroot)
        results.append(res)
