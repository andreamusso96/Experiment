from typing import Union, Tuple
import numpy as np
import copy


def degroot_err(m: np.ndarray, initial_belief: np.ndarray, correct_belief: float, niter: int) -> float:
    m = m / np.sum(m, axis=1)
    mt = np.linalg.matrix_power(m, niter)
    pt = mt @ initial_belief
    error = np.sum(np.square(pt - correct_belief))
    return error


class Result:
    def __init__(self, m: np.ndarray, m0: np.ndarray, error: float, initial_error: float, converged: bool, iterations: int):
        self.m = m
        self.m0 = m0
        self.error = error
        self.initial_error = initial_error
        self.converged = converged
        self.iterations = iterations


def myopic_search(m0: np.ndarray, niter_search: int, convergence_bar: int, max_edge_value: int, initial_belief: np.ndarray, correct_belief: float, niter_degroot: int):
    m = m0.copy()
    initial_error = degroot_err(m=m, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

    n_rounds_no_improvement = 0
    iteration = 0
    current_error = copy.deepcopy(initial_error)
    converged = False

    for i in range(niter_search):

        i, j = get_random_edge(m)
        k = get_target_of_edge_rewire(i, m)
        if k is None:
            continue

        m_rewire = m.copy()
        m_rewire[i, j] = 0
        m_rewire[i, k] = np.random.randint(1, max_edge_value + 1)
        error_rewire = degroot_err(m=m_rewire, initial_belief=initial_belief, correct_belief=correct_belief, niter=niter_degroot)

        if error_rewire < current_error:
            m = m_rewire
            current_error = error_rewire
            n_rounds_no_improvement = 0
        else:
            n_rounds_no_improvement += 1

        iteration += 1
        if n_rounds_no_improvement > convergence_bar:
            converged = True
            break

    result = Result(m=m, m0=m0, error=current_error, initial_error=initial_error, converged=converged, iterations=iteration - n_rounds_no_improvement)
    return result


def get_random_edge(m: np.ndarray) -> Union[Tuple[int, int], Tuple[None, None]]:
    indices_non_zero = np.argwhere(m > 0)
    index_non_zero = np.random.choice(np.arange(len(indices_non_zero)))
    i, j = indices_non_zero[index_non_zero]
    return i, j


def get_target_of_edge_rewire(i: int, m: np.ndarray) -> Union[int, None]:
    indices_zero = np.argwhere(m[i] == 0).flatten()
    if len(indices_zero) == 1:  # Just the self-loop
        return None
    index_zero = np.random.choice(np.arange(len(indices_zero)))
    k = int(indices_zero[index_zero])
    return k