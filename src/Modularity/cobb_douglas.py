import numpy as np
import networkx as nx
from typing import Callable
from scipy.optimize import minimize, LinearConstraint, Bounds


def generate_random_matrix(n: int):
    g = nx.erdos_renyi_graph(n=n, p=0.5)
    m = nx.adjacency_matrix(g).todense()
    return m


def get_constraint(n: int):
    diag = np.random.uniform(0.5, 5, (n, n))
    return LinearConstraint(diag, np.zeros(n) + 0.1, np.ones(n))

def get_bounds(n: int):
    return Bounds(np.zeros(n) + 0.001, 5*np.ones(n))


def step(x, s):
    if x < s:
        return 0
    else:
        return 1


def step_product(x: np.ndarray, s: np.array):
    return np.prod([step(x[i], s[i]) for i in range(len(x))])


def get_stepwise(m: np.ndarray) -> Callable[[np.ndarray], float]:
    def stepwise(x: np.ndarray):
        vals = []
        for i in range(m.shape[0]):
            indices = np.argwhere(m[i, :] > 0).flatten()
            vals.append(step_product(x[indices], m[i, indices]))
        return np.sum(vals)

    return stepwise


def maximise_stepwise(stepwise: Callable[[np.ndarray], float], n: int, bounds: Bounds):
    x0 = np.random.uniform(0.5, 3, n)
    return minimize(lambda x: -1 * stepwise(x), x0=x0, bounds=bounds)


if __name__ == '__main__':
    n = 20
    m = generate_random_matrix(n=n)
    step_wise = get_stepwise(m)
    bounds = Bounds(np.zeros(n), 10*np.ones(n))
    res = [maximise_stepwise(step_wise, n, bounds) for _ in range(50)]
    from scipy.spatial.distance import pdist
    dist = pdist([r.x for r in res], metric='euclidean')
    import plotly.express as px
    fig = px.histogram(x=dist.flatten())
    fig.show()

