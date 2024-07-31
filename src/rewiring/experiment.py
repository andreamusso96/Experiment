import sys

import networkx as nx
import numpy as np
import pandas as pd

from degroot import find_local_peaks, extract_local_peak_stats

# Fixed parameters
k_watts_strogatz = 4
p_watts_strogatz = 0.3
correct_belief = 0.9
niter_degroot = 2
n_samples_rewiring = 100
n_samples_initial_belief = 10


def run(dims: int, vision: int, path_save: str):
    rewiring_sample = [nx.adjacency_matrix(nx.connected_watts_strogatz_graph(n=dims, k=k_watts_strogatz, p=p_watts_strogatz)).todense() for _ in range(n_samples_rewiring)]
    initial_beliefs = [np.random.uniform(0, 1, dims) for _ in range(n_samples_initial_belief)]

    res = []
    for i, init_belief in enumerate(initial_beliefs):
        peaks = find_local_peaks(sample=rewiring_sample, initial_belief=init_belief, correct_belief=correct_belief, niter_degroot=niter_degroot, vision=vision)
        peak_stats = extract_local_peak_stats(peaks=peaks)
        peak_stats['experiment'] = i
        res.append(peak_stats)

    res = pd.concat(res)
    res['dims'] = dims
    res['vision'] = vision
    res['correct_belief'] = correct_belief
    res['niter_degroot'] = niter_degroot
    res['k_watts_strogatz'] = k_watts_strogatz
    res['p_watts_strogatz'] = p_watts_strogatz
    res.to_csv(path_save, index=False)


if __name__ == '__main__':
    d = sys.argv[1]
    vis = sys.argv[2]
    path_save = sys.argv[3]
    run(dims=int(d), vision=int(vis), path_save=path_save)
