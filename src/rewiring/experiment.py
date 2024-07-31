import sys

import networkx as nx
import numpy as np
import pandas as pd
import h5py
import json

from degroot import find_local_peaks, get_rewiring_sample, extract_local_peaks

# Fixed parameters
k_watts_strogatz = 4
p_watts_strogatz = 0.3
correct_belief = 0.9
niter_degroot = 2
n_samples_rewiring = 1000
n_samples_initial_belief = 100


def run(dims: int, path_save: str):
    base_m = nx.adjacency_matrix(nx.connected_watts_strogatz_graph(n=dims, k=k_watts_strogatz, p=p_watts_strogatz)).todense()
    rewiring_sample = get_rewiring_sample(m0=base_m, n_rewiring_attempts=n_samples_rewiring)
    initial_beliefs = [np.random.uniform(0, 1, dims) for _ in range(n_samples_initial_belief)]

    peaks = []
    for i, init_belief in enumerate(initial_beliefs):
        peaks_ = find_local_peaks(sample=rewiring_sample, initial_belief=init_belief, correct_belief=correct_belief, niter_degroot=niter_degroot, vision=1)
        peaks_ = extract_local_peaks(peaks_)
        peaks_ = [{j: p.tolist()} for j, p in enumerate(peaks_)]
        peaks.append({i: peaks_})

    json.dump(peaks, open(path_save, 'w'))


if __name__ == '__main__':
    d = sys.argv[1]
    path_save = sys.argv[2]
    run(dims=int(d), path_save=path_save)
