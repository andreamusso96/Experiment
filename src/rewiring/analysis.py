import json

import numpy as np
import scipy as sp


def read_experiment(dim: int):
    path = f'/Users/andrea/Desktop/PhD/Projects/Current/Experiment/src/rewiring/experiment_attempts/dim/experiment_{dim}.json'
    data = json.load(open(path, 'r'))
    peaks = {}
    for e in data:
        for k, v in e.items():
            peaks[k] = []
            for p in v:
                for k_, m in p.items():
                    peaks[k].append(np.array(m))
    return peaks


def peak_statistics(dim: int):
    peaks = read_experiment(dim=dim)
    for k, v in peaks.items():
        flat_peaks = [p.flatten() for p in v]
        hamming_distances = sp.spatial.distance.pdist(flat_peaks, metric='hamming')
        mean, std = np.mean(hamming_distances), np.std(hamming_distances)

if __name__ == '__main__':
    len(read_experiment(dim=16))