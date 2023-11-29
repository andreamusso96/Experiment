from typing import List

import mab_experiment as exp
import numpy as np


def get_inverting_non_stationary_multi_armed_bandit(mab_start_steps: List[int], n_bandits: int) -> exp.NonStationaryMultiArmedBandit:
    std = 0.1 * np.ones(n_bandits)
    mean_1 = np.linspace(0.7, 1.3, n_bandits)
    mean_2 = mean_1[::-1]
    return _create_ns_mab(mean1=mean_1, std1=std, mean2=mean_2, std2=std, mab_start_steps=mab_start_steps)


def get_spiking_non_stationary_multi_armed_bandit(mab_start_steps: List[int], n_bandits: int) -> exp.NonStationaryMultiArmedBandit:
    std = 0.1 * np.ones(n_bandits)
    mean_1 = np.linspace(0.7, 1.3, n_bandits)
    mean_2 = mean_1.copy()
    mean_2[0] = 4
    return _create_ns_mab(mean1=mean_1, std1=std, mean2=mean_2, std2=std, mab_start_steps=mab_start_steps)


def _create_ns_mab(mean1: np.ndarray, std1: np.ndarray, mean2: np.ndarray, std2: np.ndarray, mab_start_steps: List[int]) -> exp.NonStationaryMultiArmedBandit:
    bandits_1 = [exp.NormalDistributionBandit(bid=i, mean=mean, std=std) for i, (mean, std) in enumerate(zip(mean1, std1))]
    multi_armed_bandit_1 = exp.MultiArmedBandit(bandits=bandits_1)

    bandits_2 = [exp.NormalDistributionBandit(bid=i, mean=mean, std=std) for i, (mean, std) in enumerate(zip(mean2, std2))]
    multi_armed_bandit_2 = exp.MultiArmedBandit(bandits=bandits_2)

    ns_mab = exp.NonStationaryMultiArmedBandit(multi_armed_bandits=[multi_armed_bandit_1, multi_armed_bandit_2], mab_start_steps=mab_start_steps)
    return ns_mab
