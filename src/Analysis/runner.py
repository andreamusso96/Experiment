import os
import sys

import mab_experiment as exp
import pandas as pd
import itertools
from tqdm import tqdm

from mab import get_spiking_non_stationary_multi_armed_bandit, get_inverting_non_stationary_multi_armed_bandit
from network import get_power_law_network, get_watts_strogatz_network, get_complete_network
from config import EXPERIMENT_RESULT_BASE_FOLDER


def experiment_1():
    # This experiment tests the hypothesis that centralized networks are worst at adapting to changes in a COMPLEX MAB problem than decentralized networks.
    folder = f'{EXPERIMENT_RESULT_BASE_FOLDER}/experiment_1'
    os.makedirs(folder) if not os.path.isdir(folder) else None

    n_ag = 50
    n_steps = 300
    agent_type = exp.AgentType.VOTER_MODEL

    # A complex MAB problem is one with many bandits
    mab_start_steps = [0, 50]
    n_bandits = [5, 10, 20]
    problems = []
    for nb in n_bandits:
        problem_inverting = get_inverting_non_stationary_multi_armed_bandit(mab_start_steps=mab_start_steps, n_bandits=nb)
        problem_spiked = get_spiking_non_stationary_multi_armed_bandit(mab_start_steps=mab_start_steps, n_bandits=nb)
        problems.append((problem_inverting, 'inverting', nb))
        problems.append((problem_spiked, 'spiking', nb))

    network_centralized_1 = get_power_law_network(n_nodes=n_ag, min_degree=3, exponent=2.1)
    network_centralized_2 = get_power_law_network(n_nodes=n_ag, min_degree=3, exponent=2.5)
    network_centralized_3 = get_power_law_network(n_nodes=n_ag, min_degree=3, exponent=2.9)
    network_decentralized_1 = get_watts_strogatz_network(n_nodes=n_ag, k=4, p=1)
    network_decentralized_2 = get_watts_strogatz_network(n_nodes=n_ag, k=4, p=0.1)
    network_decentralized_3 = get_watts_strogatz_network(n_nodes=n_ag, k=4, p=0.01)
    networks = [network_centralized_1, network_centralized_2, network_centralized_3, network_decentralized_1, network_decentralized_2, network_decentralized_3]

    # We look at a variety of softmax probabilities more for robustness than anything else
    softmax_probs = [1/(100*n_ag), 1 / (10 * n_ag), 1 / n_ag, 10 / n_ag, 1]
    memory_decay = [0.01, 0.1, 0.5, 0.9, 0.99]
    n_samples = 100

    experiment_params = itertools.product(softmax_probs, memory_decay, networks, problems, range(n_samples))
    experiment_info = []
    for eid, params in tqdm(list(enumerate(experiment_params))):
        softmax_prob, mem_decay, net, problem, k = params
        problem, problem_name, n_bandits = problem

        agents = exp.get_agents(n_agents=n_ag, agent_type=agent_type, softmax_prob=softmax_prob, memory_decay=mem_decay, initial_action=n_bandits-1)
        agent_params = {'agent_type': agent_type.value, 'initial_action': n_bandits-1, 'softmax_prob': softmax_prob, 'memory_decay': mem_decay}

        experiment = exp.run_experiment(eid=eid, n_steps=n_steps, agents=agents, non_stationary_multi_armed_bandit=problem, network=net.network)
        experiment.save(folder_path=folder)
        exp_params = {'eid': eid, 'n_agents': n_ag, 'n_steps': n_steps, 'network': net.name, 'mab_start_steps': mab_start_steps, 'n_bandits': n_bandits, 'sample': k, 'problem': problem_name}

        exp_params.update(agent_params)
        experiment_info.append(exp_params)

    experiment_info = pd.DataFrame(experiment_info)
    experiment_info.to_csv(f'{folder}/experiment_info.csv', index=False)


def experiment_2():
    # This experiment tests the hypothesis that small groups are worst at adapting to changes in a MAB problem than larger groups.
    folder = f'{EXPERIMENT_RESULT_BASE_FOLDER}/experiment_2'
    os.makedirs(folder) if not os.path.isdir(folder) else None

    n_steps = 300
    agent_type = exp.AgentType.VOTER_MODEL

    # A simple MAB problem is one with few bandits
    mab_start_steps = [0, 50]
    n_bandits = [3, 5, 10, 20]

    problems = []
    for nb in n_bandits:
        problem_inverting = get_inverting_non_stationary_multi_armed_bandit(mab_start_steps=mab_start_steps, n_bandits=nb)
        problem_spiked = get_spiking_non_stationary_multi_armed_bandit(mab_start_steps=mab_start_steps, n_bandits=nb)
        problems.append((problem_inverting, 'inverting', nb))
        problems.append((problem_spiked, 'spiking', nb))

    n_ag = [5, 10, 20, 50, 100]
    networks = [get_complete_network(n_nodes=n) for n in n_ag]

    softmax_probs = [0.0001, 0.001, 0.01, 0.1, 1]
    memory_decay = [0.01, 0.1, 0.5, 0.9, 0.99]
    n_samples = 100

    experiment_params = itertools.product(softmax_probs, memory_decay, networks, problems, range(n_samples))

    experiment_info = []
    for eid, params in tqdm(list(enumerate(experiment_params))):
        softmax_prob, mem_decay, net, problem, k = params
        problem, problem_name, n_bandits = problem

        agents = exp.get_agents(n_agents=net.network.number_of_nodes(), agent_type=agent_type, softmax_prob=softmax_prob, memory_decay=mem_decay, initial_action=n_bandits-1)
        agent_params = {'agent_type': agent_type.value, 'initial_action': n_bandits-1, 'softmax_prob': softmax_prob, 'memory_decay': mem_decay}

        experiment = exp.run_experiment(eid=eid, n_steps=n_steps, agents=agents, non_stationary_multi_armed_bandit=problem, network=net.network)
        experiment.save(folder_path=folder)
        exp_params = {'eid': eid, 'n_agents': net.network.number_of_nodes(), 'n_steps': n_steps, 'network': net.name, 'mab_start_steps': mab_start_steps, 'n_bandits': n_bandits, 'sample': k, 'problem': problem_name}

        exp_params.update(agent_params)
        experiment_info.append(exp_params)

    experiment_info = pd.DataFrame(experiment_info)
    experiment_info.to_csv(f'{folder}/experiment_info.csv', index=False)


if __name__ == '__main__':
    n_exp = sys.argv[1]
    if n_exp == '1':
        experiment_1()
    elif n_exp == '2':
        experiment_2()
