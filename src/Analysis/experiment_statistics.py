from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment import Experiment


def average_per_step_regret(experiment: Experiment) -> pd.DataFrame:
    optimal_policy = infer_optimal_policy(experiment=experiment)
    regret_per_step__avg_across_agents = optimal_policy['payoff'] - experiment.payoff_history.mean(axis=1)
    regret_per_step__avg_across_agents = regret_per_step__avg_across_agents.to_frame(name=f'avg_regret_per_step_{experiment.eid}')
    return regret_per_step__avg_across_agents


def infer_optimal_policy(experiment: Experiment) -> pd.DataFrame:
    mab_change_step = list(experiment.ns_mab_specs['mab_start_step'].unique()) + [experiment.get_number_of_steps()]
    action_history, payoff_history = [], []
    for i in range(len(mab_change_step) - 1):
        step = mab_change_step[i]
        mab_means_and_ids = experiment.ns_mab_specs.loc[experiment.ns_mab_specs['mab_start_step'] == step][['bid', 'mean']]
        mab_means_and_ids = mab_means_and_ids.sort_values(by='mean', ascending=False)
        best_bandit_id = mab_means_and_ids.iloc[0]['bid']
        best_bandit_payoff = mab_means_and_ids.iloc[0]['mean']
        action_history.append(best_bandit_id)
        payoff_history.append(best_bandit_payoff)
        action_history += [best_bandit_id] * (mab_change_step[i + 1] - step - 1)
        payoff_history += [best_bandit_payoff] * (mab_change_step[i + 1] - step - 1)

    optimal_policy = pd.DataFrame({'action': action_history, 'payoff': payoff_history})
    return optimal_policy


def successful_adaptation(experiment: Experiment, threshold: float) -> int:
    average_payoff_last_50_steps = experiment.payoff_history.iloc[-50:].mean(axis=0).mean()
    if average_payoff_last_50_steps > threshold:
        return 1
    else:
        return 0


if __name__ == '__main__':
    experiment_folder_path = '/Users/andrea/Desktop/PhD/Projects/Current/Experiment/Data/ExperimentResults/experiment_1'
    experiments = pd.read_csv(f'{experiment_folder_path}/experiment_info.csv')
    experiments = experiments.loc[experiments['problem'] == 'inverting']['eid'].values
    successful_adapts = []
    for eid in tqdm(experiments):
        exp = Experiment.load(folder_path=experiment_folder_path, eid=eid)
        successful_adapts.append({'eid': eid, 'successful_adaptation': successful_adaptation(experiment=exp, threshold=1.1)})

    successful_adapts = pd.DataFrame(successful_adapts)
    successful_adapts.to_csv('/Users/andrea/Desktop/PhD/Projects/Current/Experiment/Data/Statistics/successful_adapts_inverting.csv', index=False)
