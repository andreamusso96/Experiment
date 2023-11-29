import pandas as pd
import networkx as nx


class Experiment:
    def __init__(self, eid: int, payoff_history: pd.DataFrame, action_history: pd.DataFrame, ns_mab_specs: pd.DataFrame, network: nx.Graph, agent_metadata: pd.DataFrame):
        self.eid = eid
        self.payoff_history = payoff_history
        self.action_history = action_history
        self.ns_mab_specs = ns_mab_specs
        self.network = network
        self.agent_metadata = agent_metadata

    @classmethod
    def load(cls, folder_path: str, eid: int):
        payoff_history = pd.read_csv(f'{folder_path}/experiment_{eid}_payoff_history.csv', index_col=0)
        action_history = pd.read_csv(f'{folder_path}/experiment_{eid}_action_history.csv', index_col=0)
        ns_mab_specs = pd.read_csv(f'{folder_path}/experiment_{eid}_variable_multi_armed_bandit.csv', index_col=0)
        network_adjacency = pd.read_csv(f'{folder_path}/experiment_{eid}_network_adj.csv', index_col=0)
        network_adjacency.columns = network_adjacency.columns.astype(int)
        network = nx.from_pandas_adjacency(network_adjacency)
        agent_metadata = pd.read_csv(f'{folder_path}/experiment_{eid}_agent_metadata.csv', index_col=0)
        return Experiment(eid=eid, payoff_history=payoff_history, action_history=action_history, ns_mab_specs=ns_mab_specs, network=network, agent_metadata=agent_metadata)

    def get_number_of_steps(self):
        return self.payoff_history.shape[0]

    def get_number_of_agents(self):
        return self.payoff_history.shape[1]


if __name__ == '__main__':
    exp = Experiment.load(folder_path='/Users/andrea/Desktop/PhD/Packages/Utils/mab_experiment/temp', eid=0)