import networkx as nx
import pandas as pd
from config import is_cluster

if not is_cluster:
    import graph_tool.all as gt
    import network_utils as nu


class Network:
    def __init__(self, network: nx.Graph, name: str):
        self.network = network
        self.name = name


def get_power_law_network(n_nodes: int, min_degree: int, exponent: float):
    if not is_cluster:
        c = min_degree * (exponent - 3)
        seed_graph = gt.complete_graph(N=min_degree + 1)
        g = gt.price_network(N=n_nodes - min_degree - 1, m=min_degree, c=c, directed=False, seed_graph=seed_graph)
        g.ep['weight'] = g.new_edge_property('double', val=1)
        g = nu.converter.gt_to_nx(g)
    else:
        g = nx.from_pandas_adjacency(pd.read_csv(f'../data/networks/power_law_{min_degree}_{exponent}.csv', index_col=0))
    return Network(network=g, name=f'pl_{min_degree}_{exponent}')


def get_watts_strogatz_network(n_nodes: int, k: int, p: float):
    g = nx.watts_strogatz_graph(n=n_nodes, k=k, p=p)
    return Network(network=g, name=f'ws_{k}_{p}')


def get_relaxed_caveman_network(n_nodes: int, k: int, p: float):
    assert n_nodes % k == 0, 'n_nodes must be divisible by k (number of nodes per clique)'
    n_cliques = int(n_nodes / k)
    g = nx.relaxed_caveman_graph(l=n_cliques, k=k, p=p)
    return Network(network=g, name=f'rc_{k}_{p}')


def get_connected_caveman_network(n_nodes: int, k: int):
    assert n_nodes % k == 0, 'n_nodes must be divisible by k (number of nodes per clique)'
    n_cliques = int(n_nodes / k)
    g = nx.connected_caveman_graph(l=n_cliques, k=k)
    return Network(network=g, name=f'cc_{k}')


def save_adjacency_network(network: nx.Graph, file_path: str):
    nx.to_pandas_adjacency(network).to_csv(file_path)


if __name__ == '__main__':
    pass
