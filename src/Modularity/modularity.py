import itertools
from enum import Enum

import networkx as nx
from draw_plotly_network import draw_plotly_network
import numpy as np
from networkx.algorithms.flow import edmonds_karp


class Position(Enum):
    PARALLEL = 'p'
    SEQUENTIAL = 's'
    RANDOM = 'r'

def set_max_flow(g, s, t):
    flow_val, flow_edges = nx.maximum_flow(g, _s=s, _t=t, capacity='capacity', flow_func=edmonds_karp)
    flow_vals_edges = {}
    for e in g.edges():
        flow_val_edge = flow_edges[e[0]][e[1]] + flow_edges[e[1]][e[0]]
        flow_vals_edges[e] = {'flow': flow_val_edge}

    nx.set_edge_attributes(g, flow_vals_edges)
    return g


def get_flow_network(components: int, n_nodes_component: int, n_edges_component: int, parallel: bool, spacing: float, position: Position) -> nx.Graph():
    g = nx.Graph()
    source = 0
    sink = n_nodes_component * components + 1
    g.add_node(source)
    g.add_node(sink)
    if parallel:
        g = _build_parallel_flow_network(g, components, n_nodes_component, n_edges_component)
    else:
        g = _build_sequential_flow_network(g, components, n_nodes_component, n_edges_component)

    if position == Position.PARALLEL:
        _set_parallel_pos(g=g, components=components, n_nodes_component=n_nodes_component, spacing=spacing)
    elif position == Position.SEQUENTIAL:
        _set_sequential_pos(g, components, n_nodes_component, spacing)
    elif position == Position.RANDOM:
        _set_random_pos(g, components, n_nodes_component, spacing)

    capacity = {e: {'capacity': np.random.randint(1 ,10)} for e in g.edges()}
    nx.set_edge_attributes(G=g, values=capacity)
    return g


def _build_parallel_flow_network(g: nx.Graph, components: int, n_nodes_component: int, n_edges_component: int) -> nx.Graph():
    for i in range(components):
        nodes, edges = _get_nodes_and_edges(i=i, n_nodes_component=n_nodes_component, n_edges_component=n_edges_component)
        g.add_nodes_from(nodes_for_adding=nodes)
        g.add_edges_from(ebunch_to_add=edges)
        g.add_edge(0, 1 + i * n_nodes_component)
        g.add_edge(5 + i * n_nodes_component, components * n_nodes_component + 1)

    return g


def _build_sequential_flow_network(g: nx.Graph, components: int, n_nodes_component: int, n_edges_component: int) -> nx.Graph():
    for i in range(components):
        nodes, edges = _get_nodes_and_edges(i=i, n_nodes_component=n_nodes_component, n_edges_component=n_edges_component)
        g.add_nodes_from(nodes_for_adding=nodes)
        g.add_edges_from(ebunch_to_add=edges)
        g.add_edge(i * n_nodes_component, 5 + i * n_nodes_component)

    g.add_edge((components - 1) * n_nodes_component + 1, components*n_nodes_component + 1)
    return g


def _get_nodes_and_edges(i: int, n_nodes_component: int, n_edges_component: int):
    nodes = list(range(1 + i * n_nodes_component, 1 + (i + 1) * n_nodes_component))
    all_edges = np.array(list(itertools.product(nodes, nodes)))
    edge_index = np.random.choice(list(range(n_nodes_component ** 2)), size=n_edges_component, replace=False)
    edges = all_edges[edge_index,]
    return nodes, edges


def _set_parallel_pos(g: nx.Graph, components: int, n_nodes_component: int, spacing: float):
    node_pos = {0: (0, spacing * components // 2), components * n_nodes_component + 1: (3, spacing * components // 2)}
    for i in range(components):
        for j in range(n_nodes_component):
            node_pos[1 + i*n_nodes_component + j] = (1 + np.random.uniform(), spacing*i + np.random.uniform())

    nx.set_node_attributes(g, node_pos, 'pos')
    return g


def _set_sequential_pos(g: nx.Graph, components: int, n_nodes_component: int, spacing: float):
    node_pos = {0: (0, 0), components * n_nodes_component + 1: (spacing * components, 0)}

    for i in range(components):
        for j in range(n_nodes_component):
            node_pos[1 + i*n_nodes_component + j] = (1 + spacing * i + np.random.uniform(), np.random.uniform() - 1/2)

    nx.set_node_attributes(g, node_pos, 'pos')
    return g


def _set_random_pos(g: nx.Graph, components: int, n_nodes_component: int, spacing: float):
    node_pos = {0: (-spacing, 0), components * n_nodes_component + 1: (spacing, 0)}
    for i in range(components):
        for j in range(n_nodes_component):
            node_pos[1 + i*n_nodes_component + j] = (spacing * (np.random.uniform() - 1/2), spacing * (np.random.uniform() - 1/2))

    nx.set_node_attributes(g, node_pos, 'pos')


if __name__ == '__main__':
    components, n_nodes_component = 5, 10
    g = get_flow_network(components=components, n_nodes_component=n_nodes_component, n_edges_component=30, parallel=True, spacing=3, position=Position.PARALLEL)
    set_max_flow(g, s=0, t=n_nodes_component * components + 1)
    draw_plotly_network(g=g)
