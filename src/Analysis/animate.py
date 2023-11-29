from typing import List, Tuple, Dict, Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc

from experiment import Experiment
from config import EXPERIMENT_RESULT_BASE_FOLDER

external_stylesheets = [dbc.themes.CYBORG]
experiment_folder = f'{EXPERIMENT_RESULT_BASE_FOLDER}/experiment_1'
experiment_info = pd.read_csv(f'{experiment_folder}/experiment_info.csv')
n_experiments = len(experiment_info)
n_steps_experiment = experiment_info['n_steps'].unique()[0]
slider_step = 10
ms_interval_simulation = 1000 / slider_step


def get_dropdown_options() -> Dict[str, List[Dict[str, Any]]]:
    options = {'network': [{'label': network, 'value': network} for network in experiment_info['network'].unique()],
               'problem': [{'label': problem, 'value': problem} for problem in experiment_info['problem'].unique()],
               'softmax_prob': [{'label': softmax_prob, 'value': softmax_prob} for softmax_prob in experiment_info['softmax_prob'].unique()],
               'memory_decay': [{'label': memory_decay, 'value': memory_decay} for memory_decay in experiment_info['memory_decay'].unique()]}
    return options


dropdown_options = get_dropdown_options()

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    html.H1("Experiment visualizer", className="text-center"),
    html.Hr(),
    dcc.Interval(id="animate", interval=ms_interval_simulation, n_intervals=0, disabled=True),
    dbc.Row([
        dbc.Col(dbc.Button("Play", id="play", className="mr-1"), width=1),
        dbc.Col(dcc.Dropdown(id='dropdown_problem', options=dropdown_options['problem'], value=dropdown_options['problem'][0]['value']), width=1),
        dbc.Col(dcc.Dropdown(id='dropdown_softmax_prob', options=dropdown_options['softmax_prob'], value=dropdown_options['softmax_prob'][0]['value']), width=1),
        dbc.Col(dcc.Dropdown(id='dropdown_memory_decay', options=dropdown_options['memory_decay'], value=dropdown_options['memory_decay'][0]['value']), width=1),
        dbc.Col(dcc.Input(id="input_sample", type="number", value=0, min=0, max=99, step=1), width=1),

        dbc.Col(dcc.Dropdown(id='dropdown_network_1', options=dropdown_options['network'], value=dropdown_options['network'][0]['value']), width=2),
        dbc.Col(dcc.Dropdown(id='dropdown_network_2', options=dropdown_options['network'], value=dropdown_options['network'][1]['value']), width=2)]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='network_fig_1'), width=6),
        dbc.Col(dcc.Graph(id='network_fig_2'), width=6)]),
    html.Hr(),
    dcc.Slider(0, n_steps_experiment - 1, slider_step, value=0, id='step_slider'),
    html.Hr(),
    dbc.Row([
        dbc.Col(dcc.Graph(id='avg_return_fig'), width=8),
        dbc.Col(dcc.Graph(id='bandit_payoff_fig'), width=4)
    ])])


@callback(
    Output('network_fig_1', 'figure'),
    Output('network_fig_2', 'figure'),
    Output('bandit_payoff_fig', 'figure'),
    Output('avg_return_fig', 'figure'),
    Input('dropdown_problem', 'value'),
    Input('dropdown_softmax_prob', 'value'),
    Input('dropdown_memory_decay', 'value'),
    Input('input_sample', 'value'),
    Input('dropdown_network_1', 'value'),
    Input('dropdown_network_2', 'value'),
    Input('step_slider', 'value'))
def change_figure_on_slider_change(problem: str, softmax_prob: float, memory_decay: float, sample: int, network_1: str, network_2: str, step: int) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    eid1 = get_eid(network=network_1, problem=problem, softmax_prob=softmax_prob, memory_decay=memory_decay, sample=sample)
    eid2 = get_eid(network=network_2, problem=problem, softmax_prob=softmax_prob, memory_decay=memory_decay, sample=sample)
    experiment_1 = Experiment.load(folder_path=experiment_folder, eid=eid1)
    experiment_2 = Experiment.load(folder_path=experiment_folder, eid=eid2)
    fig_network_1 = make_network_figure(experiment=experiment_1, step=step)
    fig_network_2 = make_network_figure(experiment=experiment_2, step=step)
    bandit_payoff = make_bandit_payoff_figure(experiment=experiment_1, step=step)
    fig_avg_return = make_average_return_figure(experiment=experiment_1, step=step, line_color='blue')
    fig_avg_return = make_average_return_figure(experiment=experiment_2, step=step, fig=fig_avg_return, line_color='red')
    return fig_network_1, fig_network_2, bandit_payoff, fig_avg_return


def get_eid(network: str, problem: str, softmax_prob: float, memory_decay: float, sample: int) -> int:
    eid = experiment_info.loc[(experiment_info['network'] == network) & (experiment_info['softmax_prob'] == softmax_prob) & (experiment_info['memory_decay'] == memory_decay) & (experiment_info['sample'] == sample) & (experiment_info['problem'] == problem)]['eid'].values[0]
    return eid


@callback(
    Output("step_slider", "value"),
    Input("animate", "n_intervals"),
    State("step_slider", "value"),
    prevent_initial_call=True)
def animate_figure(n, current_slider_value):
    next_slider_value = current_slider_value + 1
    if next_slider_value > n_steps_experiment - 1:
        next_slider_value = next_slider_value - 1
    return next_slider_value


@callback(
    Output("animate", "disabled"),
    Input("play", "n_clicks"),
    State("animate", "disabled"),
)
def toggle_animation(n, playing):
    if n:
        return not playing
    return playing


def make_network_figure(experiment: Experiment, step: int) -> go.Figure:
    map_action_to_color = get_map_action_to_color(experiment=experiment, step=step)
    set_network_layout(network=experiment.network)
    node_trace = get_node_trace(network=experiment.network, actions=experiment.action_history.loc[step].values, map_action_to_color=map_action_to_color)
    edge_trace = get_edge_trace(network=experiment.network)

    fig = go.Figure()
    fig.add_traces([edge_trace, node_trace])
    fig.update_layout(title=f'Step {step}', title_x=0.5, margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      template='plotly_white', font=dict(size=15, color='black'), height=600, plot_bgcolor='black', paper_bgcolor='black')
    return fig


def make_bandit_payoff_figure(experiment: Experiment, step: int) -> go.Figure:
    map_action_to_color = get_map_action_to_color(experiment=experiment, step=step)
    map_action_to_payoff = get_map_action_mean_payoff(experiment=experiment, step=step)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(map_action_to_payoff.keys()), y=list(map_action_to_payoff.values()), marker=dict(color=list(map_action_to_color.values()))))
    fig.update_layout(xaxis_title='Bandit id', yaxis_title='Bandit mean payoff', font=dict(size=15, color='white'), plot_bgcolor='black', paper_bgcolor='black', template='plotly_white')
    return fig


def make_average_return_figure(experiment: Experiment, step: int, fig: go.Figure = None, line_color: str = 'white') -> go.Figure:
    average_returns = experiment.payoff_history.loc[:step].mean(axis=1)
    if fig is None:
        fig = go.Figure()
        fig.update_layout(xaxis_title='Step', yaxis_title='Average return', font=dict(size=15, color='white'), plot_bgcolor='black', paper_bgcolor='black', template='plotly_white')

    trace = go.Scatter(x=average_returns.index, y=average_returns.values, mode='lines', name=f'AR {experiment.eid}', line=dict(color=line_color, width=2))
    fig.add_trace(trace)
    return fig


def set_network_layout(network: nx, seed: int = 42) -> None:
    pos = nx.spring_layout(network, seed=seed)
    for node in network.nodes():
        network.nodes[node]['pos'] = pos[node]


def get_map_action_mean_payoff(experiment: Experiment, step: int) -> Dict[int, float]:
    current_bandit_specs = _get_current_map_specs(experiment=experiment, step=step)
    bandit_ids_and_means = current_bandit_specs[['bid', 'mean']].copy()
    return bandit_ids_and_means.set_index('bid')['mean'].to_dict()


def get_map_action_to_color(experiment: Experiment, step: int) -> Dict[int, str]:
    current_map_specs = _get_current_map_specs(experiment=experiment, step=step)
    bandit_ids = sorted(current_map_specs['bid'].values)
    colors = px.colors.qualitative.Light24
    n_colors = len(colors)
    map_action_to_color = {bid: colors[i % n_colors] for i, bid in enumerate(bandit_ids)}
    return map_action_to_color


def _get_current_map_specs(experiment: Experiment, step: int) -> pd.DataFrame:
    mab_start_steps = experiment.ns_mab_specs['mab_start_step'].unique()
    start_step_current_mab = mab_start_steps[mab_start_steps <= step].max()
    current_map_specs = experiment.ns_mab_specs.loc[experiment.ns_mab_specs['mab_start_step'] == start_step_current_mab]
    return current_map_specs


def get_node_trace(network: nx.Graph, actions: List[int], map_action_to_color: Dict[int, str]) -> go.Scatter:
    node_x = []
    node_y = []
    for node in network.nodes():
        x, y = network.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    colors = [map_action_to_color[action] for action in actions]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(color=colors, size=15, line=dict(color='white', width=0.5)), showlegend=False)
    return node_trace


def get_edge_trace(network: nx.Graph) -> go.Scatter:
    edge_x = []
    edge_y = []
    for edge in network.edges():
        x0, y0 = network.nodes[edge[0]]['pos']
        x1, y1 = network.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines', showlegend=False)
    return edge_trace


if __name__ == '__main__':
    app.run_server(debug=True)
