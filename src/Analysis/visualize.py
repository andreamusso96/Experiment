import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

statistics_folder = '/Users/andrea/Desktop/PhD/Projects/Current/Experiment/Data/Statistics'
experiment_folder = '/Users/andrea/Desktop/PhD/Projects/Current/Experiment/Data/ExperimentResults'


def plot_softmax_prob_against_total_regret_after_step(step: int) -> px.scatter:
    average_per_round_regret = pd.read_csv(f'{statistics_folder}/avg_per_step_regret.csv', index_col=0)
    average_per_round_regret.columns = [int(c.split('_')[-1]) for c in average_per_round_regret.columns]
    total_per_round_regret = average_per_round_regret.loc[step:].sum(axis=0).to_frame(name='total_regret')

    experiment_specs = pd.read_csv(f'{experiment_folder}/experiment_specs.csv')
    experiment_specs = experiment_specs.set_index('eid')
    softmax_prob_network_and_regret = pd.concat([experiment_specs[['softmax_prob', 'network']], total_per_round_regret], axis=1)

    fig = go.Figure()
    for network in softmax_prob_network_and_regret['network'].unique():
        experiments_with_network = softmax_prob_network_and_regret.loc[softmax_prob_network_and_regret['network'] == network]
        grouped_regret = experiments_with_network.groupby('softmax_prob')['total_regret']
        mean = grouped_regret.mean()
        std = grouped_regret.std()
        count = grouped_regret.count()
        conf_int = 1.96 * std / np.power(count, 0.5)
        fig.add_trace(go.Scatter(x=mean.index, y=mean.values, error_y=dict(type='data', array=conf_int.values, visible=True),
                                 mode='markers', name=network, marker=dict(size=10)))

    fig.update_layout(title=f'Total regret after step {step}', xaxis_title='Softmax probability', yaxis_title='Total regret', font=dict(size=20, color='black'), template='plotly_white')
    fig.update_xaxes(type='log')
    return fig


def plot_heatmap_successful_adaptations_by_network_type(problem_type: str) -> px.imshow:
    successful_adaptations = pd.read_csv(f'{statistics_folder}/successful_adapts_{problem_type}.csv')
    successful_adaptations = successful_adaptations.set_index('eid')
    experiment_specs = pd.read_csv(f'{experiment_folder}/experiment_1/experiment_info.csv')
    experiment_specs = experiment_specs.set_index('eid')
    experiment_specs_and_successful_adaptations = pd.concat([experiment_specs[['softmax_prob', 'memory_decay', 'network']], successful_adaptations], axis=1).dropna()
    networks = experiment_specs['network'].unique()
    networks = [n for n in networks if n != 'cc_5']
    hms = []
    zmin, zmax = 1, 0
    for network in networks:
        heatmap = experiment_specs_and_successful_adaptations.loc[experiment_specs['network'] == network].groupby(['softmax_prob', 'memory_decay'])['successful_adaptation'].mean().to_frame(name='share_successful_adaptations').reset_index()
        heatmap = heatmap.pivot(index='memory_decay', columns='softmax_prob', values='share_successful_adaptations')
        heatmap = heatmap.sort_index(ascending=False)
        if heatmap.min().min() < zmin:
            zmin = heatmap.min().min()
        if heatmap.max().max() > zmax:
            zmax = heatmap.max().max()
        hms.append(heatmap)

    rows, cols = int(np.ceil(len(networks) / 3)), 3
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=networks, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.05, vertical_spacing=0.05)
    for i, hm in enumerate(hms):
        row, col = i // cols + 1, i % cols + 1
        hm_trace = go.Heatmap(z=hm.values, x=[str(a) for a in hm.columns], y=[str(a) for a in hm.index], name=networks[i], colorscale='Viridis', zmin=zmin, zmax=zmax)
        fig.add_trace(hm_trace, row=row, col=col)
        if row == rows:
            fig.update_xaxes(title_text='Softmax probability', row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text='Memory decay', row=row, col=col)

    fig.update_layout(title=f'Share of successful adaptations by network type ({problem_type})', font=dict(size=20, color='black'), template='plotly_white')
    fig.update_yaxes(title_text='Memory decay')
    return fig

if __name__ == '__main__':
    plot_heatmap_successful_adaptations_by_network_type(problem_type='spiked').show()
