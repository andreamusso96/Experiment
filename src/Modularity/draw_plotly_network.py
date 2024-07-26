import networkx as nx
import plotly.graph_objects as go


def draw_plotly_network(g: nx.Graph):
    fig = go.Figure()
    edge_trace = get_edge_trace(g)
    node_trace = get_node_trace(g)
    fig.add_traces(edge_trace)
    fig.add_trace(node_trace)
    fig.update_layout(template='plotly_white', hovermode='closest', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), showlegend=False)
    fig.show()


def get_edge_trace(g):
    traces = []
    for edge in g.edges():
        x0, y0 = g.nodes[edge[0]]['pos']
        x1, y1 = g.nodes[edge[1]]['pos']
        flow = g.edges[edge]['flow']
        color = 'red' if flow > 0 else 'black'
        width = g.edges[edge]['capacity']

        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none')

        traces.append(edge_trace)

    return traces

def get_node_trace(g):
    node_x = []
    node_y = []
    color = []
    for node in g.nodes():
        x, y = g.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        color.append('blue' if (node == 0 or node == len(g.nodes())-1) else 'black')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=color,
            size=10,
            line_width=2))

    return node_trace