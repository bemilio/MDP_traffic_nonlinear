import matplotlib as mpl
mpl.interactive(True)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
import networkx as nx
import numpy as np
import torch


def plot_agent_route(x, i_agent, T_horiz, road_graph, edge_time_to_index):
    fig, ax = plt.subplots(T_horiz, 2, figsize=(5, 10 * 1.4), layout='constrained')
    pos = nx.get_node_attributes(road_graph, 'pos')
    colors = []
    edgelist = []
    nodes = []
    for t in range(T_horiz):
        for edge in road_graph.edges:
            # if not edge[0] == edge[1]:
            weight_road = torch.sum(x[0, i_agent, edge_time_to_index[(edge, t)]], 0)
            color_edge = weight_road
            colors.append(int(206 * color_edge) + 50)
            edgelist.append(edge)
        for node in road_graph.nodes:
            nodes.append(node)
        pos = nx.kamada_kawai_layout(road_graph)
        plt.sca(ax[t, 0])
        nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
        nx.draw_networkx_labels(road_graph, pos)
        nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.Reds,
                               connectionstyle='arc3, rad = 0.1')
    plt.show(block=True)
