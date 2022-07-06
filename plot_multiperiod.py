import matplotlib as mpl
import seaborn as sns
import pandas as pd
from cmath import inf
from Utilities.generate_permutations import generate_permutations
from Utilities.multinomial_factor import multinomial_factor


mpl.interactive(True)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
import networkx as nx
import numpy as np
import pickle
import torch
from Utilities.plot_agent_route import plot_agent_route

f = open('/Users/ebenenati/surfdrive/TUDelft/Simulations/MDP_traffic_nonlinear/6_july_22_2/saved_test_result_multiperiod.pkl', 'rb')
f = open('saved_test_result_multiperiod.pkl', 'rb')
## Data structure:
## visited_nodes: Tensor with dimension (N_random_tests, N_agents, T_horiz+1)
visited_nodes, road_graph, edge_time_to_index, node_time_to_index, T_horiz, \
     initial_junctions_stored, final_destinations_stored, congestion_baseline_stored, cost_baseline_stored, is_feasible = pickle.load(f)
f.close()

N_tests = visited_nodes.size(0)
N_agents = visited_nodes.size(1)
N_edges = road_graph.number_of_edges()
N_vertices = road_graph.number_of_nodes()

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

# draw graph
# print("Plotting road graphs...")
# fig, ax = plt.subplots(T_horiz, 2, figsize=(5, 10 * 1.4), layout='constrained')
# pos = nx.get_node_attributes(road_graph, 'pos')
# colors = []
# edgelist = []
# nodes = []
# for t in range(T_horiz):
#     for edge in road_graph.edges:
#         if not edge[0] == edge[1]:
#             color_edge = np.matrix([0])
#             congestion = torch.sum(x[0, :, edge_time_to_index[(edge, t)]], 0) / N_agents
#             color_edge = congestion
#             colors.append(int(206 * color_edge) + 50)
#             edgelist.append(edge)
#     for node in road_graph.nodes:
#         nodes.append(node)
#     pos = nx.kamada_kawai_layout(road_graph)
#     plt.sca(ax[t, 0])
#     nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
#     nx.draw_networkx_labels(road_graph, pos)
#     nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.Reds,
#                            connectionstyle='arc3, rad = 0.1')
# Draw baseline
# for t in range(T_horiz):
#     for edge in road_graph.edges:
#         if not edge[0] == edge[1]:
#             color_edge = np.matrix([0])
#             if (edge, t) in congestion_baseline_stored[0].keys():
#                 congestion = congestion_baseline_stored[0][(edge, t)]
#             else:
#                 congestion = 0
#             color_edge = congestion
#             colors.append(int(206 * color_edge) + 50)
#             edgelist.append(edge)
#     for node in road_graph.nodes:
#         nodes.append(node)
#     pos = nx.kamada_kawai_layout(road_graph)
#     plt.sca(ax[t, 1])
#     nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
#     nx.draw_networkx_labels(road_graph, pos)
#     nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.Reds,
#                            connectionstyle='arc3, rad = 0.1')
# plt.show(block=False)
# plt.savefig('graph_plot.png')

shortest_paths = {}
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for i_agent in range(N_agents):
            shortest_paths.update({(i_test, i_agent): nx.shortest_path(road_graph,  source=initial_junctions_stored[i_test][i_agent],\
                             target = final_destinations_stored[i_test][i_agent], weight='travel_time' )})


print("Plotting congestion comparison...")
# bar plot of maximum edge congestion compared to constraint
fig, ax = plt.subplots(figsize=(5 * 1.5, 3.6 * 1.5), layout='constrained')
relative_congestions = torch.zeros(N_tests, N_edges, T_horiz)
relative_congestion_baseline = torch.zeros(N_tests, N_edges, T_horiz)
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for t in range(T_horiz):
            i_edge = 0
            for edge in road_graph.edges:
                if not edge[0] == edge[1]:
                    for i_agent in range(N_agents):
                        if (visited_nodes[i_test, i_agent, t], visited_nodes[i_test, i_agent, t+1]) == edge:
                            relative_congestions[i_test, i_edge, t] = relative_congestions[i_test, i_edge, t] + ((1. / N_agents) / road_graph[edge[0]][edge[1]]['limit_roads'])
                        if t+1 < len(shortest_paths[(i_test,i_agent)]):
                            if edge == (shortest_paths[(i_test,i_agent)][t], shortest_paths[(i_test, i_agent)][t + 1]):
                                relative_congestion_baseline[i_test, i_edge, t] = relative_congestion_baseline[i_test, i_edge, t] + ((1. / N_agents) / road_graph[edge[0]][edge[1]]['limit_roads'])
                    i_edge = i_edge + 1
max_congestions = torch.amax(relative_congestions[:, 0:i_edge, :], 2)  # ignore self-loops
max_congestion_baseline = torch.amax(relative_congestion_baseline[:, 0:i_edge, :], 2)
max_congestion_dataframe = pd.DataFrame(columns=['test', 'edge', 'method', 'value'])
for test in range(N_tests):
    if is_feasible[test]:
        for edge in range(i_edge):
            s_row = pd.DataFrame([[test, edge, 'Proposed', max_congestions[test, edge].item()]],
                                 columns=['test', 'edge', 'method', 'value'])
            max_congestion_dataframe = max_congestion_dataframe.append(s_row)
            s_row = pd.DataFrame([[test, edge, 'Baseline', max_congestion_baseline[test, edge].item()]],
                                 columns=['test', 'edge', 'method', 'value'])
            max_congestion_dataframe = max_congestion_dataframe.append(s_row)

if N_tests >= 2:
    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y=0.5, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax = sns.boxplot(x="edge", y="value", hue="method", data=max_congestion_dataframe, palette="Set3")
    ax.grid(True)
else:
    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y=0.5, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax.bar([k - 0.2 for k in range(max_congestions.size(1))], max_congestions.flatten(), width=0.4, align='center',
           label="Proposed")
    ax.bar([k + 0.2 for k in range(max_congestion_baseline.size(1))], max_congestion_baseline.flatten(), width=0.4,
           align='center',
           label="Naive")

plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'Congestion')
ax.set_ylim([-0.1, 1.2])

plt.show(block=False)
plt.savefig('congestion_multiperiod.png')


print("Plotting cost comparison")
fig, ax = plt.subplots(figsize=(5 * 1.5, 3.6 * 1.5), layout='constrained')

## Compute cost and baseline
cost_incurred = torch.zeros(N_tests, N_agents)
congestion = {}
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for i_agent in range(N_agents):
            for t in range(T_horiz):
                for edge in road_graph.edges:
                    if edge == (visited_nodes[i_test, i_agent, t], visited_nodes[i_test, i_agent, t+1]) and visited_nodes[i_test, i_agent, t] != visited_nodes[i_test, i_agent, t+1]:
                        if (i_test, edge, t) in congestion.keys():
                            congestion.update({(i_test, edge, t): congestion[(i_test, edge, t)] + 1. / N_agents})
                        else:
                            congestion.update({(i_test, edge, t): 1. / N_agents})
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for i_agent in range(N_agents):
            for t in range(T_horiz):
                if visited_nodes[i_test, i_agent, t] != visited_nodes[i_test, i_agent, t+1]:
                    edge_taken = (int(visited_nodes[i_test, i_agent, t].item()), int(visited_nodes[i_test, i_agent, t+1].item()))
                    capacity_edge = road_graph[edge_taken[0]][edge_taken[1]]['capacity']
                    cost_edge = road_graph[edge_taken[0]][edge_taken[1]]['travel_time'] * (
                                1 + 0.15 * (congestion[(i_test, edge_taken, t)] / capacity_edge) ** 4)
                    cost_incurred[i_test, i_agent] = cost_incurred[i_test, i_agent] + cost_edge
congestion_baseline = {}
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for i_agent in range(N_agents):
            for t in range(len(shortest_paths[(i_test,i_agent)])-1):
                for edge in road_graph.edges:
                    if edge == (shortest_paths[(i_test,i_agent)][t], shortest_paths[(i_test,i_agent)][t+1]):
                        if (i_test, edge, t) in congestion_baseline.keys():
                            congestion_baseline.update({(i_test, edge, t): congestion_baseline[(i_test, edge,t)] + 1./N_agents})
                        else:
                            congestion_baseline.update({(i_test, edge, t): 1. / N_agents})
cost_incurred_baseline = torch.zeros(N_tests, N_agents)
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for i_agent in range(N_agents):
            for t in range(len(shortest_paths[(i_test,i_agent)])-1):
                edge_taken = (shortest_paths[(i_test,i_agent)][t], shortest_paths[(i_test,i_agent)][t+1])
                capacity_edge = road_graph[edge_taken[0]][edge_taken[1]]['capacity']
                cost_edge = road_graph[edge_taken[0]][edge_taken[1]]['travel_time'] * ( 1 + 0.15 * (congestion_baseline[(i_test, edge_taken,t)]/capacity_edge)**4 )
                cost_incurred_baseline[i_test, i_agent] = cost_incurred_baseline[i_test, i_agent] + cost_edge
costs_dataframe = pd.DataFrame(columns=['test', 'agent', 'value'])
for test in range(N_tests):
    if is_feasible[test]:
        for agent in range(N_agents):
            s_row = pd.DataFrame([[test, agent,
                                   (cost_incurred_baseline[test, agent].item() - cost_incurred[test, agent].item()) / cost_incurred_baseline[
                                       test, agent].item()]], columns=['test', 'agent', 'value'])
            costs_dataframe = costs_dataframe.append(s_row)
if N_tests >= 2:
    ax = sns.boxplot(x="agent", y="value", data=costs_dataframe, palette="Set3")
    # ax.set_ylim([-1, 1])
    ax.grid(True)
else:
    print("Impossible to plot with only 1 test")
ax.axhline(y=0, linewidth=1, color='red')

ax.set_ylabel(r'$J(x_b) - J(x^{\star})/J(x_b)$ ')

plt.legend(prop={'size': 8})
plt.show(block=False)
plt.savefig('cost_multiperiod.png')
print("Done")


