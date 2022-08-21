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

f = open('/Users/ebenenati/surfdrive/TUDelft/Simulations/MDP_traffic_nonlinear/7_july_22/saved_test_result.pkl', 'rb')
# f = open('saved_test_result.pkl', 'rb')
## Data structure:
## x: Tensor with dimension (n. random tests, N agents, n variables)
## dual: Tensor with dimension (n. random tests, n constraints, )
## Residual: Tensor with dimension (n. random tests, K iterations)
x, dual, residual, cost, road_graph, edge_time_to_index, node_time_to_index, T_horiz, initial_junctions, final_destinations, \
congestion_baseline_stored, cost_baseline_stored = pickle.load(f)
f.close()

N_tests = x.size(0)
N_agents = x.size(1)
N_opt_var = x.size(2)
N_iter = residual.size(1)
N_edges = road_graph.number_of_edges()
N_vertices = road_graph.number_of_nodes()

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

## Plot # 1: Residuals
print("Plotting residual...")
fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained')
ax.loglog(np.arange(0, N_iter * 10, 10), residual[21, :])
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.legend()
plt.savefig('Residual.png')
plt.show()
plt.grid(True)

# print("Plotting road graphs...")
# fig, ax = plt.subplots(T_horiz, 2, figsize=(5, 10 * 1.4), layout='constrained')
# pos = nx.get_node_attributes(road_graph, 'pos')
#
# for t in range(T_horiz):
#     colors = []
#     edgelist = []
#     nodes = []
#     for edge in road_graph.edges:
#         if not edge[0] == edge[1]:
#             color_edge = np.matrix([0])
#             congestion = torch.sum(x[0, :, edge_time_to_index[(edge, t)]], 0) / N_agents
#             color_edge = congestion
#             colors.append(int(256 * color_edge))
#             edgelist.append(edge)
#     for node in road_graph.nodes:
#         nodes.append(node)
#     pos = nx.kamada_kawai_layout(road_graph)
#     plt.sca(ax[t, 0])
#     nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
#     nx.draw_networkx_labels(road_graph, pos)
#     nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.cool,
#                            connectionstyle='arc3, rad = 0.1')
# plt.show(block=False)

# Draw baseline
# for t in range(T_horiz):
#     colors = []
#     edgelist = []
#     nodes = []
#     for edge in road_graph.edges:
#         if not edge[0] == edge[1]:
#             color_edge = np.matrix([0])
#             if (edge, t) in congestion_baseline_stored[0].keys():
#                 congestion = congestion_baseline_stored[0][(edge, t)]
#             else:
#                 congestion = 0
#             color_edge = congestion
#             colors.append(int(256 * color_edge))
#             edgelist.append(edge)
#     for node in road_graph.nodes:
#         nodes.append(node)
#     pos = nx.kamada_kawai_layout(road_graph)
#     plt.sca(ax[t, 1])
#     nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
#     nx.draw_networkx_labels(road_graph, pos)
#     nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist=edgelist, edge_cmap=plt.cm.cool,
#                            connectionstyle='arc3, rad = 0.1')
# plt.show(block=False)
# plt.savefig('graph_plot.png')

### Plot #2 : congestion in tests vs. shortest path

print("Plotting congestion comparison...")
# bar plot of maximum edge congestion compared to constraint
fig, ax = plt.subplots(figsize=(5 * 1.5, 3.6 * 1.5), layout='constrained')
congestions = torch.zeros(N_tests, N_edges)
congestion_baseline = torch.zeros(N_tests, N_edges)
for i_test in range(N_tests):
    if is_feasible[i_test]:
        i_edges = 0
        for edge in road_graph.edges:
            if not edge[0] == edge[1]:
                congestions[i_test, i_edges] = 0
                for t in range(T_horiz):
                    relative_congestion = (torch.sum(x[i_test, :, edge_time_to_index[(edge, t)]], 0) / N_agents) \
                                          / road_graph[edge[0]][edge[1]]['limit_roads']
                    if relative_congestion>1.002:
                        print("Constraint not satisfied at test " + str(i_test)  )
                    congestions[i_test, i_edges] = max(relative_congestion, congestions[i_test, i_edges])
                    if (edge, t) in congestion_baseline_stored[i_test].keys():
                        relative_congestion_baseline = congestion_baseline_stored[i_test][(edge, t)] / \
                                                       road_graph[edge[0]][edge[1]]['limit_roads']
                        congestion_baseline[i_test, i_edges] = max(relative_congestion_baseline,
                                                                   congestion_baseline[i_test, i_edges])
                i_edges = i_edges + 1
congestions = congestions[:, 0:i_edges]  # ignore self-loops
congestion_baseline = congestion_baseline[:, 0:i_edges]
congestion_dataframe = pd.DataFrame(columns=['test', 'edge', 'method', 'value'])
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for edge in range(i_edges):
            s_row = pd.DataFrame([[i_test, edge, 'Proposed', congestions[i_test, edge].item()]],
                                 columns=['test', 'edge', 'method', 'value'])
            congestion_dataframe = congestion_dataframe.append(s_row)
            s_row = pd.DataFrame([[i_test, edge, 'Baseline', congestion_baseline[i_test, edge].item()]],
                                 columns=['test', 'edge', 'method', 'value'])
            congestion_dataframe = congestion_dataframe.append(s_row)

if N_tests >= 2:

    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y= 0.05/0.15, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax = sns.boxplot(x="edge", y="value", hue="method", data=congestion_dataframe, palette="muted")
    ax.grid(True)
else:
    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y= 0.05/0.15, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax.bar([k - 0.2 for k in range(congestions.size(1))], congestions.flatten(), width=0.4, align='center',
           label="Proposed")
    ax.bar([k + 0.2 for k in range(congestion_baseline.size(1))], congestion_baseline.flatten(), width=0.4,
           align='center',
           label="Naive")

plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'Congestion')
ax.set_ylim([-0.1, 1.5])

plt.show(block=False)
plt.savefig('congestion.png')

### Plot #3 : traversing time in tests vs. shortest path

print("Plotting cost comparison")
fig, ax = plt.subplots(figsize=(5 * 1.5, 3.6 * 1.5), layout='constrained')
## Computed cost function with exact expected congestion
# Generate vectors of length N that sum to 4
print("Computing exact cost... this will take a while")
k_all = generate_permutations(4, N_agents)
expected_sigma_4th = torch.zeros(N_tests, N_opt_var) # For consistency, include also the variables associated to nodes (with no congestion)
cost_exact = torch.zeros(N_tests,N_agents)
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for edge in road_graph.edges:
            for t in range(T_horiz):
                for i_perm in range(k_all.size(0)):
                    k = k_all[i_perm, :]
                    prod_p = 1
                    factor = multinomial_factor(4,k)
                    for index_agent in range(N_agents):
                        if k[index_agent]>0.001 : # if the corresponding factor is non-zero
                            if x[i_test, index_agent, edge_time_to_index[(edge, t)]].item() < 0.00001: # save on some computation if the current factor is 0
                                prod_p = 0
                                break
                            else:
                                prod_p = prod_p * x[i_test, index_agent, edge_time_to_index[(edge, t)]].item()
                    expected_sigma_4th[i_test, edge_time_to_index[(edge, t)]] = \
                        expected_sigma_4th[i_test, edge_time_to_index[(edge, t)]] + factor * prod_p

for i_agent in range(N_agents):
    for i_test in range(N_tests):
        if is_feasible[i_test]:
            for edge in road_graph.edges:
                for t in range(T_horiz):
                    congestion = road_graph[edge[0]][edge[1]]['travel_time'] * ( 1 +  0.15 * expected_sigma_4th[i_test, edge_time_to_index[(edge, t)]] / \
                                                                                 ((N_agents * road_graph[edge[0]][edge[1]]['capacity'])**4))
                    cost_partial = x[i_test, i_agent, edge_time_to_index[(edge, t)]] * congestion
                    cost_exact[i_test, i_agent] = cost_exact[i_test, i_agent] + cost_partial
print("Computed, plotting...")
##
costs_baseline = torch.zeros(N_tests, N_agents)
for i_test in range(N_tests):
    if is_feasible[i_test]:
        costs_baseline[i_test, :] = cost_baseline_stored[i_test]
costs_dataframe = pd.DataFrame(columns=['test', 'agent', 'value'])
for i_test in range(N_tests):
    if is_feasible[i_test]:
        for agent in range(N_agents):
            s_row = pd.DataFrame([[i_test, agent,
                                   (costs_baseline[i_test, agent].item() - cost_exact[i_test, agent].item()) / costs_baseline[
                                       i_test, agent].item()]], columns=['test', 'agent', 'value'])
            costs_dataframe = costs_dataframe.append(s_row)
if N_tests >= 2:
    ax = sns.boxplot(x="agent", y="value", data=costs_dataframe, palette="Set3")
    ax.set_ylim([-1, 1])
    ax.grid(True)
else:
    ax.bar([k for k in range(cost.size(1))], (costs_baseline.flatten() - cost_exact.flatten()) / costs_baseline.flatten(),
           width=0.4, align='center',
           label="Proposed")

ax.axhline(y=0, linewidth=1, color='red')

ax.set_ylabel(r'$J(x_b) - J(x^{\star})/J(x_b)$ ')

plt.legend(prop={'size': 8})
plt.show(block=False)
plt.savefig('cost.png')
print("Done")

### Plot #4 : expected value congestion error vs. # of agents

### Plot #4 : computed traversing time vs. # of agents
