from functools import total_ordering
from re import X
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from cmath import inf

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

f = open('saved_test_result.pkl', 'rb')
## Data structure:
## x: Tensor with dimension (n. random tests, N agents, n variables)
## dual: Tensor with dimension (n. random tests, n constraints, )
## Residual: Tensor with dimension (n. random tests, K iterations)
x, dual, residual, cost, road_graph, edge_time_to_index, node_time_to_index, T_horiz, initial_junctions, final_destinations, \
    congestion_baseline_stored, cost_baseline_stored = pickle.load(f)
f.close()

N_tests = x.size(0)
N_agents = x.size(1)
N_iter = residual.size(1)
N_edges = road_graph.number_of_edges()
N_vertices = road_graph.number_of_nodes()

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors


fig, ax = plt.subplots(figsize=(5, 1.8), layout='constrained') 
ax.loglog(np.arange(0, N_iter*10,10), residual[0,:])
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.legend()
plt.savefig('Residual.png')
plt.show()
plt.grid(True)

# draw graph
fig, ax = plt.subplots(T_horiz,2, figsize=(5, 10*1.4), layout='constrained')
pos=nx.get_node_attributes(road_graph, 'pos')
colors=[]
edgelist=[]
nodes=[]
for t in range(T_horiz):
    for edge in road_graph.edges:
        if not edge[0]==edge[1]:
            color_edge = np.matrix([0])
            congestion = torch.sum(x[0, :, edge_time_to_index[(edge,t)]], 0)/N_agents
            color_edge = congestion
            colors.append(int(206* color_edge)+50)
            edgelist.append(edge)
    for node in road_graph.nodes:
        nodes.append(node)
    pos = nx.kamada_kawai_layout(road_graph)
    plt.sca(ax[t,0])
    nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
    nx.draw_networkx_labels(road_graph, pos)
    nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist = edgelist, edge_cmap=plt.cm.Reds, connectionstyle='arc3, rad = 0.1')
# Draw baseline
for t in range(T_horiz):
    for edge in road_graph.edges:
        if not edge[0]==edge[1]:
            color_edge = np.matrix([0])
            if (edge, t) in congestion_baseline_stored[0].keys():
                congestion = congestion_baseline_stored[0][(edge, t)]
            else:
                congestion=0
            color_edge = congestion
            colors.append(int(206* color_edge)+50)
            edgelist.append(edge)
    for node in road_graph.nodes:
        nodes.append(node)
    pos = nx.kamada_kawai_layout(road_graph)
    plt.sca(ax[t,1])
    nx.draw_networkx_nodes(road_graph, pos=pos, nodelist=nodes, node_size=150, node_color='blue')
    nx.draw_networkx_labels(road_graph, pos)
    nx.draw_networkx_edges(road_graph, pos=pos, edge_color=colors, edgelist = edgelist, edge_cmap=plt.cm.Reds, connectionstyle='arc3, rad = 0.1')
plt.show(block=False)
plt.savefig('graph_plot.png')

# bar plot of maximum edge congestion compared to constraint
fig, ax = plt.subplots(figsize=(5*1.5, 3.6*1.5), layout='constrained')
congestions = torch.zeros( N_tests, N_edges )
congestion_baseline = torch.zeros( N_tests, N_edges )
for i_test in range(N_tests):
    i_edges = 0
    for edge in road_graph.edges:
        if not edge[0]==edge[1]:
            congestions[ i_test, i_edges ] = 0
            for t in range(T_horiz):
                relative_congestion = (torch.sum(x[i_test, :, edge_time_to_index[(edge, t)]], 0) / N_agents) \
                    / road_graph[edge[0]][edge[1]]['limit_roads']
                congestions[ i_test, i_edges ] = max(relative_congestion , congestions[ i_test, i_edges ])
                if (edge, t) in congestion_baseline_stored[i_test].keys():
                    relative_congestion_baseline = congestion_baseline_stored[i_test][ (edge, t) ]/ road_graph[edge[0]][edge[1]]['limit_roads']
                    congestion_baseline [ i_test, i_edges ] = max( relative_congestion_baseline, congestion_baseline[ i_test, i_edges ])
            i_edges = i_edges+1
congestions = congestions[:, 0:i_edges] # ignore self-loops
congestion_baseline = congestion_baseline[:, 0:i_edges]
congestion_dataframe = pd.DataFrame(columns=['test', 'edge', 'method', 'value'])
for test in range(N_tests):
    for edge in range(i_edges):
        s_row = pd.DataFrame([[test , edge, 'Proposed', congestions[test,edge].item()]], columns=['test', 'edge', 'method', 'value'])
        congestion_dataframe = congestion_dataframe.append(s_row)
        s_row = pd.DataFrame([[test , edge, 'Baseline', congestion_baseline[test,edge].item()]], columns=['test', 'edge', 'method', 'value'])
        congestion_dataframe = congestion_dataframe.append(s_row)

if N_tests>=2:
    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y=0.5, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax = sns.boxplot(x="edge", y="value", hue="method", data=congestion_dataframe, palette="Set3")
    ax.grid(True)
else:
    ax.axhline(y=1, linewidth=1, color='red', label="Road limit")
    ax.axhline(y=0.5, linewidth=1, color='orange', linestyle='--', label="Free-flow limit")
    ax.bar([k - 0.2 for k in range(congestions.size(1))], congestions.flatten(), width=0.4, align='center',
            label="Proposed")
    ax.bar([k + 0.2 for k in range(congestion_baseline.size(1))], congestion_baseline.flatten(), width=0.4, align='center',
            label="Naive")

plt.legend(prop={'size': 8})
ax.set_xlabel(r'Edge')
ax.set_ylabel(r'Congestion')
ax.set_ylim([-0.1, 1.2])

plt.show(block=False)
plt.savefig('congestion.png')

fig, ax = plt.subplots(figsize=(5*1.5, 3.6*1.5), layout='constrained')
costs_baseline = torch.zeros( N_tests, N_agents )
for i_test in range(N_tests):
    costs_baseline[i_test, :] = cost_baseline_stored[i_test]
costs_dataframe = pd.DataFrame(columns=['test', 'agent',  'value'])
for test in range(N_tests):
    for agent in range(N_agents):
        s_row = pd.DataFrame([[test, agent, (costs_baseline[test,agent].item() - cost[test,agent].item())/costs_baseline[test,agent].item()]], columns=['test', 'agent', 'value'])
        costs_dataframe = costs_dataframe.append(s_row)
if N_tests>=2:
    ax = sns.boxplot(x="agent", y="value", data=costs_dataframe, palette="Set3")
    ax.set_ylim([-1, 1])
    ax.grid(True)
else:
    ax.bar([k for k in range(cost.size(1))], (costs_baseline.flatten()-cost.flatten())/costs_baseline.flatten(), width=0.4, align='center',
            label="Proposed")

ax.axhline(y=0, linewidth=1, color='red')

ax.set_ylabel(r'$J(x_b) - J(x^{\star})/J(x_b)$ ')

plt.legend(prop={'size': 8})
plt.show(block=False)
plt.savefig('cost.png')
print("Done")
