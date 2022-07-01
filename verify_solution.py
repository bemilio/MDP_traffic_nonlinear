# Check validity of solution

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
eps = 1e-2
is_ok = True
# check if all out-edges for each node sum to the probability of coming from that node
# check if all in-edges for each node sum to the probability of arriving to that node
for i_test in range(N_tests):
        for i_agent in range(N_agents):
            for t in range(1, T_horiz): # There is no decision variable for the 1st timestep nodes
                for node in road_graph.nodes:
                    sum_in = 0
                    sum_out = 0
                    for edge in road_graph.in_edges(node):
                        sum_in=sum_in + x[i_test, i_agent, edge_time_to_index[(edge,t)]]
                    for edge in road_graph.out_edges(node):
                        sum_out=sum_out + x[i_test, i_agent, edge_time_to_index[(edge,t)]]
                    if torch.norm(x[i_test, i_agent, node_time_to_index[(node, t)]] - sum_out) > eps:
                        is_ok = False
                    if torch.norm(x[i_test, i_agent, node_time_to_index[(node, t+1)] ] - sum_in) > eps:
                        is_ok = False

# Check if edges connected to initial state sum to 1
for i_test in range(N_tests):
    for i_agent in range(N_agents):
        sum_edge = 0
        for edge in road_graph.out_edges(initial_junctions[i_test][i_agent]):
            sum_edge = sum_edge + x[i_test, i_agent, edge_time_to_index[edge, 0]]
        if torch.norm( sum_edge - 1) > eps:
            is_ok = False


#Check if nodes sum to 1
for i_test in range(N_tests):
        for i_agent in range(N_agents):
            for t in range(1, T_horiz): # There is no decision variable for the 1st timestep nodes
                sum_node = 0
                for node in road_graph.nodes:
                    sum_node = sum_node + x[i_test, i_agent, node_time_to_index[node, t]]
                if torch.norm( sum_node - 1) > eps:
                    is_ok = False

#Check if final destination node is approx. 1
for i_test in range(N_tests):
        for i_agent in range(N_agents):
            if torch.norm(x[i_test, i_agent, node_time_to_index[final_destinations[i_test][i_agent], T_horiz]]) < 1 - 0.05 - eps:
                is_ok = False


if is_ok:
    print("The solution seems ok")
else:
    print("Something's off")