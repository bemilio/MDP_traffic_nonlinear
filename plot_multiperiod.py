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

f = open('/Users/ebenenati/surfdrive/TUDelft/Simulations/MDP_traffic_nonlinear/13_oct_22/saved_test_result_multiperiod.pkl', 'rb')
# f = open('saved_test_result_multiperiod.pkl', 'rb')
## Data structure:
## visited_nodes: dictionary with keys (index test, T_horiz) whose elements are tensors with dimension (N_agents, N_vehicles, T_horiz+1)
x_store, x_oneshot_store, visited_nodes, road_graph, edge_time_to_index_oneshot, node_time_to_index_oneshot, T_horiz_tested, T_simulation, \
initial_junctions_stored, final_destinations_stored, congestion_baseline_stored, cost_baseline, N_tests =  pickle.load(f)
f.close()

# f = open('realization_oneshot.pkl', 'wb')
# visited_nodes_oneshot, count_edge_taken_at_time_oneshot = pickle.load(f)

N_agents = visited_nodes[(0, T_horiz_tested[0])].size(0)
N_edges = road_graph.number_of_edges()
N_vertices = road_graph.number_of_nodes()
N_vehicles = visited_nodes[(0, T_horiz_tested[0])].size(1)

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

# Preliminaries: generate realization of probabilistic controller (one-shor solution, used for baseline)
visited_nodes_oneshot = torch.zeros((N_tests, N_agents, N_vehicles, T_simulation+1))
count_edge_taken_at_time_oneshot = {} # count how many vehicles pass by the edge at time t
for test in range(N_tests):
    for edge in road_graph.edges:
        for t in range(T_simulation):
            count_edge_taken_at_time_oneshot.update({(test, edge, t): 0})
for test in range(N_tests):
    for i_agent in range(N_agents):
        visited_nodes_oneshot[test, i_agent, :, 0] = initial_junctions_stored[test][i_agent] * torch.ones((N_vehicles, 1)).flatten()
    for t in range(T_simulation):
        for i_agent in range(N_agents):
            for vehicle in range(N_vehicles):
                starting_node = visited_nodes_oneshot[test, i_agent, vehicle, t].int().item()
                if t==0:
                    prob_starting_node=1 #probability of initial condition
                else:
                    prob_starting_node = max(x_oneshot_store[test, i_agent, node_time_to_index_oneshot[(starting_node, t)]], 0)
                if prob_starting_node < 0.00001:
                    print("Warning: a vehicle ended up in a very unlikely node")
                conditioned_prob = np.zeros(road_graph.number_of_nodes())
                for end_node in range(road_graph.number_of_nodes()):
                    if (starting_node, end_node) in road_graph.edges:
                        conditioned_prob[end_node] =  max(x_oneshot_store[test, i_agent, edge_time_to_index_oneshot[(( starting_node, end_node), t)] ], 0) / prob_starting_node
                    else:
                        conditioned_prob[end_node] = 0
                if sum(conditioned_prob) -1 >=0.0001:
                    print("Warning: conditioned prob. does not sum to 1, but to " + str(sum(conditioned_prob)))
                conditioned_prob = conditioned_prob/sum(conditioned_prob) #necessary just for numerical tolerancies
                next_visited_node = np.random.choice(range(road_graph.number_of_nodes()), p=conditioned_prob)
                visited_nodes_oneshot[test, i_agent, vehicle, t + 1] = next_visited_node
                count_edge_taken_at_time_oneshot.update( {( test, (starting_node, next_visited_node), t): count_edge_taken_at_time_oneshot[(test, (starting_node, next_visited_node), t)] +1 } )

# Check how many vehicles reached final dest.
total_vehicles_and_tests = len(T_horiz_tested) * N_tests * N_agents * N_vehicles
ratio_vehicles_reached_final_dest = 0
for T_horiz in T_horiz_tested:
    for test in range(N_tests):
        for agent in range(N_agents):
            ratio_vehicles_reached_final_dest = ratio_vehicles_reached_final_dest + \
                np.count_nonzero(visited_nodes[test, T_horiz][agent,:, -1] == final_destinations_stored[test][agent] ) / total_vehicles_and_tests
print("The percentage of vehicles that reached the final destination is " + str(ratio_vehicles_reached_final_dest*100))

# Plot how much time to reach destination
fig, ax = plt.subplots(figsize=(5 * 1, 3.6 * 1), layout='constrained')

N_datapoints = N_tests * len(T_horiz_tested) * N_agents * N_vehicles
list_tests = np.zeros(N_datapoints)
list_T_horiz = np.zeros(N_datapoints)
list_agents = np.zeros(N_datapoints)
list_vehicles = np.zeros(N_datapoints)
list_value = np.zeros(N_datapoints)
index_datapoint =0
for test in range(N_tests):
    for agent in range(N_agents):
        SP_length = nx.shortest_path_length(road_graph, source=initial_junctions_stored[test][agent],
                                target=final_destinations_stored[test][agent])
        for T_horiz in T_horiz_tested:
            for vehicle in range(N_vehicles):
                is_destination_reached = ((visited_nodes[(test, T_horiz)][agent, vehicle, :] == final_destinations_stored[test][agent]).nonzero().numel() >0)
                if is_destination_reached:
                    time_destination = \
                        (visited_nodes[(test, T_horiz)][agent, vehicle, :] == final_destinations_stored[test][agent]).nonzero()[0].item() - SP_length
                else:
                    time_destination = 10
                list_tests[index_datapoint] = test
                list_T_horiz[index_datapoint] = T_horiz
                list_agents[index_datapoint] = agent
                list_vehicles[index_datapoint] = vehicle
                list_value[index_datapoint] = time_destination
                index_datapoint = index_datapoint +1
time_destination_dataframe = pd.DataFrame(list(zip(list_tests, list_T_horiz, list_agents, list_vehicles, list_value)), columns=['test', 'T horiz', 'agent','vehicle', 'value'])
ax = sns.boxplot(x = "T horiz", y="value", data=time_destination_dataframe, palette="muted", medianprops={'color': 'lime', 'lw': 2})
ax.grid(True)
plt.show(block=False)

# Plot distance from  destination over time
fig, ax = plt.subplots(figsize=(5 * 1, 3.6 * 1), layout='constrained')
distance_from_dest_dataframe = pd.DataFrame()
SP = {}
for test in range(N_tests):
    for i in range(N_agents):
        SP.update({(test, i) : nx.shortest_path(road_graph, source=initial_junctions_stored[test][i], target=final_destinations_stored[test][i], weight='travel_time')})
cost_SP = torch.zeros(N_tests, N_agents, road_graph.number_of_nodes())
for test in range(N_tests):
    for a in road_graph.nodes:
        for i in range(N_agents):
            cost_SP[test, i, a]=nx.shortest_path_length(road_graph, source=a, target = final_destinations_stored[test][i])

N_datapoints = N_tests * len(T_horiz_tested) * T_simulation * N_agents * N_vehicles + T_simulation * N_tests * N_agents \
               + T_simulation * N_tests * N_agents * N_vehicles
list_tests = [None] *N_datapoints
list_T_horiz = [None] *N_datapoints
list_timestep = [None] *N_datapoints
list_agents = [None] *N_datapoints
list_vehicles = [None] *N_datapoints
list_value = [None] *N_datapoints
index_datapoint =0
for T_horiz in T_horiz_tested:
    for t in range(T_simulation):
        for test in range(N_tests):
            for i in range(N_agents):
                for vehicle in range(N_vehicles):
                    current_node = visited_nodes[(test, T_horiz)][i, vehicle, t]
                    distance_from_dest = cost_SP[test, i, int(current_node.item())]
                    list_tests[index_datapoint] = test
                    list_timestep[index_datapoint] = t
                    list_T_horiz[index_datapoint] = T_horiz
                    list_agents[index_datapoint] = i
                    list_vehicles[index_datapoint] = vehicle
                    list_value[index_datapoint] = distance_from_dest.item()
                    index_datapoint = index_datapoint +1
# Baseline 1: Shortest path
for t in range(T_simulation):
    for test in range(N_tests):
        for i in range(N_agents):
            if t < len(SP[(test,i)]):
                distance_baseline = cost_SP[test, i, SP[(test,i)][t]].item()
            else:
                distance_baseline = 0
            list_tests[index_datapoint] = test
            list_timestep[index_datapoint] = t
            list_T_horiz[index_datapoint] = "SP"
            list_agents[index_datapoint] = i
            list_vehicles[index_datapoint] = 0
            list_value[index_datapoint] = distance_baseline
            index_datapoint = index_datapoint + 1
# Baseline 2: one-shot
for t in range(T_simulation):
    for test in range(N_tests):
        for i in range(N_agents):
            for vehicle in range(N_vehicles):
                current_node = visited_nodes_oneshot[test, i, vehicle, t]
                distance_oneshot = cost_SP[test, i, int(current_node.item())]
                list_tests[index_datapoint] = test
                list_timestep[index_datapoint] = t
                list_T_horiz[index_datapoint] = "Single-shot"
                list_agents[index_datapoint] = i
                list_vehicles[index_datapoint] = 0
                list_value[index_datapoint] = distance_oneshot.item()
                index_datapoint = index_datapoint + 1
distance_from_dest_dataframe = pd.DataFrame(list(zip(list_tests, list_timestep, list_T_horiz, list_agents, list_vehicles, list_value)), columns=['test', 't', 'T horizon', 'agent','vehicle', 'Distance from endpoint'])

sns.lineplot(data = distance_from_dest_dataframe, ci=None , x='t', y='Distance from endpoint', hue='T horizon')
ax.set_ylim([0, 3])
ax.set_xlim([0, T_simulation - 1])

ax.grid(True)
ax.set_ylabel('Average distance from endpoint (N. of junctions)')
ax.set_xlabel('Timestep')
plt.savefig('1_multiperiod_average_distance.png')
plt.savefig('1_multiperiod_average_distance.pdf')

plt.show(block=False)

#### Plot how much cost incurred to reach destination
fig, ax = plt.subplots(figsize=(5 * 1, 3.6 * 1), layout='constrained')

# compute congestions
count_edge_taken_at_time = {} # count how many vehicles pass by the edge at time t
for T_horiz in T_horiz_tested:
    for test in range(N_tests):
        for edge in road_graph.edges:
            for t in range(T_simulation):
                count_edge_taken_at_time.update({(T_horiz, test, edge, t): 0})
for T_horiz in T_horiz_tested:
    for test in range(N_tests):
        for t in range(T_simulation):
            for edge in road_graph.edges:
                n_vehicles_on_edge_at_t = np.count_nonzero( torch.logical_and( (visited_nodes[(test, T_horiz)][:, :, t] == edge[0]), (visited_nodes[(test, T_horiz)][:, :, t+1] == edge[1]) ) )
                count_edge_taken_at_time.update({(T_horiz, test, edge, t): n_vehicles_on_edge_at_t})

N_datapoints = N_tests * len(T_horiz_tested) + N_tests
list_tests = np.zeros(N_datapoints)
list_T_horiz = [None] *N_datapoints
list_value = np.zeros(N_datapoints)
index_datapoint =0
for T_horiz in T_horiz_tested:
    for test in range(N_tests):
        social_cost=0
        social_cost_baseline=0
        for edge in road_graph.edges:
            for t in range(T_simulation):
                uncontrolled_traffic_edge = road_graph[edge[0]][edge[1]]['uncontrolled_traffic']
                sigma = (count_edge_taken_at_time[(T_horiz, test, edge, t)] / N_vehicles) / N_agents
                cost_edge = road_graph[edge[0]][edge[1]]['travel_time'] * (
                            1 + 0.15 * ((sigma + (uncontrolled_traffic_edge / N_agents)) / (road_graph[edge[0]][edge[1]]['capacity'])))
                if (edge, t) in congestion_baseline_stored[test].keys():
                    sigma_baseline = congestion_baseline_stored[test][(edge, t)]
                else:
                    sigma_baseline = 0
                cost_edge_baseline = road_graph[edge[0]][edge[1]]['travel_time'] * (
                        1 + 0.15 * ((sigma_baseline + (uncontrolled_traffic_edge / N_agents)) / (road_graph[edge[0]][edge[1]]['capacity'])))
                cost_partial = sigma * cost_edge
                cost_partial_baseline = sigma_baseline * cost_edge_baseline
                social_cost = social_cost + cost_partial
                social_cost_baseline = social_cost_baseline + cost_partial_baseline
        list_tests[index_datapoint] = test
        list_T_horiz[index_datapoint] = int(T_horiz)
        list_value[index_datapoint] = (social_cost_baseline - social_cost) / social_cost_baseline
        index_datapoint = index_datapoint +1
# Compute one shot social cost
for test in range(N_tests):
    social_cost_oneshot = 0
    social_cost_baseline = 0
    for edge in road_graph.edges:
        for t in range(T_simulation):
            sigma_oneshot = (count_edge_taken_at_time_oneshot[(test, edge, t)] / N_vehicles) / N_agents
            cost_edge_oneshot = road_graph[edge[0]][edge[1]]['travel_time'] * (
                    1 + 0.15 * ((sigma_oneshot + (uncontrolled_traffic_edge / N_agents)) / (
            road_graph[edge[0]][edge[1]]['capacity'])))
            if (edge, t) in congestion_baseline_stored[test].keys():
                sigma_baseline = congestion_baseline_stored[test][(edge, t)]
            else:
                sigma_baseline = 0
            cost_edge_baseline = road_graph[edge[0]][edge[1]]['travel_time'] * (
                    1 + 0.15 * ((sigma_baseline + (uncontrolled_traffic_edge / N_agents)) / (
            road_graph[edge[0]][edge[1]]['capacity'])))
            cost_partial_oneshot = sigma_oneshot * cost_edge_oneshot
            cost_partial_baseline = sigma_baseline * cost_edge_baseline
            social_cost_oneshot = social_cost_oneshot + cost_partial_oneshot
            social_cost_baseline = social_cost_baseline + cost_partial_baseline
    list_tests[index_datapoint] = test
    list_T_horiz[index_datapoint] = "Single-shot"
    list_value[index_datapoint] = (social_cost_baseline - social_cost_oneshot) / social_cost_baseline
    index_datapoint = index_datapoint + 1
social_cost_dataframe = pd.DataFrame(list(zip(list_tests,list_T_horiz, list_value)), columns=['test', 'T horiz', 'value'])
ax = sns.boxplot(x ="T horiz" ,y="value", data=social_cost_dataframe, palette="muted", medianprops={'color': 'lime', 'lw': 2})
ax.set_ylabel(r'$\frac{\sum_i J_i(x_b) - J_i(x^{\star})}{\sum_i J_i(x_b)}$ ')
ax.set_xlabel(r'Horizon')
ax.grid(True)
plt.savefig('2_multiperiod_advantage.png')
plt.savefig('2_multiperiod_advantage.pdf')
plt.show(block=False)


# Save dataframes

f = open('saved_multiperiod_dataframes.pkl', 'wb')
pickle.dump([distance_from_dest_dataframe, social_cost_dataframe], f)
f.close()

print("Done")
