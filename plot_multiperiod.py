import matplotlib as mpl
import seaborn as sns
import pandas as pd

mpl.interactive(True)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
import networkx as nx
import numpy as np
import pickle
import torch
from Utilities.plot_agent_route import plot_agent_route
import os

#### Toggle between loading saved file in this directory or all files in a specific directory
load_files_from_current_dir = False
if load_files_from_current_dir:
    f = open('saved_test_result_multiperiod_0.pkl', 'rb')
    x_store, x_oneshot_store, visited_nodes, road_graph, edge_time_to_index_oneshot, node_time_to_index_oneshot, T_horiz_tested, T_simulation, \
        initial_junctions_stored, final_destinations_stored, congestion_baseline_stored, cost_baseline, N_tests,\
        T_road_blockage, edge_blockage_list, blockage_time_factor, blockage_capacity_factor = pickle.load(f)
    f.close()
else:
    # Load all files in a directory
    directory = r"C:\Users\ebenenati\surfdrive - Emilio Benenati@surfdrive.surf.nl2\TUDelft\Simulations\MDP_traffic_nonlinear\30_gen_24\Results"
    N_files = 0
    for filename in os.listdir(directory):
        if filename.find('.pkl')>=0:
            N_files=N_files+1 #count all files
    visited_nodes = {} #[test, T_horiz]
    initial_junctions_stored = {} #[test]
    final_destinations_stored = {} #[test]
    congestion_baseline_stored = {} #[test]
    N_tests=0
    for filename in os.listdir(directory):
        if filename.find('.pkl')>=0:
            f=open(directory+"\\"+filename, 'rb')
            x_store_file, x_oneshot_store_file, visited_nodes_file, road_graph, edge_time_to_index_oneshot, node_time_to_index_oneshot, T_horiz_tested, T_simulation, \
            initial_junctions_stored_file, final_destinations_stored_file, congestion_baseline_stored_file, cost_baseline_file, N_tests_file,\
                T_road_blockage, edge_blockage_list, blockage_time_factor, blockage_capacity_factor =  pickle.load(f)
            if N_tests ==0:
                x_oneshot_store= x_oneshot_store_file
            else:
                x_oneshot_store = torch.cat((x_oneshot_store, x_oneshot_store_file),dim=0)
            for index in range(N_tests_file):
                for T_hor in T_horiz_tested:
                    visited_nodes.update({(index + N_tests, T_hor): visited_nodes_file[(index,T_hor)]})
                initial_junctions_stored.update({index + N_tests: initial_junctions_stored_file[index]})
                final_destinations_stored.update({index + N_tests: final_destinations_stored_file[index]})
                congestion_baseline_stored.update({index + N_tests: congestion_baseline_stored_file[index]})
            N_tests = N_tests +N_tests_file
print("Files loaded, plotting...")

N_agents = visited_nodes[(0, T_horiz_tested[0])].size(0)
N_edges = road_graph.number_of_edges()
N_vertices = road_graph.number_of_nodes()
N_vehicles = visited_nodes[(0, T_horiz_tested[0])].size(1)
T_horiz_to_plot_1 = [1,8]
T_horiz_to_plot_2 = [1,3,5,8]
# T_horiz_to_plot_1 = [8]
# T_horiz_to_plot_2 = [8]

for T in T_horiz_to_plot_1:
    if T not in T_horiz_tested:
        raise("Requested plot of non existing horizon")
for T in T_horiz_to_plot_2:
    if T not in T_horiz_tested:
        raise ("Requested plot of non existing horizon")

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

# Plot distance from  destination over time
fig, ax = plt.subplots(figsize=(4 * 1, 2.1 * 1), layout='constrained')
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

N_datapoints = N_tests * len(T_horiz_to_plot_1) * T_simulation * N_agents * N_vehicles + T_simulation * N_tests * N_agents \
               + T_simulation * N_tests * N_agents * N_vehicles
list_tests = [None] *N_datapoints
list_T_horiz = [None] *N_datapoints
list_timestep = [None] *N_datapoints
list_agents = [None] *N_datapoints
list_vehicles = [None] *N_datapoints
list_value = [None] *N_datapoints
list_is_baseline = [None] *N_datapoints
list_is_baseline_2= [None] * (len(T_horiz_to_plot_1) + 2)
index_datapoint =0
index_horizon = 0
for T_horiz in T_horiz_to_plot_1:
    list_is_baseline_2[index_horizon]=T_horiz
    index_horizon = index_horizon+1
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
                    list_is_baseline[index_datapoint] = False
                    index_datapoint = index_datapoint +1
# Baseline 1: one-shot
list_is_baseline_2[index_horizon]=r'$\infty$'#"Open loop"
index_horizon = index_horizon+1
for t in range(T_simulation):
    for test in range(N_tests):
        for i in range(N_agents):
            for vehicle in range(N_vehicles):
                current_node = visited_nodes_oneshot[test, i, vehicle, t]
                distance_oneshot = cost_SP[test, i, int(current_node.item())]
                list_tests[index_datapoint] = test
                list_timestep[index_datapoint] = t
                list_T_horiz[index_datapoint] = r'$\infty$'#"Open loop"
                list_agents[index_datapoint] = i
                list_vehicles[index_datapoint] = 0
                list_value[index_datapoint] = distance_oneshot.item()
                list_is_baseline[index_datapoint] = True
                index_datapoint = index_datapoint + 1
# Baseline 2: Shortest path
list_is_baseline_2[index_horizon]="SP"
index_horizon = index_horizon+1
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
            list_is_baseline[index_datapoint] = True
            index_datapoint = index_datapoint + 1

distance_from_dest_dataframe = pd.DataFrame(list(zip(list_tests, list_timestep, list_T_horiz, list_agents, list_vehicles, list_value, list_is_baseline)),
                                            columns=['test', 't', 'T horizon', 'agent','vehicle', 'Distance from endpoint', 'Baseline'])
dic_dashes = {False:'', True:(2,2)}
sns.lineplot(data=distance_from_dest_dataframe, drawstyle='steps-pre', ci=None , x='t', palette=['b','lime', 'grey', 'r'],
             y='Distance from endpoint', hue='T horizon', style=list_is_baseline, dashes=dic_dashes, linewidth = 3.0)
plt.legend(labels=list_is_baseline_2)
ax.set_ylim([0, 2.3])
ax.set_xlim([0, T_simulation - 1])

ax.grid(True)
ax.set_ylabel("\# nodes to destination (avg.)", fontsize = 9)
ax.set_xlabel('timestep', fontsize = 9)
fig.savefig('Figures/1_multiperiod_average_distance.png')
fig.savefig('Figures/1_multiperiod_average_distance.pdf')

plt.show(block=False)

#### Plot how much cost incurred to reach destination
fig, ax = plt.subplots(figsize=(4, 4.5), layout='constrained')

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
for T_horiz in T_horiz_to_plot_2:
    for test in range(N_tests):
        social_cost=0
        social_cost_baseline=0
        for edge in road_graph.edges:
            for t in range(T_simulation):
                uncontrolled_traffic_edge = road_graph[edge[0]][edge[1]]['uncontrolled_traffic']
                sigma = (count_edge_taken_at_time[(T_horiz, test, edge, t)] / N_vehicles) / N_agents
                if t>=T_road_blockage and edge in edge_blockage_list:
                    travel_time = blockage_time_factor * road_graph[edge[0]][edge[1]]['travel_time']
                    capacity = blockage_capacity_factor * road_graph[edge[0]][edge[1]]['capacity']
                else:
                    travel_time = road_graph[edge[0]][edge[1]]['travel_time']
                    capacity = road_graph[edge[0]][edge[1]]['capacity']
                cost_edge = travel_time * (1 + 0.15 * ((sigma + (uncontrolled_traffic_edge / N_agents)) / capacity))
                if (edge, t) in congestion_baseline_stored[test].keys():
                    sigma_baseline = congestion_baseline_stored[test][(edge, t)]
                else:
                    sigma_baseline = 0
                cost_edge_baseline = travel_time * (1 + 0.15 * ((sigma_baseline + (uncontrolled_traffic_edge / N_agents)) / capacity))
                cost_partial = sigma * cost_edge
                cost_partial_baseline = sigma_baseline * cost_edge_baseline
                social_cost = social_cost + cost_partial
                social_cost_baseline = social_cost_baseline + cost_partial_baseline
        list_tests[index_datapoint] = test
        list_T_horiz[index_datapoint] = int(T_horiz)
        list_value[index_datapoint] = (social_cost - social_cost_baseline ) / social_cost_baseline
        index_datapoint = index_datapoint + 1
# Compute one shot social cost
for test in range(N_tests):
    social_cost_oneshot = 0
    social_cost_baseline = 0
    for edge in road_graph.edges:
        for t in range(T_simulation):
            if t >= T_road_blockage and edge in edge_blockage_list:
                travel_time = blockage_time_factor * road_graph[edge[0]][edge[1]]['travel_time']
                capacity = blockage_capacity_factor * road_graph[edge[0]][edge[1]]['capacity']
            else:
                travel_time = road_graph[edge[0]][edge[1]]['travel_time']
                capacity = road_graph[edge[0]][edge[1]]['capacity']
            sigma_oneshot = (count_edge_taken_at_time_oneshot[(test, edge, t)] / N_vehicles) / N_agents
            cost_edge_oneshot = travel_time * (1 + 0.15 * ((sigma_oneshot + (uncontrolled_traffic_edge / N_agents))/capacity))
            if (edge, t) in congestion_baseline_stored[test].keys():
                sigma_baseline = congestion_baseline_stored[test][(edge, t)]
            else:
                sigma_baseline = 0
            cost_edge_baseline = travel_time * (1 + 0.15 * ((sigma_baseline + (uncontrolled_traffic_edge / N_agents)) / capacity))
            cost_partial_oneshot = sigma_oneshot * cost_edge_oneshot
            cost_partial_baseline = sigma_baseline * cost_edge_baseline
            social_cost_oneshot = social_cost_oneshot + cost_partial_oneshot
            social_cost_baseline = social_cost_baseline + cost_partial_baseline
    list_tests[index_datapoint] = test
    list_T_horiz[index_datapoint] = r'$\infty$'#"Open loop"
    list_value[index_datapoint] = (social_cost_oneshot - social_cost_baseline) / social_cost_baseline
    index_datapoint = index_datapoint + 1
social_cost_dataframe = pd.DataFrame(list(zip(list_tests,list_T_horiz, list_value)), columns=['test', 'T horiz', 'value'])
ax = sns.boxplot(x ="T horiz" ,y="value", data=social_cost_dataframe, palette=['b', 'c', 'm', 'lime', 'grey'], medianprops={'color': 'k', 'lw': 2})
ax.set_ylabel(r'$\frac{\sum_i J_i(x^{*})-J_i(x_{SP})}{\sum_i J_i(x_{SP})}$ ')
ax.set_xlabel(r'horizon')
ax.set_yticklabels([str(int(k*100))+r'\%' if k!=0 else r'$\pm$'+str(int(k*100))+r'\%' for k in ax.get_yticks()])
ax.grid(True)
fig.savefig('Figures/2_multiperiod_advantage.png')
fig.savefig('Figures/2_multiperiod_advantage.pdf')
plt.show(block=False)


# Save dataframes

f = open('saved_multiperiod_dataframes.pkl', 'wb')
pickle.dump([distance_from_dest_dataframe, social_cost_dataframe, list_is_baseline], f)
f.close()

print("Done")
