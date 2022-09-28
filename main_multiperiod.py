import numpy as np
import networkx as nx
import torch
import pickle
from FRB_algorithm import FRB_algorithm
from GameDefinition import Game
import time
import logging
from operator import itemgetter

def set_stepsizes(N, road_graph, A_ineq_shared, algorithm='FRB'):
    theta = 0
    c = road_graph.edges[(0, 1)]['capacity']
    tau = road_graph.edges[(0, 1)]['travel_time']
    a = road_graph.edges[(0, 1)]['uncontrolled_traffic']
    k = 0.15 * 12 * tau / c
    L = k* (N+1) + (1 + c/tau) * (tau*road_graph.number_of_nodes())
    if algorithm == 'FRB':
        # L = 2*k/(4*N) * (N+a)**3 + k/N * (N+a)**2 * (N + (N/3) * (N+a) + \
        #                      np.sqrt( ((N/3)**2) * (N+a)**2  + (2*(N**2)/3) * (N+a) + N ) )
        delta = 2*L / (1-3*theta)
        eigval, eigvec = torch.linalg.eig(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)))
        eigval = torch.real(eigval)
        alpha = 0.5/((torch.max(torch.max(eigval, 1)[0])) + delta)
        beta = N * 0.5/(torch.sum(torch.max(eigval, 1)[0]) + delta)
    if algorithm == 'FBF':
        eigval, eigvec = torch.linalg.eig(torch.sum(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)), 0)  )
        eigval = torch.real(eigval)
        alpha = 0.5/(L+torch.max(eigval))
        beta = 0.5/(L+torch.max(eigval))
    return (alpha.item(), beta.item(), theta)

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_graph = True
    N_random_tests = 3
    N_vehicles_per_agent = 100
    print("Initializing road graph...")
    N_agents=8   # N agents
    f = open('test_graph.pkl', 'rb')
    Road_graph = pickle.load(f)
    f.close()
    T_horiz_to_test= [1, 3, 5]
    T_simulation=8

    n_juncs = len(Road_graph.nodes)
    print("Done")
    print("Running simulation with ", n_juncs," nodes and", N_agents," agents")

    np.random.seed(1)
    N_iter=10000
    # containers for saved variables
    x_hsdm={}
    x_not_hsdm={}

    cost_hsdm={}
    cost_not_hsdm={}

    congestion_baseline={}
    cost_baseline={}

    initial_junctions_stored = {}
    final_destinations_stored = {}

    x_store = {}
    visited_nodes = {}
    for test in range(N_random_tests):
        # Change start-destination
        initial_junctions = np.random.randint(0, high=n_juncs, size=(N_agents))
        final_destinations = np.random.randint(0, high=n_juncs, size=(N_agents))
        # If there is no way from the starting point to the final, try a different start-goal pair
        for i in range(N_agents):
            while initial_junctions[i] == final_destinations[i] or not nx.has_path(Road_graph, initial_junctions[i], final_destinations[i]):
                initial_junctions[i]= np.random.randint(0, high=n_juncs)
                final_destinations[i] = np.random.randint(0, high=n_juncs)
        initial_junctions_stored.update({test:initial_junctions.copy()})
        final_destinations_stored.update({test:final_destinations.copy()})
        print("Initializing game for test " + str(test) + " out of " + str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " + str(N_random_tests))
        ### Begin tests
        for T_horiz in T_horiz_to_test:
            # Create initial prob. distribution
            initial_state = torch.zeros(Road_graph.number_of_nodes(), N_agents)
            for i in range(N_agents):
                initial_state[initial_junctions[i], i] = 1
            for t in range(T_simulation):
                print("Initializing game for timestep " + str(t+1) + " out of " + str(T_simulation))
                logging.info("Initializing game for timestep " + str(t+1) + " out of " + str(T_simulation))
                game = Game(T_horiz, N_agents, Road_graph, initial_state, final_destinations, add_terminal_cost=True,
                            add_destination_constraint=False, xi=1)
                if t==0:
                    print("The game has " + str(N_agents) + " agents; " + str(
                        game.n_opt_variables) + " opt. variables per agent; " \
                          + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(
                        game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(
                        game.n_shared_ineq_constr) + " shared ineq. constraints")
                    logging.info("The game has " + str(N_agents) + " agents; " + str(
                        game.n_opt_variables) + " opt. variables per agent; " \
                                 + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(
                        game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(
                        game.n_shared_ineq_constr) + " shared ineq. constraints")
                    # Initialize storing
                    x_store.update({(test, T_horiz) : torch.zeros(N_agents, game.n_opt_variables, T_simulation)})
                    visited_nodes.update({(test, T_horiz) : torch.zeros((N_agents, N_vehicles_per_agent, T_simulation + 1))})
                print("Done")
                visited_nodes[(test,T_horiz)][:, :, 0] = torch.from_numpy(initial_junctions).unsqueeze(1).repeat(1,N_vehicles_per_agent)
                [alpha, beta, theta] = set_stepsizes(N_agents, Road_graph, game.A_ineq_shared, algorithm='FRB')
                alg = FRB_algorithm(game, beta=beta, alpha=alpha, theta=theta)
                status = alg.check_feasibility()
                ### Check feasibility
                is_problem_feasible = (status == 'solved')
                if not is_problem_feasible:
                    print("the problem is not feasible: Test # " + str(test) + "In multiperiod optimization, this should not happen - increase the road capacity!")
                index_store = 0
                avg_time_per_it = 0
                ### Main iterations
                for k in range(N_iter):
                    start_time = time.time()
                    alg.run_once()
                    end_time = time.time()
                    avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)
                    if k % 10 == 0:
                        x, d, r, c = alg.get_state()
                        print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(
                            avg_time_per_it))
                        logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(
                            avg_time_per_it))
                        index_store = index_store + 1
                        if r <= 10 ** (-1):
                            break
                # store results
                x, d, r, c = alg.get_state()
                # Sample new state of the system
                for i_agent in range(N_agents):
                    for vehicle in range(N_vehicles_per_agent):
                        starting_node = visited_nodes[(test,T_horiz)][i_agent, vehicle, t].int().item()
                        prob_starting_node = initial_state[starting_node, i_agent]
                        conditioned_prob = np.zeros(Road_graph.number_of_nodes())
                        for end_node in range(Road_graph.number_of_nodes()):
                            if (starting_node, end_node) in Road_graph.edges:
                                conditioned_prob[end_node] = max(
                                    x[i_agent, game.edge_time_to_index[((starting_node, end_node), 0)]],
                                    0) / prob_starting_node
                            else:
                                conditioned_prob[end_node] = 0
                        conditioned_prob = conditioned_prob / sum(
                            conditioned_prob)  # necessary just for numerical tolerancies
                        next_visited_node = np.random.choice(range(Road_graph.number_of_nodes()), p=conditioned_prob)
                        visited_nodes[(test,T_horiz)][i_agent, vehicle, t + 1] = next_visited_node
                    for node in range(Road_graph.number_of_nodes()):
                        initial_state[node, i_agent] = np.count_nonzero(visited_nodes[(test,T_horiz)][i_agent,:,t+1] == node) / N_vehicles_per_agent
                x_store[(test,T_horiz)][:, :, t] = x.flatten(1)
                if t==0 and T_horiz==T_horiz_to_test[0]:
                    congestion_baseline_instance, cost_baseline_instance = game.compute_baseline(initial_junctions, final_destinations) # Compute cost of naive shortest path
                    congestion_baseline.update({test : congestion_baseline_instance})
                    cost_baseline.update({test : cost_baseline_instance.flatten(0)})

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_multiperiod.pkl', 'wb')
    pickle.dump([ x_store, visited_nodes, Road_graph, game.edge_time_to_index, game.node_time_to_index, T_horiz_to_test, T_simulation, \
                 initial_junctions_stored, final_destinations_stored, congestion_baseline, cost_baseline, N_random_tests], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")
