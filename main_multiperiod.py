import numpy as np
import networkx as nx
import torch
import pickle
from FRB_algorithm import FRB_algorithm
from GameDefinition import Game
import time
import logging
from operator import itemgetter

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_graph = True
    N_random_tests = 30
    print("Initializing road graph...")
    if use_test_graph:
        N_agents=20   # N agents
        f = open('test_graph.pkl', 'rb')
        Road_graph, travel_time = pickle.load(f)
        f.close()
        T_horiz =8
    else:
        N_agents=60    # N agents
        f = open('manhattan_road_graph.pkl', 'rb')
        Road_graph, travel_time = pickle.load(f)
        f.close()
        T_horiz = 80

    n_juncs = len(Road_graph.nodes)
    print("Done")
    print("Running simulation with ", n_juncs," nodes and", N_agents," agents")

    np.random.seed(1)
    N_iter=300
    # containers for saved variables
    x_hsdm={}
    x_not_hsdm={}

    cost_hsdm={}
    cost_not_hsdm={}

    congestion_baseline={}
    cost_baseline={}

    initial_junctions_stored = {}
    final_destinations_stored = {}

    is_feasible = torch.zeros(N_random_tests, 1)

    visited_nodes = torch.zeros(N_random_tests, N_agents, T_horiz+1)

    for test in range(N_random_tests):
        # Change start-destination
        initial_junctions = np.random.randint(0, high=n_juncs, size=(N_agents))
        final_destinations = np.random.randint(0, high=n_juncs, size=(N_agents))
        # If there is no way from the starting point to the final, try a different start-goal pair
        for i in range(N_agents):
            while initial_junctions[i] == final_destinations[i] or not nx.has_path(Road_graph, initial_junctions[i], final_destinations[i]):
                initial_junctions[i]= np.random.randint(0, high=n_juncs - 1)
                final_destinations[i] = np.random.randint(0, high=n_juncs - 1)
        initial_junctions_stored.update({test:initial_junctions.copy()})
        final_destinations_stored.update({test:final_destinations.copy()})
        visited_nodes[test, :, 0] = torch.from_numpy(initial_junctions)
        print("Test " + str(test+1) + " out of " + str(N_random_tests))
        logging.info("Test " + str(test+1) + " out of " + str(N_random_tests) )
        for t in range(T_horiz):
            print("Initializing game for timestep " + str(t+1) + " out of " + str(T_horiz))
            logging.info("Initializing game for timestep " + str(t+1) + " out of " + str(T_horiz))
            game = Game(T_horiz-t, N_agents, Road_graph, initial_junctions, final_destinations, epsilon_probability=0.01)
            print("Done")
            alg = FRB_algorithm(game, beta=0.01, alpha=0.1, theta=0.25)
            status = alg.check_feasibility()
            if status != 'solved':
                print("The problem is not feasible, status: " + status + ", skipping test...")
                logging.info("The problem is not feasible, status: " + status + ", skipping test...")
                is_feasible[test, 0] = 0
                break
            else:
                print("The problem is feasible")
                logging.info("The problem is feasible")
                is_feasible[test, 0] = 1
            index_store = 0
            avg_time_per_it = 0
            for k in range(N_iter):
                start_time = time.time()
                alg.run_once()
                end_time = time.time()
                avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)
                if k % 10 == 0:
                    x, d, r, c  = alg.get_state()
                    print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                    logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                    index_store = index_store + 1
            x, d, r, c = alg.get_state()
            # Sample new state of the system
            for i_agent in range(N_agents):
                keys = [(k, 1) for k in range(n_juncs)]
                indexes = itemgetter(*keys)(game.node_time_to_index)
                # Probabilities need be >=0 and sum to 1, max and normalization are needed because of OSQP tolerance
                if torch.norm(torch.sum(torch.maximum(x[i_agent, indexes], torch.tensor([0]))) - 1) > 0.001:
                    print("Warning: probabilities do not sum to 1")
                    logging.info("Warning: probabilities do not sum to 1")
                prob_dist = torch.maximum(x[i_agent, indexes], torch.tensor([0])).flatten()  / torch.sum(torch.maximum(x[i_agent, indexes], torch.tensor([0])))
                initial_junctions[i_agent] = np.random.choice(range(n_juncs), p=prob_dist)
                visited_nodes[test, :, t+1] = torch.from_numpy(initial_junctions)
            if t==0:
                congestion_baseline_instance, cost_baseline_instance = game.compute_baseline() # Compute cost of naive shortest path
                congestion_baseline.update({test : congestion_baseline_instance})
                cost_baseline.update({test : cost_baseline_instance.flatten(0)})

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_multiperiod.pkl', 'wb')
    pickle.dump([ visited_nodes, Road_graph, game.edge_time_to_index, game.node_time_to_index, T_horiz, \
                 initial_junctions_stored, final_destinations_stored, congestion_baseline, cost_baseline, is_feasible], f)
    f.close
    print("Saved")
    logging.info("Saved, job done")