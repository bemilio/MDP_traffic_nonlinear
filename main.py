import numpy as np
import networkx as nx
import torch
import pickle
from FRB_algorithm import FRB_algorithm
from GameDefinition import Game
import time
import logging

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_graph = True
    N_random_tests = 2
    print("Initializing road graph...")
    if use_test_graph:
        N_agents=20  # N agents
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
    N_iter=3000
    # containers for saved variables
    x_hsdm={}
    x_not_hsdm={}

    cost_hsdm={}
    cost_not_hsdm={}

    congestion_baseline={}
    cost_baseline={}

    initial_junctions_stored = {}
    final_destinations_stored = {}

    for test in range(N_random_tests):
        # Change start-destination
        initial_junctions = np.random.randint(0, high=n_juncs, size=(N_agents))
        final_destinations = np.random.randint(0, high=n_juncs, size=(N_agents))
        # If there is no way from the starting point to the final, try a different start-goal pair
        for i in range(N_agents):
            while initial_junctions[i] == final_destinations[i] or not nx.has_path(Road_graph, initial_junctions[i], final_destinations[i]):
                initial_junctions[i]= np.random.randint(0, high=n_juncs - 1)
                final_destinations[i] = np.random.randint(0, high=n_juncs - 1)
        initial_junctions_stored.update({test:initial_junctions})
        final_destinations_stored.update({test:final_destinations})
    ###############################
        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        game = Game(T_horiz, N_agents, Road_graph, initial_junctions, final_destinations, epsilon_probability=0.05)
        if test == 0:
            print("The game has " + str(N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
            logging.info("The game has " + str(N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
            # Initialize storing
            x_store = torch.zeros(N_random_tests, N_agents, game.n_opt_variables)
            dual_store = torch.zeros(N_random_tests, game.n_shared_ineq_constr)
            residual_store = torch.zeros(N_random_tests, N_iter // 10)
            cost_store = torch.zeros(N_random_tests, N_agents)
            is_feasible = torch.zeros(N_random_tests,1)
        print("Done")
        alg = FRB_algorithm(game, beta=0.01, alpha=0.1, theta=0.25)
        status = alg.check_feasibility()
        if status != 'solved':
            print("the problem is not feasible, status: " + status + ", skipping test...")
            is_feasible[test,0] = 0
        else:
            print("the problem is feasible")
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
                    residual_store[test, index_store] = r
                    print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                    logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                    index_store = index_store + 1
            # store results
            x, d, r, c = alg.get_state()
            x_store[test, :, :] = x.flatten(1)
            dual_store[test, :] = d.flatten(0)
            cost_store[test, :] = c.flatten(0)

            congestion_baseline_instance, cost_baseline_instance = game.compute_baseline() # Compute cost of naive shortest path
            congestion_baseline.update({test : congestion_baseline_instance})
            cost_baseline.update({test : cost_baseline_instance.flatten(0)})
    # Check constraints satisfaction for last random test, last opt. step
    ineq_constr_eval = torch.bmm(game.A_ineq_loc, x_store[-1, :, :].unsqueeze(2)) - game.b_ineq_loc
    eq_constr_eval = torch.bmm(game.A_eq_loc, x_store[-1, :, :].unsqueeze(2)) - game.b_eq_loc

    tol = 0.001
    if torch.max(ineq_constr_eval) >= tol:
        print("Inequality constraints not satisfied by " + str(torch.max(ineq_constr_eval) ))
    if torch.max(torch.abs(eq_constr_eval)) >= tol:
        print("Equality constraints not satisfied by " + str(torch.max(torch.abs(eq_constr_eval)) ) )

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result.pkl', 'wb')
    pickle.dump([x_store, dual_store, residual_store, cost_store, Road_graph, game.edge_time_to_index, game.node_time_to_index, T_horiz, \
                 initial_junctions_stored, final_destinations_stored, congestion_baseline, cost_baseline, is_feasible], f)
    f.close
    print("Saved")
    logging.info("Saved, job done")