import threading
import numpy as np
import networkx as nx
import matplotlib as mpl
import torch

mpl.interactive(True)
import pickle
from FRB_algorithm import FRB_algorithm
from GameDefinition import Game


if __name__ == '__main__':

    use_test_graph = True
    N_random_tests = 1
    print("Initializing road graph...")
    if use_test_graph:
        N_agents=10   # N agents
        f = open('test_graph.pkl', 'rb')
        Road_graph, nabla_travel_time = pickle.load(f)
        f.close()
        T_horiz =8
    else:
        N_agents=60    # N agents
        f = open('manhattan_road_graph.pkl', 'rb')
        Road_graph, nabla_travel_time = pickle.load(f)
        f.close()
        T_horiz = 80

    n_juncs = len(Road_graph.nodes)
    print("Done")
    print("Running simulation with ", n_juncs," nodes and", N_agents," agents")

    np.random.seed(1)
    N_iter=600
    stepsize_primal=0.01
    dual_stepsize = 0.01
    stepsize_hsdm = 0.02
    exponent_hsdm = 0.8
    # containers for saved variables
    x_hsdm={}
    x_not_hsdm={}

    cost_hsdm={}
    cost_not_hsdm={}


    for test in range(N_random_tests):
        # Change start-destination
        initial_junctions = np.random.randint(0, high=n_juncs, size=(N_agents))
        final_destinations = np.random.randint(0, high=n_juncs, size=(N_agents))
        # If there is no way from the starting point to the final, try a different start-goal pair
        for i in range(N_agents):
            while not nx.has_path(Road_graph, initial_junctions[i], final_destinations[i]):
                initial_junctions[i]= np.random.randint(0, high=n_juncs - 1)
                final_destinations[i] = np.random.randint(0, high=n_juncs - 1)
    ###############################
        print("Initializing game...")
        game = Game(T_horiz, N_agents, Road_graph, initial_junctions, final_destinations, epsilon_probability=0.05)
        if test == 0:
            # Initialize storing
            x_store = torch.zeros(N_random_tests, N_agents, game.n_opt_variables, N_iter // 10)
            dual_store = torch.zeros(N_random_tests, game.n_shared_ineq_constr, N_iter // 10)
            residual_store = torch.zeros(N_random_tests, N_iter // 10)
        print("Done")
        alg = FRB_algorithm(game, beta=0.01, alpha=0.01, theta=0.1)
        index_store = 0
        for k in range(N_iter):

            alg.run_once()
            if k % 10 == 0:
                x, d, r  = alg.get_state()
                x_store[test, :, :, index_store] = x.flatten(1)
                dual_store[test, :, index_store] = d.flatten(0)
                residual_store[test, index_store] = r
                print("Iteration " + str(k) + " Residual: " + str(r.item()))
                index_store = index_store + 1

    f = open('saved_test_result.pkl', 'wb')
    pickle.dump([x_store, dual_store, residual_store, Road_graph, game.edge_time_to_index, game.node_time_to_index], f)
    f.close