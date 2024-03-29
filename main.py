import numpy as np
import networkx as nx
import torch
import pickle
from FRB_algorithm import FRB_algorithm

from GameDefinition import Game
import time
import logging


def set_stepsizes(N, road_graph, A_ineq_shared, xi, algorithm='FRB'):
    theta = 0
    c = road_graph.edges[(0, 1)]['capacity']
    tau = road_graph.edges[(0, 1)]['travel_time']
    zeta = road_graph.edges[(0, 1)]['uncontrolled_traffic']
    k = 0.15 * tau / (c**xi)
    L = (2*k/N)* ((N+1) + (1 + zeta)**(xi-1) + (xi-1 * (1+zeta)**(xi-2)) )
    if algorithm == 'FRB':
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
    N_random_tests = 100
    xi = 4. # Exponent BPT function
    print("Initializing road graph...")
    if use_test_graph:
        N_agents=8 # N agents
        f = open('test_graph.pkl', 'rb')
        Road_graph = pickle.load(f)
        f.close()
        T_horiz =8
    else:
        N_agents=60    # N agents
        f = open('manhattan_road_graph.pkl', 'rb')
        Road_graph = pickle.load(f)
        f.close()
        T_horiz = 80

    n_juncs = len(Road_graph.nodes)
    print("Done")
    print("Running simulation with ", n_juncs," nodes and", N_agents," agents")

    np.random.seed(1)
    N_iter=100000
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
        is_problem_feasible = False
        attempt_create_feasible_problem = 0
        while  not is_problem_feasible:
            attempt_create_feasible_problem = attempt_create_feasible_problem+1
            # Change start-destination
            initial_junctions = np.random.randint(0, high=n_juncs, size=(N_agents))
            final_destinations = np.random.randint(0, high=n_juncs, size=(N_agents))
            # If there is no way from the starting point to the final, try a different start-goal pair
            for i in range(N_agents):
                while initial_junctions[i] == final_destinations[i] or not nx.has_path(Road_graph, initial_junctions[i], final_destinations[i]):
                    initial_junctions[i]= np.random.randint(0, high=n_juncs )
                    final_destinations[i] = np.random.randint(0, high=n_juncs )
            initial_junctions_stored.update({test:initial_junctions})
            final_destinations_stored.update({test:final_destinations})
            initial_state = torch.zeros(Road_graph.number_of_nodes(), N_agents) #initial prob. distribution (dirac delta, as it is deterministic)
            for i in range(N_agents):
                initial_state[initial_junctions[i], i] = 1
            ###############################
            print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
            logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
            game = Game(T_horiz, N_agents, Road_graph, initial_state, final_destinations, receding_horizon=False,
                        epsilon_probability=0.05, xi=xi)
            if test == 0:
                print("The game has " + str(N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                      + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
                logging.info("The game has " + str(N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                      + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
                # Initialize storing
                x_store = torch.zeros(N_random_tests, N_agents, game.n_opt_variables)
                dual_store = torch.zeros(N_random_tests, game.n_shared_ineq_constr)
                residual_store = torch.zeros(N_random_tests, N_iter // 1)
                cost_store = torch.zeros(N_random_tests, N_agents)
            print("Done")
            [alpha, beta, theta] = set_stepsizes(N_agents, Road_graph, game.A_ineq_shared, xi, algorithm='FRB')
            alg = FRB_algorithm(game, beta=beta, alpha= alpha, theta=theta)
            # alg = FBF_algorithm(game, beta=beta, alpha=alpha)
            status = alg.check_feasibility()
            is_problem_feasible = (status == 'solved')
            if not is_problem_feasible:
                print("the problem is not feasible: Test # " + str(test) + ", attempt # " + str(attempt_create_feasible_problem))
        index_store = 0
        avg_time_per_it = 0
        for k in range(N_iter):
            start_time = time.time()
            alg.run_once()
            end_time = time.time()
            avg_time_per_it = (avg_time_per_it * k + (end_time - start_time)) / (k + 1)
            if k % 1 == 0:
                x, d, r, c  = alg.get_state()
                residual_store[test, index_store] = r
                print("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                logging.info("Iteration " + str(k) + " Residual: " + str(r.item()) + " Average time: " + str(avg_time_per_it))
                index_store = index_store + 1
                if r <= 10 ** (-3):
                    break
        # store results
        x, d, r, c = alg.get_state()
        x_store[test, :, :] = x.flatten(1)
        dual_store[test, :] = d.flatten(0)
        cost_store[test, :] = c.flatten(0)
        # Compute baseline with shortest path
        congestion_baseline_instance, cost_baseline_instance = game.compute_baseline(initial_junctions, final_destinations) # Compute cost of naive shortest path
        congestion_baseline.update({test : congestion_baseline_instance})
        cost_baseline.update({test : cost_baseline_instance.flatten(0)})

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result.pkl', 'wb')
    pickle.dump([x_store, dual_store, residual_store, cost_store, Road_graph, game.edge_time_to_index, game.node_time_to_index, T_horiz, \
                 initial_junctions_stored, final_destinations_stored, congestion_baseline, cost_baseline], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


