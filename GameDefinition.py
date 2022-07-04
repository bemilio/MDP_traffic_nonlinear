import torch
import numpy as np
import networkx as nx
from cmath import inf

torch.set_default_dtype(torch.float64)

class Game:
    def __init__(self, T_horiz, N, road_graph, initial_junctions, final_destinations, epsilon_probability=0.05):
        # self.index_to_edge = dict
        self.N_agents = N
        self.epsilon_probability = epsilon_probability
        self.edge_time_to_index = {}
        self.node_time_to_index = {}
        self.road_graph = road_graph
        self.initial_junctions = initial_junctions
        self.final_destinations = final_destinations
        self.T_horiz = T_horiz
        index_x = 0
        for edge in road_graph.edges:
            for t in range (T_horiz):
                self.edge_time_to_index.update({(edge, t) : index_x})
                index_x = index_x + 1
        for node in road_graph.nodes:
            for t in range (T_horiz):
                self.node_time_to_index.update({(node, t+1): index_x})
                index_x = index_x + 1
        self.n_opt_variables = T_horiz * (road_graph.number_of_edges() + road_graph.number_of_nodes())
        # Local constraints
        self.A_ineq_loc, self.b_ineq_loc, self.A_eq_loc, self.b_eq_loc = \
            self.define_local_constraints(T_horiz, N, road_graph, initial_junctions, final_destinations)
        self.A_ineq_loc = self.A_ineq_loc
        self.A_eq_loc  = self.A_eq_loc
        # Shared constraints
        self.A_ineq_shared, self.b_ineq_shared = \
            self.define_shared_constraints(T_horiz, N, road_graph)
        self.A_ineq_shared = self.A_ineq_shared
        self.n_shared_ineq_constr = self.A_ineq_shared.size(1)
        # Define the (nonlinear) game mapping as a torch custom activation function
        self.F = self.GameMapping(self.n_opt_variables, road_graph, N, T_horiz, self.edge_time_to_index)
        self.J = self.GameCost(self.n_opt_variables, road_graph, N, T_horiz, self.edge_time_to_index)

    class GameCost(torch.nn.Module):
        def __init__(self, n_opt_variables, road_graph, N, T_horiz, edge_time_to_index):
            super().__init__()
            self.tau = torch.zeros(n_opt_variables, 1) # Stack all free-flow traversing time. To vectorize, vertices are treated the same as edges, but with cost 0.
            self.capacity = torch.ones(n_opt_variables, 1) # Stack of road (normalized) capacities. To vectorize, vertices are treated the same as edges. We initialize to 1 to avoid dividing by 0.
            for t in range(T_horiz):
                for edge in road_graph.edges:
                    self.tau[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['travel_time']
                    self.capacity[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['capacity']
            self.capacity_4th = torch.pow(self.capacity, 4)
            self.k = 0.15 * torch.div(self.tau, self.capacity_4th)
            self.N = N

        def forward(self, x):
                sigma = torch.sum(x, 0) / self.N
                sigma_4th = torch.pow(sigma, 4)
                ell = torch.add(self.tau, torch.mul(self.k, sigma_4th))  # l(sigma) where l is capacity
                return torch.matmul(x.transpose(1,2), ell)

    class GameMapping(torch.nn.Module):
        def __init__(self, n_opt_variables, road_graph, N, T_horiz, edge_time_to_index):
            super().__init__()
            self.tau = torch.zeros(n_opt_variables, 1) # Stack all free-flow traversing time. To vectorize, vertices are treated the same as edges, but with cost 0.
            self.capacity = torch.ones(n_opt_variables, 1) # Stack of road (normalized) capacities. To vectorize, vertices are treated the same as edges. We initialize to 1 to avoid dividing by 0.
            for t in range(T_horiz):
                for edge in road_graph.edges:
                    self.tau[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['travel_time']
                    self.capacity[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['capacity']
            self.capacity_4th = torch.pow(self.capacity, 4)
            self.k = 0.15 * torch.div(self.tau, self.capacity_4th) # multiplicative factor in the capacity function
            self.N = N

        def forward(self, x):
            sigma = torch.sum(x,0)/self.N
            sigma_3rd = torch.pow(sigma, 3)
            sigma_4th = torch.pow(sigma, 4)
            ell = torch.add( self.tau, torch.mul(self.k, sigma_4th)) # l(sigma) where l is capacity
            nabla_ell = torch.mul( torch.mul(self.k, sigma_3rd) , 4/self.N) # \nabla l(sigma)
            # \nabla_{x_i} J_i = ell(sigma) + (x_i/N) \nabla_{sigma} ell(sigma)
            return torch.add(ell, torch.mul(x/self.N, nabla_ell))

    def define_local_constraints(self, T_horiz, N, road_graph, initial_junctions, final_destinations):
        # Evolution constraint
        # sum_a M^t_{a->b} =rho^{t+1}_b

        E = road_graph.number_of_edges()
        V = road_graph.number_of_nodes()
        # Num of local constr: evolution (TV), M definition ((T-1)V), self loops ban (V-1)T, initial state (2)
        n_local_const_eq =  T_horiz * V + (T_horiz-1) * V + (V - 1) * T_horiz + 2
        A_eq_loc_const = torch.zeros(N, n_local_const_eq, self.n_opt_variables)
        b_eq_loc_const = torch.zeros(N, n_local_const_eq, 1)
        # Num of local inequality constraints: Probability are positive (n_opt_variables), final constraint(1)
        n_local_const_ineq = self.n_opt_variables + 1
        A_ineq_loc_const = torch.zeros(N, n_local_const_ineq, self.n_opt_variables)
        b_ineq_loc_const = torch.zeros(N, n_local_const_ineq, 1)

        for junc in road_graph.nodes:
            if not road_graph.has_edge(junc, junc):
                raise ValueError('The provided graph must have self loops at every junction.')

        for i_agent in range(N):
            i_constr_eq = 0  # counter
            # Evolution constraint
            # sum_a M^t_{a->b} =rho^{t+1}_b
            for t in range(T_horiz):
                for junc in road_graph.nodes:
                    A_eq_loc_const[i_agent, i_constr_eq, self.node_time_to_index[(junc, t + 1)]] = 1
                    for edge in road_graph.in_edges(junc):
                        A_eq_loc_const[i_agent, i_constr_eq, self.edge_time_to_index[(edge, t)]] = -1
                    i_constr_eq = i_constr_eq + 1
            # Definition of M
            # sum_b M^t_{a->b} = rho^{t}_a
            for t in range(1, T_horiz): # the initial state is handled separately
                for junc in road_graph.nodes:
                    A_eq_loc_const[i_agent, i_constr_eq, self.node_time_to_index[(junc, t)]] = 1
                    for edge in road_graph.out_edges(junc):
                        A_eq_loc_const[i_agent, i_constr_eq, self.edge_time_to_index[(edge, t)]] = -1
                    i_constr_eq = i_constr_eq + 1
            # Allow self-loops only on destination
            for junc in road_graph.nodes:
                if not junc == final_destinations[i_agent]:
                    for t in range(T_horiz):
                        A_eq_loc_const[i_agent, i_constr_eq, self.edge_time_to_index[((junc, junc), t)]] = 1
                        i_constr_eq = i_constr_eq + 1
            # Initial state
            # sum_b M^0_{x_0->b} = 1
            for edge in road_graph.out_edges(initial_junctions[i_agent]):
                A_eq_loc_const[i_agent, i_constr_eq, self.edge_time_to_index[(edge, 0)]] = 1
            b_eq_loc_const[i_agent, i_constr_eq, 0] = 1
            i_constr_eq = i_constr_eq + 1
            # Initial state cont.
            # set to 0 all transition probabilities NOT from the initial state - this is needed
            for edge in road_graph.edges:
                if edge not in road_graph.out_edges(initial_junctions[i_agent]):
                    A_eq_loc_const[i_agent, i_constr_eq, self.edge_time_to_index[(edge, 0)]] = 1
            i_constr_eq = i_constr_eq + 1

            ### Inequality constraints
            i_constr_ineq = 0
            # Probabilities are positive
            A_ineq_loc_const[i_agent, i_constr_ineq:i_constr_ineq + self.n_opt_variables, i_constr_ineq:i_constr_ineq + self.n_opt_variables] = \
                -torch.from_numpy(np.eye(self.n_opt_variables))
            i_constr_ineq = i_constr_ineq + self.n_opt_variables
            # Final state
            A_ineq_loc_const[
                i_agent, i_constr_ineq, self.node_time_to_index[(final_destinations[i_agent], T_horiz)]] = -1
            b_ineq_loc_const[
                i_agent, i_constr_ineq, 0] = - (1-self.epsilon_probability)
            i_constr_ineq = i_constr_ineq + 1
        return A_ineq_loc_const, b_ineq_loc_const, A_eq_loc_const, b_eq_loc_const

    def define_shared_constraints(self, T_horiz, N, road_graph):
        # n shared constraints: capacity of roads (n_edges * T)
        # ignore edges with infinite capacity
        n_limited_edges = 0
        for edge in road_graph.edges:
            road_capacity = road_graph[edge[0]][edge[1]]['limit_roads']
            if road_capacity<inf:
                n_limited_edges = n_limited_edges+1
        n_shared_ineq_constr = n_limited_edges*T_horiz
        A_ineq_shared = torch.zeros(N, n_shared_ineq_constr, self.n_opt_variables)
        b_ineq_shared = torch.zeros(N, n_shared_ineq_constr, 1)
        i_constr = 0
        for t in range(T_horiz):
            for edge in road_graph.edges:
                road_capacity = road_graph[edge[0]][edge[1]]['limit_roads']
                if road_capacity < inf:
                    # since the capacity in the graph is actually c/N, we find:
                    # \sum M_(a,b) / N < c_(a,b) => \sum M_(a,b) < N c_(a,b)
                    # therefore, to write the constraint as
                    # \sum A_i x_i < \sum b_i
                    # A_i is a selection matrix and b_i = c_(a,b)
                    for i_agent in range(N):
                        A_ineq_shared[i_agent, i_constr, self.edge_time_to_index[(edge,t)]] = 1
                        b_ineq_shared[i_agent, i_constr, 0] = road_capacity
                    i_constr = i_constr + 1
        return A_ineq_shared, b_ineq_shared

    def compute_baseline(self):
        # compute congestion and cost function incurred by shortest paths
        shortest_paths = {}
        for i_agent in range(self.N_agents):
            shortest_paths.update({i_agent: nx.shortest_path(self.road_graph,  source=self.initial_junctions[i_agent],\
                             target = self.final_destinations[i_agent], weight='travel_time' )})
        congestion = {}
        for i_agent in range(self.N_agents):
            for t in range(len(shortest_paths[i_agent])-1):
                for edge in self.road_graph.edges:
                    if edge == (shortest_paths[i_agent][t], shortest_paths[i_agent][t+1]):
                        if (edge, t) in congestion.keys():
                            congestion.update({(edge, t): congestion[(edge,t)] + 1./self.N_agents})
                        else:
                            congestion.update({(edge, t): 1. / self.N_agents})
        cost_incurred = torch.zeros(self.N_agents, 1)
        for i_agent in range(self.N_agents):
            for t in range(len(shortest_paths[i_agent])-1):
                edge_taken = (shortest_paths[i_agent][t], shortest_paths[i_agent][t+1])
                capacity_edge = self.road_graph[edge_taken[0]][edge_taken[1]]['capacity']
                cost_edge = self.road_graph[edge_taken[0]][edge_taken[1]]['travel_time'] * ( 1 + 0.15 * (congestion[(edge_taken,t)]/capacity_edge)**4 )
                cost_incurred[i_agent] = cost_incurred[i_agent] + cost_edge
        return congestion, cost_incurred