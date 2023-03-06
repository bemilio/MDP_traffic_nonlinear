import torch
import numpy as np
import networkx as nx
from cmath import inf

torch.set_default_dtype(torch.float64)

class Game:
    # N: number of agents
    # final_destinations: vector of size N containing the destination of each agent
    # initial_state: matrix of size n_nodes x N. contains the initial probability distribution of each agent over the nodes.

    def __init__(self, T_horiz, N, road_graph, communication_graph, initial_state, final_destinations, receding_horizon=False,
                 epsilon_probability=0.05, xi=4, wardrop=False, penalized_agents=(0,)):
        self.N_agents = N
        self.epsilon_probability = epsilon_probability
        self.receding_horizon = receding_horizon
        self.add_terminal_cost = True if receding_horizon else False
        self.add_destination_constraint = False if receding_horizon else True
        self.edge_time_to_index = {}
        self.node_time_to_index = {}
        self.road_graph = road_graph
        self.final_destinations = final_destinations
        self.T_horiz = T_horiz
        self.xi = xi
        self.penalized_agents=penalized_agents
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
        self.A_ineq_loc, self.b_ineq_loc, self.A_eq_loc, self.b_eq_loc, self.index_soft_constraints = \
            self.define_local_constraints(T_horiz, N, road_graph, initial_state, final_destinations)
        self.A_ineq_loc = self.A_ineq_loc
        self.A_eq_loc  = self.A_eq_loc
        # Shared constraints
        self.A_ineq_shared, self.b_ineq_shared = \
            self.define_shared_constraints(T_horiz, N, road_graph)
        self.A_ineq_shared = self.A_ineq_shared
        self.n_shared_ineq_constr = self.A_ineq_shared.size(1)
        cost_SP = self.compute_cost_shortest_paths(road_graph, N, self.node_time_to_index, self.n_opt_variables, final_destinations, T_horiz) # Matrix N x n_x. Contains cost of Short path for each agent from each node (the variables x associated to edges have cost_SP 0). Used for terminal cost.
        if self.add_terminal_cost:
            weight_terminal_cost = self.compute_weight_terminal_cost(road_graph, N)
        else:
            weight_terminal_cost = 0.0
        # Define the (nonlinear) game mapping as a torch custom activation function
        self.F = self.GameMapping(self.n_opt_variables, road_graph, N, T_horiz, \
                                  self.edge_time_to_index, xi, weight_terminal_cost, cost_SP, wardrop)
        self.J = self.GameCost(self.n_opt_variables, road_graph, N, T_horiz, \
                               self.edge_time_to_index, xi, weight_terminal_cost, cost_SP)
        self.K = self.Consensus(communication_graph, self.n_shared_ineq_constr)
        # Define the selection function gradient (can be zero)
        self.nabla_phi = self.SelFunGrad(self.road_graph, N, T_horiz, self.edge_time_to_index, \
                                         self.n_opt_variables, self.penalized_agents)
        self.phi = self.SelFun(self.road_graph, N, T_horiz, self.edge_time_to_index, \
                               self.n_opt_variables, self.penalized_agents)

    class GameCost(torch.nn.Module):
        def __init__(self, n_opt_variables, road_graph, N, T_horiz, edge_time_to_index, xi, weight_terminal_cost, cost_SP):
            super().__init__()
            self.tau = torch.zeros(n_opt_variables, 1) # Stack all free-flow traversing time. To vectorize, vertices are treated the same as edges, but with cost 0.
            self.capacity = torch.ones(n_opt_variables, 1) # Stack of road (normalized) capacities. To vectorize, vertices are treated the same as edges. We initialize to 1 to avoid dividing by 0.
            self.uncontrolled_traffic = torch.ones(n_opt_variables, 1) # see above
            for t in range(T_horiz):
                for edge in road_graph.edges:
                    self.tau[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['travel_time']
                    self.capacity[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['capacity']
                    self.uncontrolled_traffic[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['uncontrolled_traffic']
            self.capacity_xith = torch.pow(self.capacity, xi)
            self.k = 0.15 * torch.div(self.tau, self.capacity_xith)
            self.N = N
            self.xi = xi
            self.weight_terminal_cost = weight_terminal_cost
            self.cost_SP = cost_SP

        def forward(self, x):
            sigma = torch.add(torch.sum(x, 0), self.uncontrolled_traffic) / self.N
            sigma_xith = torch.pow(sigma, self.xi)
            ell = torch.add(self.tau, torch.mul(self.k, sigma_xith))  # l(sigma) where l is capacity
            term_cost = self.weight_terminal_cost * torch.sum(torch.mul(self.cost_SP, x), 1).unsqueeze(2)
            return torch.add(torch.matmul(x.transpose(1,2), ell), term_cost)

    class GameMapping(torch.nn.Module):
        def __init__(self, n_opt_variables, road_graph, N, T_horiz, edge_time_to_index, xi, weight_terminal_cost, cost_SP, wardrop):
            super().__init__()
            self.tau = torch.zeros(n_opt_variables, 1) # Stack all free-flow traversing time. To vectorize, vertices are treated the same as edges, but with cost 0.
            self.capacity = torch.ones(n_opt_variables, 1) # Stack of road (normalized) capacities. To vectorize, vertices are treated the same as edges. We initialize to 1 to avoid dividing by 0.
            self.uncontrolled_traffic = torch.ones(n_opt_variables, 1) # see above
            self.xi = xi
            for t in range(T_horiz):
                for edge in road_graph.edges:
                    self.tau[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['travel_time']
                    self.capacity[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['capacity']
                    self.uncontrolled_traffic[edge_time_to_index[(edge, t)]] = road_graph[edge[0]][edge[1]]['uncontrolled_traffic']
            self.capacity_xith = torch.pow(self.capacity, xi)
            self.k = 0.15 * torch.div(self.tau, self.capacity_xith) # multiplicative factor in the capacity function
            self.N = N
            self.weight_terminal_cost = weight_terminal_cost
            self.cost_SP = cost_SP
            self.is_wardrop = wardrop


        def forward(self, x):
            sigma = torch.add(torch.sum(x, 0), self.uncontrolled_traffic) / self.N
            sigma_ximinone_th = torch.pow(sigma, self.xi-1)
            sigma_xith = torch.pow(sigma, self.xi)
            ell = torch.add( self.tau, torch.mul(self.k, sigma_xith)) # l(sigma) where l is capacity
            nabla_ell = torch.mul( torch.mul(self.k, sigma_ximinone_th) , self.xi) # \nabla l(sigma)
            # \nabla_{x_i} J_i = ell(sigma) + (x_i/N) \nabla_{sigma} ell(sigma)
            term_nabla = self.weight_terminal_cost * self.cost_SP
            if self.is_wardrop:
                return torch.add(ell, term_nabla)
            else:
                return torch.add(torch.add(ell, torch.mul(x/self.N, nabla_ell)), term_nabla)

        def get_strMon_Lip_constants(self):
            # TODO: find Lipschitz for Wardrop eq.
            # Return strong monotonicity and Lipschitz constant. See traffic paper, Prop. 2
            L = torch.max( torch.mul( (2*self.k/self.N),
                                  torch.add(torch.pow(1+self.uncontrolled_traffic,self.xi), \
                                            self.xi*torch.pow(1+self.uncontrolled_traffic,self.xi-1) ) ) )
            return 0, L

    def define_local_constraints(self, T_horiz, N, road_graph, initial_state, final_destinations):
        # Evolution constraint
        # sum_a M^t_{a->b} =rho^{t+1}_b

        E = road_graph.number_of_edges()
        V = road_graph.number_of_nodes()
        # Num of local constr: evolution (TV), M definition ((T-1)V), self loops ban (V-1)T, initial state (V)
        n_local_const_eq =  T_horiz * V + (T_horiz-1) * V + (V - 1) * T_horiz + V
        A_eq_loc_const = torch.zeros(N, n_local_const_eq, self.n_opt_variables)
        b_eq_loc_const = torch.zeros(N, n_local_const_eq, 1)
        # Num of local inequality constraints: Probability are positive (n_opt_variables), final constraint(1)
        n_local_const_ineq = self.n_opt_variables + (self.add_destination_constraint*1)
        A_ineq_loc_const = torch.zeros(N, n_local_const_ineq, self.n_opt_variables)
        b_ineq_loc_const = torch.zeros(N, n_local_const_ineq, 1)
        index_soft_constraints = torch.zeros(N, 1)
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
            # sum_b M^0_{x_0->b} = rho^{0}_a
            for junc in road_graph.nodes:
                for edge in road_graph.out_edges(junc):
                    A_eq_loc_const[i_agent, i_constr_eq, self.edge_time_to_index[(edge, 0)]] = 1
                b_eq_loc_const[i_agent, i_constr_eq, 0] = initial_state[junc, i_agent]
                i_constr_eq = i_constr_eq + 1

            ### Inequality constraints
            i_constr_ineq = 0
            # Probabilities are positive
            A_ineq_loc_const[i_agent, i_constr_ineq:i_constr_ineq + self.n_opt_variables, i_constr_ineq:i_constr_ineq + self.n_opt_variables] = \
                -torch.from_numpy(np.eye(self.n_opt_variables))
            i_constr_ineq = i_constr_ineq + self.n_opt_variables
            # Final state (Softened to avoid infeasibility)
            if self.add_destination_constraint:
                A_ineq_loc_const[
                    i_agent, i_constr_ineq, self.node_time_to_index[(final_destinations[i_agent], T_horiz)]] = -1
                b_ineq_loc_const[
                    i_agent, i_constr_ineq, 0] = - (1-self.epsilon_probability)
                index_soft_constraints[i_agent, 0] = i_constr_ineq
                i_constr_ineq = i_constr_ineq + 1
        return A_ineq_loc_const, b_ineq_loc_const, A_eq_loc_const, b_eq_loc_const, index_soft_constraints

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

    def compute_shortest_paths(self, initial_junctions, final_destinations):
        # compute congestion and cost function incurred by shortest paths
        shortest_paths = {}
        for i_agent in range(self.N_agents):
            shortest_paths.update({i_agent: nx.shortest_path(self.road_graph,  source=initial_junctions[i_agent],\
                             target = final_destinations[i_agent], weight='travel_time' )})
        return shortest_paths

    def compute_cost_shortest_paths(self, road_graph, N_agents, node_time_to_index, n_opt_variables, final_destinations, T_horiz):
        # Matrix N x n_x. Contains cost of Short path for each agent from each node (the variables x associated to edges have cost_SP 0). Used for terminal cost.
        K_SP = torch.zeros(N_agents, n_opt_variables)
        for a in road_graph.nodes:
            for i in range(N_agents):
                K_SP[i, node_time_to_index[(a, T_horiz)]]=nx.shortest_path_length(road_graph, source=a, target = final_destinations[i], weight='travel_time')
        return K_SP.unsqueeze(2)

    def compute_weight_terminal_cost(self, road_graph, N_agents):
        tau  = road_graph[0][1]['travel_time']
        capacity = road_graph[0][1]['capacity']
        k = 0.15 * tau/capacity
        weight = 1.1*(1 + (k*(N_agents+1))/(2*N_agents*tau))
        return weight

    def compute_baseline(self, initial_junctions, final_destinations):
        # compute congestion and cost function incurred by shortest paths
        shortest_paths = self.compute_shortest_paths(initial_junctions, final_destinations)
        sigma = {}
        for i_agent in range(self.N_agents):
            for t in range(len(shortest_paths[i_agent])-1):
                for edge in self.road_graph.edges:
                    if edge == (shortest_paths[i_agent][t], shortest_paths[i_agent][t+1]):
                        if (edge, t) in sigma.keys():
                            sigma.update({(edge, t): sigma[(edge,t)] + 1./self.N_agents})
                        else:
                            sigma.update({(edge, t): 1. / self.N_agents})
        cost_incurred = torch.zeros(self.N_agents, 1)
        for i_agent in range(self.N_agents):
            for t in range(len(shortest_paths[i_agent])-1):
                edge_taken = (shortest_paths[i_agent][t], shortest_paths[i_agent][t+1])
                capacity_edge = self.road_graph[edge_taken[0]][edge_taken[1]]['capacity']
                uncontrolled_traffic_edge = self.road_graph[edge_taken[0]][edge_taken[1]]['uncontrolled_traffic']
                cost_edge = self.road_graph[edge_taken[0]][edge_taken[1]]['travel_time'] * ( 1 + 0.15 * ( (sigma[(edge_taken,t)] + uncontrolled_traffic_edge)/capacity_edge)**self.xi )
                cost_incurred[i_agent] = cost_incurred[i_agent] + cost_edge
        return sigma, cost_incurred

    class Consensus(torch.nn.Module):
        def __init__(self, communication_graph, N_dual_variables):
            super().__init__()
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            L = nx.laplacian_matrix(communication_graph).tocoo()
            values = L.data
            rows = L.row
            cols = L.col
            indices = np.vstack((rows, cols))
            L = L.tocsr()
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            L_torch = torch.zeros(L.shape[0], L.shape[1], N_dual_variables, N_dual_variables)
            for i in rows:
                for j in cols:
                    L_torch[i, j, :, :] = L[i, j] * torch.eye(N_dual_variables)
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()
            self.L = L_torch

        def forward(self, dual):
            return torch.sum(torch.matmul(self.L, dual), dim=1)  # This applies the laplacian matrix to each of the dual variables

    class SelFunGrad(torch.nn.Module):
        def __init__(self, road_graph, N_agents, T_horiz, edge_time_to_index, n_opt_variables, penalized_agents):
            super().__init__()
            self.Q = torch.zeros(N_agents, n_opt_variables, n_opt_variables) # "block-diagonal" matrix, every i-th agent is associated a block
            self.c = torch.zeros(N_agents, n_opt_variables, 1)
            for i in range(N_agents):
                for t in range(T_horiz):
                    for edge in road_graph.edges:
                        if road_graph[edge[0]][edge[1]]['is_penalized_edge']>0 and i in penalized_agents:
                            self.Q[i, edge_time_to_index[(edge, t)], edge_time_to_index[(edge, t)]] = 1

        def forward(self, x):
            return torch.bmm(self.Q, x) + self.c

    class SelFun(torch.nn.Module):
        def __init__(self, road_graph, N_agents, T_horiz, edge_time_to_index, n_opt_variables, penalized_agents):
            super().__init__()
            self.Q = torch.zeros(N_agents, n_opt_variables, n_opt_variables) # "block-diagonal" matrix, every i-th agent is associated a block
            self.c = torch.zeros(N_agents, n_opt_variables, 1)
            for i in range(N_agents):
                for t in range(T_horiz):
                    for edge in road_graph.edges:
                        if road_graph[edge[0]][edge[1]]['is_penalized_edge'] and i in penalized_agents:
                            self.Q[i, edge_time_to_index[(edge, t)], edge_time_to_index[(edge, t)]] = 1

        def forward(self, x):
            # Cost is .5*x'Qx + c'x,  where Q is a block-diagonal matrix stored in a tensor N*N*n*n. The i,j-th block is in Q[i,j,:,:].
            return torch.sum(torch.bmm(x.transpose(1,2), .5*torch.bmm(self.Q, x) + self.c))

        def get_strMon_Lip_constants(self):
            # Return strong monotonicity and Lipschitz constant
            # Convert Q from block-diagonal matrix to standard matrix #TODO: turn block-diagonal matrix into separate class
            N = self.Q.size(0)
            n_x = self.Q.size(1)
            Q_mat = torch.zeros(N*n_x, N*n_x)
            for i in range(N):
                Q_mat[i*n_x:(i+1)*n_x, i*n_x:(j+1)*n_x] = self.Q[i,:,:]
            U,S,V = torch.linalg.svd(Q_mat)
            return torch.min(S).item(), torch.max(S).item()

