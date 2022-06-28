from operators import backwardStep
import torch
from FRB_algorithm import FRB_algorithm


class TestGame:
    def __init__(self):
        # self.index_to_edge = dict
        self.N_agents = 2
        self.n_opt_variables = 1
        # Local constraints (dummies)
        self.A_ineq_loc = torch.zeros(self.N_agents, self.n_opt_variables,self.n_opt_variables)
        self.b_ineq_loc = torch.zeros(self.N_agents, self.n_opt_variables,1)
        self.A_ineq_loc[0,0,0] = -1 # x_1>=0
        self.A_ineq_loc[1, 0, 0] = -1 # x_2>=1
        self.b_ineq_loc[1, 0, 0] = 1
        self.A_eq_loc = torch.tensor(())
        self.b_eq_loc = torch.tensor(())
        # Shared constraints
        self.A_ineq_shared = torch.zeros(self.N_agents, 1, self.n_opt_variables)
        self.A_ineq_shared[0, 0, 0] = 0
        self.A_ineq_shared[1, 0, 0] = 0
        self.b_ineq_shared = torch.zeros(self.N_agents, 1, 1)
        self.n_shared_ineq_constr = self.A_ineq_shared.size(1)

        self.F = self.GameMapping()

    class GameMapping(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = torch.zeros(2, 1, 1)
            y[0, 0, 0] = 2*x[0, 0, 0] + x[1, 0, 0] + x[0, 0, 0] - 5
            y[1, 0, 0] = x[1, 0, 0]  + 2*x[1, 0, 0]
            return y


if __name__ == '__main__':

    ## Solve simple 2-players aggr. game without shared constraints
    game = TestGame()
    n_agents = 2
    n_vars = 1
    x0 = 10*torch.rand(n_agents,n_vars, 1)

    alg = FRB_algorithm(game, x_0=x0, dual_0=None, beta = 0.1, alpha = 0.1, theta= 0.1)
    for k in range (100):
        x_star, dual, residual = alg.get_state()
        print("Iteration " + str(k), " Residual: " + str(residual))
        alg.run_once()
    if torch.norm(x_star - torch.tensor([[[4./3.]], [[1.]] ] )):
        print("Test passed")
    else:
        print("test not passed!!")
        print(x_star)
