import torch
from operators import backwardStep
import numpy as np

torch.set_default_dtype(torch.float64)


# For aggregative monotone games, See Belgioioso 2021
class FBF_algorithm:
    def __init__(self, game, x_0=None, dual_0=None, beta = 0.1, alpha = 0.1):
        self.game = game
        self.alpha = alpha
        self.beta = beta
        N_agents = game.N_agents
        n = game.n_opt_variables
        m = game.n_shared_ineq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = torch.zeros(N_agents, n, 1)
        if dual_0:
            self.dual = dual_0
        else:
            self.dual = torch.zeros(m, 1)
        self.x_last = self.x
        self.dual_last = self.dual
        Q = torch.zeros(N_agents, n, n) # Local cost is zero
        q = torch.zeros(N_agents, n, 1)
        self.prox = backwardStep.BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc, \
                                              self.alpha, index_soft_constraints=game.index_soft_constraints)
        self.projection = backwardStep.BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,
                                              1, index_soft_constraints=game.index_soft_constraints)

    def run_once(self):
        x = self.x
        dual = self.dual
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        # Primal update 1
        y = x - self.alpha*(self.game.F(x)  + torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        x_tilde, status = self.projection(y)
        # Dual update 1
        d_tilde = torch.bmm(A_i, x) - b_i
        d_tilde_avg = torch.sum(d_tilde, 0) / self.game.N_agents
        lambda_tilde = torch.maximum( dual + self.beta * d_tilde_avg , torch.zeros(dual.size()))
        # Primal update 2
        r = x_tilde - self.alpha*( self.game.F(x) + torch.matmul(torch.transpose(A_i, 1, 2), lambda_tilde )  )
        x_new, status = self.projection(x - y + r)
        self.x = x_new
        d = torch.bmm(A_i, x_tilde) - b_i
        d_avg = torch.sum(d, 0) / self.game.N_agents
        # Dual update 2
        dual_new = torch.maximum( dual + self.beta * (d_avg - d_tilde_avg), torch.zeros(dual.size()) )
        self.dual = dual_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        return self.x, self.dual, residual, cost

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x_transformed, status = self.projection(self.x-self.game.F(self.x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        torch.sum(torch.bmm(A_i, self.x) - b_i, 0)
        dual_transformed = torch.maximum(self.dual + torch.sum(torch.bmm(A_i, self.x) - b_i, 0), torch.zeros(self.dual.size()))
        residual = np.sqrt( ((self.x - x_transformed).norm())**2 + ((self.dual-dual_transformed).norm())**2 )
        return residual

    def check_feasibility(self):
        n_local_ineq_constr = self.game.A_ineq_loc.size(1)
        n_local_eq_constr = self.game.A_eq_loc.size(1)
        A_ineq_all = torch.zeros(
            (1, self.game.N_agents * n_local_ineq_constr + self.game.n_shared_ineq_constr, self.game.N_agents * self.game.n_opt_variables))
        b_ineq_all = torch.zeros((1, self.game.N_agents * n_local_ineq_constr + self.game.n_shared_ineq_constr, 1))
        A_eq_all = torch.zeros(
            (1, self.game.N_agents * n_local_eq_constr + self.game.n_shared_ineq_constr, self.game.N_agents * self.game.n_opt_variables))
        b_eq_all = torch.zeros((1, self.game.N_agents * n_local_eq_constr + self.game.n_shared_ineq_constr, 1))
        for i in range(self.game.N_agents):
            A_ineq_all[0,i * n_local_ineq_constr:(i + 1) * n_local_ineq_constr,
                i * self.game.n_opt_variables:(i + 1) * self.game.n_opt_variables] = self.game.A_ineq_loc[i, :, :]
            b_ineq_all[0,i * n_local_ineq_constr:(i + 1) * n_local_ineq_constr, :] = self.game.b_ineq_loc[i, :, :]
            A_eq_all[0,i * n_local_eq_constr:(i + 1) * n_local_eq_constr,
                i * self.game.n_opt_variables:(i + 1) * self.game.n_opt_variables] = self.game.A_eq_loc[i, :, :]
            b_eq_all[0,i * n_local_eq_constr:(i + 1) * n_local_eq_constr, :] = self.game.b_eq_loc[i, :, :]
            A_ineq_all[0,-self.game.n_shared_ineq_constr:,
                i * self.game.n_opt_variables:(i + 1) * self.game.n_opt_variables] = self.game.A_ineq_shared[i, :, :]
            b_ineq_all[0,-self.game.n_shared_ineq_constr:, :] = b_ineq_all[0,-self.game.n_shared_ineq_constr:,
                                                              :] + self.game.b_ineq_shared[i, :, :]
        Q = torch.zeros(1, self.game.N_agents * self.game.n_opt_variables, self.game.N_agents * self.game.n_opt_variables)
        q = torch.zeros(1, self.game.N_agents * self.game.n_opt_variables, 1)
        proj = backwardStep.BackwardStep(Q,q, A_ineq_all, b_ineq_all, A_eq_all, b_eq_all)
        x,status = proj(torch.zeros(1,self.game.N_agents * self.game.n_opt_variables, 1))
        return status