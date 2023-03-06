import numpy as np
import torch
from operators.backwardStep import BackwardStep

class pFB_tich_prox_distr_algorithm:
    def __init__(self, game, x_0=None, dual_0=None, aux_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001, consensus_stepsize=0.001,
                 exponent_vanishing_precision=2, exponent_vanishing_selection=1, alpha_tich_regul=1):
        self.game = game
        self.primal_stepsize = primal_stepsize
        self.dual_stepsize = dual_stepsize
        self.consensus_stepsize = consensus_stepsize
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
            self.dual = torch.zeros(N_agents,m, 1)
        if aux_0:
            self.aux = aux_0
        else:
            self.aux = torch.zeros(N_agents,m, 1)
        self.x_pivot = self.x # used for tichonov regularization. It is updated at every outer iteration.
        self.dual_pivot = self.dual # used for tichonov regularization. It is updated at every outer iteration.
        self.aux_pivot = self.aux  # used for tichonov regularization. It is updated at every outer iteration.
        Q = torch.zeros(N_agents, n, n) # Local cost is zero
        q = torch.zeros(N_agents, n, 1)
        self.projection = BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,1)
        self.eps_tich = lambda t: t**(-1*exponent_vanishing_precision)
        self.weight_sel = lambda t: (t**(-1*exponent_vanishing_selection))
        self.alpha_tich_regul = alpha_tich_regul
        self.outer_iter = 1

    def run_once(self):
        x = self.x
        dual = self.dual
        aux = self.aux
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F(x)
        sel = self.weight_sel(self.outer_iter) * self.game.nabla_phi(x)
        tich_x = self.alpha_tich_regul * (x - self.x_pivot)
        tich_dual = 0 # self.alpha_tich_regul * (dual - self.dual_pivot)
        tich_aux = 0 #self.alpha_tich_regul * (aux - self.aux_pivot)
        x_new, status = self.projection(x - self.primal_stepsize * (r + sel + tich_x + torch.bmm(torch.transpose(A_i, 1, 2), self.dual ) ) )
        aux_new = aux - self.consensus_stepsize * (tich_aux + self.game.K(dual))
        d = 2 * torch.bmm(A_i, x_new) - torch.bmm(A_i, x) - b_i
        # Dual update
        dual_new = torch.maximum(dual + self.dual_stepsize * ( d - tich_dual + self.game.K(2*aux_new - aux) - self.game.K(dual)), torch.zeros(dual.size()))
        if torch.norm(x-x_new) + torch.norm(dual-dual_new) + torch.norm(aux-aux_new) <= self.eps_tich(self.outer_iter):
            self.outer_iter = self.outer_iter + 1
            self.x_pivot = x_new
        self.x = x_new
        self.dual = dual_new
        self.aux = aux_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        sel = self.game.phi(self.x)
        return self.x, self.dual, self.aux, residual, cost, sel

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection(x-self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        dual_transformed = torch.maximum(self.dual + torch.sum(torch.bmm(A_i, self.x) - b_i, 0), torch.zeros(self.dual.size()))
        residual = np.sqrt( ((x - x_transformed).norm())**2 + ((self.dual-dual_transformed).norm())**2 )
        return residual

    def set_stepsize_using_Lip_const(self, safety_margin=.5):
        #compute stepsizes as (1/L)*safety_margin, where L is the lipschitz constant of the forward operator
        N = self.x.size(0)
        n_x = self.x.size(1)
        n_constr = self.game.A_ineq_shared.size(1)
        mu_pseudog, Lip_pseudog = self.game.F.get_strMon_Lip_constants()
        str_mon = mu_pseudog + self.alpha_tich_regul
        max_neigh = max([torch.abs(self.game.K.L[i, i, 0, 0]).item() for i in range(N)]) # The diagonal of the Laplacian matrix contains the node degree
        max_A = torch.max(torch.sum(torch.abs(self.game.A_ineq_shared),dim=2)).item()
        delta = (1/(safety_margin*min( str_mon, (2/max_neigh)) ) )
        self.primal_stepsize = 1/(max_A + delta)
        self.dual_stepsize = 1 / (max_A + 2*max_neigh + delta)
        self.consensus_stepsize = 1 / ( 2 * max_neigh + delta)

    def check_feasibility(self):
        n_local_ineq_constr = self.game.A_ineq_loc.size(1)
        n_local_eq_constr = self.game.A_eq_loc.size(1)
        A_ineq_all = torch.zeros(
            (1, self.game.N_agents * n_local_ineq_constr + self.game.n_shared_ineq_constr,
             self.game.N_agents * self.game.n_opt_variables))
        b_ineq_all = torch.zeros((1, self.game.N_agents * n_local_ineq_constr + self.game.n_shared_ineq_constr, 1))
        A_eq_all = torch.zeros(
            (1, self.game.N_agents * n_local_eq_constr + self.game.n_shared_ineq_constr,
             self.game.N_agents * self.game.n_opt_variables))
        b_eq_all = torch.zeros((1, self.game.N_agents * n_local_eq_constr + self.game.n_shared_ineq_constr, 1))
        for i in range(self.game.N_agents):
            A_ineq_all[0, i * n_local_ineq_constr:(i + 1) * n_local_ineq_constr,
            i * self.game.n_opt_variables:(i + 1) * self.game.n_opt_variables] = self.game.A_ineq_loc[i, :, :]
            b_ineq_all[0, i * n_local_ineq_constr:(i + 1) * n_local_ineq_constr, :] = self.game.b_ineq_loc[i, :, :]
            A_eq_all[0, i * n_local_eq_constr:(i + 1) * n_local_eq_constr,
            i * self.game.n_opt_variables:(i + 1) * self.game.n_opt_variables] = self.game.A_eq_loc[i, :, :]
            b_eq_all[0, i * n_local_eq_constr:(i + 1) * n_local_eq_constr, :] = self.game.b_eq_loc[i, :, :]
            A_ineq_all[0, -self.game.n_shared_ineq_constr:,
            i * self.game.n_opt_variables:(i + 1) * self.game.n_opt_variables] = self.game.A_ineq_shared[i, :, :]
            b_ineq_all[0, -self.game.n_shared_ineq_constr:, :] = b_ineq_all[0, -self.game.n_shared_ineq_constr:,
                                                                 :] + self.game.b_ineq_shared[i, :, :]
        Q = torch.zeros(1, self.game.N_agents * self.game.n_opt_variables,
                        self.game.N_agents * self.game.n_opt_variables)
        q = torch.zeros(1, self.game.N_agents * self.game.n_opt_variables, 1)
        proj = BackwardStep(Q, q, A_ineq_all, b_ineq_all, A_eq_all, b_eq_all)
        x, status = proj(torch.zeros(1, self.game.N_agents * self.game.n_opt_variables, 1))
        return status

class FBF_HSDM_distr_algorithm:
    def __init__(self, game, x_0=None, dual_0=None, aux_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001, consensus_stepsize=0.001, exponent_vanishing_selection=1):
        self.game = game
        self.primal_stepsize = primal_stepsize
        self.dual_stepsize = dual_stepsize
        self.consensus_stepsize = consensus_stepsize
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
            self.dual = torch.zeros(N_agents,m, 1)
        if aux_0:
            self.aux = aux_0
        else:
            self.aux = torch.zeros(N_agents,m, 1)
        Q = torch.zeros(N_agents, n, n) # Local cost is zero
        q = torch.zeros(N_agents, n, 1)
        self.projection = BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,1)
        self.weight_sel = lambda t: .1*(t**(-1*exponent_vanishing_selection))
        self.iteration = 1

    def run_once(self):
        x = self.x
        dual = self.dual
        aux = self.aux
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F(x)
        x_half_fbf, status = self.projection(x - self.primal_stepsize * (r + torch.bmm(torch.transpose(A_i, 1, 2), self.dual ) ) )
        d = torch.bmm(A_i, x) - b_i
        dual_half_fbf =torch.maximum(dual + self.dual_stepsize * (d + self.game.K(aux - dual)), torch.zeros(dual.size()))
        aux_half_fbf = aux - self.consensus_stepsize * self.game.K(dual)

        r = self.game.F(x_half_fbf) - self.game.F(x)
        x_fbf, status = self.projection(x_half_fbf - self.primal_stepsize * (r + torch.bmm(torch.transpose(A_i, 1, 2), dual_half_fbf - dual ) ) )
        d = torch.bmm(A_i, x_half_fbf - x)
        dual_fbf = dual_half_fbf + self.dual_stepsize * (d + self.game.K(aux_half_fbf - aux) - self.game.K(dual_half_fbf - dual))
        aux_half = aux_half_fbf - self.consensus_stepsize * self.game.K(dual_half_fbf - dual)

        # HSDM step
        x_new = x_fbf - self.weight_sel(self.iteration) * self.game.nabla_phi(x)
        self.iteration = self.iteration + 1
        dual_new = dual_fbf
        aux_new = aux_half

        self.x = x_new
        self.dual = dual_new
        self.aux = aux_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        sel = self.game.phi(self.x)
        return self.x, self.dual, self.aux, residual, cost, sel

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection(x-self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        dual_transformed = torch.maximum(self.dual + torch.sum(torch.bmm(A_i, self.x) - b_i, 0), torch.zeros(self.dual.size()))
        residual = np.sqrt( ((x - x_transformed).norm())**2 + ((self.dual-dual_transformed).norm())**2 )
        return residual

    def set_stepsize_using_Lip_const(self, safety_margin=.5):
        #compute stepsizes as (1/L)*safety_margin, where L is the lipschitz constant of the forward operator
        N = self.x.size(0)
        n_x = self.x.size(1)
        n_constr = self.game.A_ineq_shared.size(1)
        # the forward operator is:
        # [ 0, A', 0;
        #   A, L, -L;
        #   0, L,  0 ] + F(x)
        # Where A=diag(A_i) and L is (kron(Laplacian, n_constr)).
        A = torch.zeros(N*n_x, N*n_constr)
        for i in range(N):
            A[i*n_x:(i+1)*n_x, i*n_constr:(i+1)*n_constr] = self.game.A_ineq_shared[i,:,:]
        L = torch.zeros(N * n_constr, N * n_constr)
        for i in range(N):
            for j in range(N):
                L[i * n_constr:(i + 1) * n_constr, j * n_constr:(j + 1) * n_constr] = self.game.K.L[i, j, :, :]
        H = torch.cat((torch.cat((torch.zeros(N*n_x,N*n_x), torch.transpose(A,0,1), torch.zeros(N*n_x, N*n_constr)), dim=1), \
                       torch.cat((A, L, -L),dim=1), \
                       torch.cat((torch.zeros(N*n_constr, N*n_x), L, torch.zeros(N*n_constr, N*n_constr)), dim=1)), dim=0)
        U, S, V = torch.linalg.svd(H)
        Lip_H = torch.max(S).item()
        mu, Lip_pseudog = self.game.F.get_strMon_Lip_constants()
        Lip_tot = Lip_H + Lip_pseudog
        self.primal_stepsize = safety_margin/Lip_tot
        self.dual_stepsize = safety_margin/Lip_tot
        self.consensus_stepsize = safety_margin/Lip_tot