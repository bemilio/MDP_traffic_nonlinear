import numpy as np
import torch
from operators.backwardStep import BackwardStep

class FBF_distr_algorithm:
    def __init__(self, game, x_0=None, dual_0=None, aux_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001, consensus_stepsize=0.001):
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

    def run_once(self):
        x = self.x
        dual = self.dual
        aux = self.aux
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F(x)
        x_half_fbf, status = self.projection(
            x - self.primal_stepsize * (r + torch.bmm(torch.transpose(A_i, 1, 2), self.dual)))
        d = torch.bmm(A_i, x) - b_i
        dual_half_fbf = torch.maximum(dual + self.dual_stepsize * (d + self.game.K(aux - dual)),
                                      torch.zeros(dual.size()))
        aux_half_fbf = aux - self.consensus_stepsize * self.game.K(dual)

        r = self.game.F(x_half_fbf) - self.game.F(x)
        x_new, status = self.projection(
            x_half_fbf - self.primal_stepsize * (r + torch.bmm(torch.transpose(A_i, 1, 2), dual_half_fbf - dual)))
        d = torch.bmm(A_i, x_half_fbf - x)
        dual_new = dual_half_fbf + self.dual_stepsize * (
                    d + self.game.K(aux_half_fbf - aux) - self.game.K(dual_half_fbf - dual))
        aux_new = aux_half_fbf - self.consensus_stepsize * self.game.K(dual_half_fbf - dual)

        self.x = x_new
        self.dual = dual_new
        self.aux = aux_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        return self.x, self.dual, self.aux, residual, cost

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection(x-self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        dual_transformed = torch.maximum(self.dual + torch.sum(torch.bmm(A_i, self.x) - b_i, 0), torch.zeros(self.dual.size()))
        residual = np.sqrt( ((x - x_transformed).norm())**2 + ((self.dual-dual_transformed).norm())**2 )
        return residual

    def set_stepsize_using_Lip_const(self, safety_margin=0.5):
        #compute stepsizes as (1/L)*safety_margin, where L is the lipschitz constant of the forward operator
        N = self.x.size(0)
        n_x = self.x.size(1)
        n_constr = self.game.A_ineq_shared.size(1)
        # In the linear-quadratic game case, the forward operator is:
        # [ 0, A', 0;
        #   A, L, -L;
        #   0, L,  0 ] + F(x)
        # Where A=diag(A_i) and L is (kron(Laplacian, n_constr)).
        A = torch.zeros(N*n_constr, N*n_x)
        for i in range(N):
            A[i*n_constr:(i+1)*n_constr, i*n_x:(i+1)*n_x] = self.game.A_ineq_shared[i,:,:]
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