import torch
from qpth.qp import QPFunction
import numpy as np
import osqp
from scipy import sparse

class BackwardStep(torch.nn.Module):
    # Proximal point operator for a quadratic cost and linear set
    # min 1/2 x'Qx + x'q + alpha/2|| x-x0 ||^2 ; x\in Ax<=b
    def __init__(self, Q, q, A_ineq, b_ineq, A_eq, b_eq, alpha=1):
        super().__init__()
        self.Q = torch.add(Q, alpha*torch.from_numpy(np.eye(Q.size(1))))  # batched sum
        self.q = q
        eps = 0.0001
        self.A_ineq = torch.cat( (torch.cat( (A_ineq, A_eq), 1) , -A_eq ), 1)
        self.b_ineq = torch.cat( (torch.cat( (b_ineq, b_eq + eps), 1), -b_eq ), 1)
        self.A_eq = torch.tensor(())
        self.b_eq = torch.tensor(())
        self.alpha = alpha # inertia

    def forward(self, x):
        q2 = torch.add(self.q,  - self.alpha * x) # Batched sum
        # y = QPFunction(verbose=True)(self.Q, q2.flatten(1), self.A_ineq, self.b_ineq.flatten(1), self.A_eq, self.b_eq)
        # return y.unsqueeze(2)

        y=torch.zeros(x.size())
        for n in range(x.size(0)):
            m = osqp.OSQP()
            Q = sparse.csr_matrix(self.Q[n,:,:].numpy())
            q2_n = q2[n,:,:].numpy()
            A_ineq = sparse.csr_matrix(self.A_ineq[n,:,:].numpy())
            b_ineq = self.b_ineq[n,:,:].numpy()
            l = -np.inf * np.ones((b_ineq.shape[0], 1))
            m.setup(P=Q, q=q2_n, A=A_ineq, l=l, u=b_ineq, verbose=False, warm_start=False, max_iter=3000, eps_abs=10**(-6), eps_rel=10**(-6))
            results = m.solve()
            y[n,:,:] = torch.from_numpy(np.transpose(np.matrix(results.x)))
        return y