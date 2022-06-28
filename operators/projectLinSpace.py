from .backwardStep import backwardStep
import numpy as np
from scipy import sparse

def projectLinSpace(Aall, l, u, x):
    Q = sparse.csc_matrix((x.shape[0], x.shape[0]), dtype=float)
    q = np.zeros((x.shape[0], 1))
    alpha = 1
    (x_proj, solution_status) =backwardStep(Q, q, Aall, l, u, x, alpha)
    return(x_proj, solution_status)