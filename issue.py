import torch
from qpth.qp import QPFunction

Q = torch.tensor([[[1.]]])
q = torch.tensor([[1.]])
A_eq = torch.tensor(())
A_ineq = torch.tensor([[[-1.]]])
b_ineq = torch.tensor([[-1.]])
b_eq = torch.tensor(())
y = QPFunction(verbose=True)(Q, q, A_ineq, b_ineq, A_eq, b_eq)

print(y)