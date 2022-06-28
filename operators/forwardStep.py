import numpy as np

def forwardStep(x0, grad, alpha):
    return x0-alpha*grad(x0)