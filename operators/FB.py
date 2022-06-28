import backwardStep
import forwardStep
import numpy as np

def FB( grad, A, b, Aeq, beq, x0, alpha ):
    x = forwardStep(x0, grad, alpha)
    return backwardStep(np.eye(np.size(x,1)), np.zeros(np.size(x,1),1), A, b, Aeq, beq, x, 1) #projection
    