from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    hinge_mat = yTr * np.matmul(w.T, xTr)

    loss = maximum(0, 1 - hinge_mat)
    loss = np.sum(loss) + lambdaa * np.matmul(w.T,w)
    
    gradient = np.where(hinge_mat > 1, 0, yTr)
    gradient = 0 - np.matmul(xTr, gradient.T) + 2 * lambdaa * w

    return loss, gradient
