
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    multiply_matrix = (np.matmul(w.T, xTr) - yTr)

    loss = np.matmul(multiply_matrix, multiply_matrix.T) + lambdaa * np.matmul(w.T, w)

    gradient = np.matmul(xTr, multiply_matrix.T) + lambdaa * w
    gradient = 2 * gradient

    return loss, gradient
