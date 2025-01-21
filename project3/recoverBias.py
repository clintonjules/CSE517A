"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    index = 0

    # YOUR CODE HERE
    margin = np.abs(C - 2 * alphas)
    minmargin = min(margin)

    index = np.where(margin == minmargin)[0][0]
    
    return yTr[index] - np.dot(K[index,:], (yTr * alphas))
    
