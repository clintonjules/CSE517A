#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: Yichen
@author: M.Joo
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#
#    Output:
#    posprob: probability vector of p(x|y=1) (dx1)
#    negprob: probability vector of p(x|y=-1) (dx1)
# =============================================================================



    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    # TODO: do not use np.matrix!
    X = np.matrix(x)
    Y = np.matrix(y)

    d,n = X.shape

    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.array([[-1, 1]])

    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
    Ynew = np.hstack((Y, Y0))

    # matrix of all-zeros -
    X1 = np.zeros((d, 2))
    Xnew = np.hstack((Xnew, X1))
    Ynew = np.hstack((Ynew, Y0))

    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape

# =============================================================================
# fill in code here

    pos_dim = np.zeros([d, 1])
    neg_dim = np.zeros([d, 1])
    n_pos = 0
    n_neg = 0

    for a in range(d):
        for i in range(n):
            if Ynew[:, i] == 1: 
                pos_dim[a, :] += Xnew[a, i]

            if Ynew[:, i] == -1: 
                neg_dim[a, :] += Xnew[a, i]

    point_sum = np.sum(Xnew, axis = 0)

    for i in range(n):
        if Ynew[:, i] == 1:
            n_pos += point_sum[:, i]
        if Ynew[:, i] == -1:
            n_neg += point_sum[:, i]


    posprob = pos_dim / n_pos
    negprob = neg_dim / n_neg
    
    return posprob,negprob

# =============================================================================
