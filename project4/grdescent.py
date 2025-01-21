# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    f = lambda w_in : func(w_in)
    w = w0
    loss, gradient = f(w0)

    for i in range(maxiter):
        new_loss, gradient = f(w)

        if np.linalg.norm(gradient) >= tolerance:
            if new_loss < loss:
                stepsize = stepsize * 1.01
            else:
                stepsize = stepsize / 2.0 
                stepsize = stepsize if stepsize > eps else eps

            w = w - stepsize * gradient
        else:
            break

        loss = new_loss  
    return w