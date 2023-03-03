import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))
    
    # YOUR CODE HERE
    x_dot_z = 2 * np.dot(np.transpose(X),Z)
    x_dot_x = np.reshape(np.sum(X*X, axis = 0),(-1,1))
    z_dot_z = np.reshape(np.sum(Z*Z, axis = 0),(1,-1))
    
    dist = np.tile(x_dot_x,m) + np.transpose(np.tile(z_dot_z.T, n)) - x_dot_z
    
    dist[dist < 0] = 0

    D = np.sqrt(dist)
    
    return D
