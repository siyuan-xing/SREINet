"""
The Rossler network with higher-order interactions

Original MATLAB code from the paper:
"Reconstructing higher-order interactions in coupled dynamical systems"
By Federico Malizia, Alessandra Corso, Lucia Valentina Gambuzza
Giovanni Russo, Vito Latora, Mattia Frasca
Nature Communications - NCOMMS-23-20846A
"""

import numpy as np


def rossler_hoi(t, x, edge_list, triangle_list):
    """
    Rossler network dynamics with higher-order interactions
    """
    
    # Get number of nodes
    N = len(x) // 3
    
    # Extract state variables
    x_old = x[:N]           # x coordinates
    y_old = x[N:2*N]        # y coordinates  
    z_old = x[2*N:]         # z coordinates
    
    # rossler system parameters
    ar = 0.2
    br = 0.2  
    cr = 5.7
    
    k = 1e-1 # pairwise coupling strength
    kD = 1e-2  # higher-order coupling strength
    
    # Initialize coupling terms
    coup_rete = np.zeros(N)        # pairwise coupling
    coup_simplicial = np.zeros(N)  # higher-order coupling
    
    # Compute pairwise interactions
    for edge in edge_list:
        i1, i2 = edge[0], edge[1]
        coup_rete[i1] += x_old[i2] - x_old[i1]
        coup_rete[i2] += x_old[i1] - x_old[i2]
    
    # Compute higher-order interactions (triangular)
    for triangle in triangle_list:
        i1, i2, i3 = triangle[0], triangle[1], triangle[2]
        
        # For node i1
        coup_simplicial[i1] += (x_old[i2]**2 * x_old[i3] - x_old[i1]**3 + 
                               x_old[i2] * x_old[i3]**2 - x_old[i1]**3)
        
        # For node i2  
        coup_simplicial[i2] += (x_old[i1]**2 * x_old[i3] - x_old[i2]**3 + 
                               x_old[i1] * x_old[i3]**2 - x_old[i2]**3)
        
        # For node i3
        coup_simplicial[i3] += (x_old[i1]**2 * x_old[i2] - x_old[i3]**3 + 
                               x_old[i1] * x_old[i2]**2 - x_old[i3]**3)
    
    # rossler system equations with coupling
    dxdt1 = -y_old - z_old + k * coup_rete + kD * coup_simplicial
    dydt1 = x_old + ar * y_old
    dzdt1 = br + z_old * (x_old - cr)
    
    # Concatenate derivatives
    dxdt = np.concatenate([dxdt1, dydt1, dzdt1])
    
    return dxdt



