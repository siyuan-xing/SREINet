"""
The Kuramoto network with higher-order interactions

Implements the Kuramoto model with first-order (pairwise) and second-order (triangular) interactions
as described in the equation:

θ̇ᵢ = ω + γ₁/(k⁽¹⁾) ∑ⱼ₌₁ⁿ Aᵢⱼ sin(θⱼ - θᵢ) + γ₂/(k⁽²⁾) ∑ⱼ,ₖ₌₁ⁿ Bᵢⱼₖ 1/2 sin(θⱼ + θₖ - 2θᵢ)

Where:
- θᵢ are the phase oscillators
- ω is the natural frequency
- γ₁, γ₂ are coupling strengths for first and second order interactions
- Aᵢⱼ is the adjacency matrix (first-order interactions)  
- Bᵢⱼₖ is the triangle tensor (second-order interactions)
- k⁽¹⁾, k⁽²⁾ are the degrees for normalization

Following the paper setting: γ₁ = 1-α, γ₂ = α, α ∈ [0,1]
"""

import numpy as np


def kuramoto_hoi(t, theta, edge_list, triangle_list, omega=1.0, alpha=0.2):
    """
    Kuramoto network dynamics with higher-order interactions
    
    Parameters:
    -----------
    t : float
        Time (not used in autonomous system but required by integrator)
    theta : numpy.ndarray
        Array of phase angles for all oscillators
    edge_list : numpy.ndarray
        Array of edges defining pairwise interactions
    triangle_list : numpy.ndarray
        Array of triangles defining higher-order interactions
    omega : float
        Natural frequency of oscillators
    alpha : float
        Parameter controlling balance between first and second order interactions
        alpha ∈ [0,1], where γ₁ = 1-α and γ₂ = α
        
    Returns:
    --------
    dthetadt : numpy.ndarray
        Derivatives of phase angles
    """
    
    N = len(theta)
    omega_arr=np.linspace(0, omega, N)
    
    # Coupling strengths
    gamma1 = 1.0 - alpha  # First-order coupling strength
    gamma2 = alpha        # Second-order coupling strength
    
    # Calculate degrees for normalization
    k1 = np.zeros(N)  # First-order degree (number of neighbors)
    k2 = np.zeros(N)  # Second-order degree (number of triangles)
    
    # Count first-order connections
    for edge in edge_list:
        i1, i2 = edge[0], edge[1]
        k1[i1] += 1
        k1[i2] += 1
    
    # Count second-order connections (triangles each node participates in)
    for triangle in triangle_list:
        i1, i2, i3 = triangle[0], triangle[1], triangle[2]
        k2[i1] += 1
        k2[i2] += 1
        k2[i3] += 1
    
    # Avoid division by zero
    k1[k1 == 0] = 1
    k2[k2 == 0] = 1
    
    # Initialize coupling terms
    first_order_coupling = np.zeros(N)
    second_order_coupling = np.zeros(N)
    
    # Compute first-order (pairwise) interactions: ∑ⱼ Aᵢⱼ sin(θⱼ - θᵢ)
    for edge in edge_list:
        i1, i2 = edge[0], edge[1]
        first_order_coupling[i1] += np.sin(theta[i2] - theta[i1])
        first_order_coupling[i2] += np.sin(theta[i1] - theta[i2])
    
    # Compute second-order (triangular) interactions: ∑ⱼ,ₖ Bᵢⱼₖ (1/2) sin(θⱼ + θₖ - 2θᵢ)
    for triangle in triangle_list:
        i1, i2, i3 = triangle[0], triangle[1], triangle[2]
        
        # For node i1: interactions with i2,i3
        second_order_coupling[i1] += 0.5 * np.sin(theta[i2] + theta[i3] - 2*theta[i1])
        
        # For node i2: interactions with i1,i3  
        second_order_coupling[i2] += 0.5 * np.sin(theta[i1] + theta[i3] - 2*theta[i2])
        
        # For node i3: interactions with i1,i2
        second_order_coupling[i3] += 0.5 * np.sin(theta[i1] + theta[i2] - 2*theta[i3])
    
    # Kuramoto equation with higher-order interactions
    dthetadt = (omega_arr + 
                gamma1 * first_order_coupling / k1 + 
                gamma2 * second_order_coupling / k2)
    
    return dthetadt



def calculate_order_parameter(theta):
    """
    Calculate the Kuramoto order parameter R
    
    R = |1/N ∑ᵢ e^(iθᵢ)|
    
    Parameters:
    -----------
    theta : numpy.ndarray
        Phase angles
        
    Returns:
    --------
    R : float
        Order parameter (0 = incoherent, 1 = fully synchronized)
    psi : float
        Average phase
    """
    
    N = len(theta)
    complex_order = np.mean(np.exp(1j * theta))
    R = np.abs(complex_order)
    psi = np.angle(complex_order)
    
    return R, psi


def calculate_local_order_parameter(theta, edge_list):
    """
    Calculate local order parameter for each node based on its neighbors
    
    Parameters:
    -----------
    theta : numpy.ndarray
        Phase angles
    edge_list : numpy.ndarray
        Network edges
        
    Returns:
    --------
    R_local : numpy.ndarray
        Local order parameter for each node
    """
    
    N = len(theta)
    R_local = np.zeros(N)
    
    # Build neighbor lists
    neighbors = [[] for _ in range(N)]
    for edge in edge_list:
        i1, i2 = edge[0], edge[1]
        neighbors[i1].append(i2)
        neighbors[i2].append(i1)
    
    # Calculate local order parameter
    for i in range(N):
        if len(neighbors[i]) > 0:
            neighbor_phases = theta[neighbors[i]]
            complex_order = np.mean(np.exp(1j * neighbor_phases))
            R_local[i] = np.abs(complex_order)
        else:
            R_local[i] = 0.0
    
    return R_local
