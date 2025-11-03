"""
Simple Roessler Network Simulation 

This is a simplified simulation program that focuses only on the dynamics,
without the complex reconstruction analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from rossler_hoi import rossler_hoi
from network_loader import load_zachary_network


def simple_rossler_simulation(tmax=50, dt=0.01, visualization=True):
    """
    Run a simple Roessler network simulation
    
    Parameters:
    -----------
    tmax : float
        Maximum simulation time
    dt : float
        Time step for output
    visualization : bool
        Whether to create plots
        
    Returns:
    --------
    results : dict
        Simulation results
    """
    
    print("Starting simple Roessler network simulation...")
    
    # Load network structure
    data, edge_list, triangle_list, N = load_zachary_network()
    
    # Generate random initial conditions
    np.random.seed(42)
    x0_old = (30 * np.random.rand(N) - 15) / 5
    y0_old = (30 * np.random.rand(N) - 15) / 5
    z0_old = (40 * np.random.rand(N) - 5) / 5
    x0 = np.concatenate([x0_old, y0_old, z0_old])
    
    print(f"Simulating {N} coupled Roessler oscillators...")
    print(f"Time span: 0 to {tmax} with dt = {dt}")
    
    # Time vector
    T = np.arange(0, tmax + dt, dt)
    
    # Integrate the system
    start_time = time.time()
    sol = solve_ivp(lambda t, x: rossler_hoi(t, x, edge_list, triangle_list),
                    [0, tmax], x0, t_eval=T, method='DOP853', rtol=1e-8)
    
    if not sol.success:
        raise RuntimeError("Integration failed!")
    
    X = sol.y.T  # Transpose to match convention (time x variables)
    T = sol.t
    simulation_time = time.time() - start_time
    
    print(f"Integration completed in {simulation_time:.2f} seconds")
    print(f"Generated {len(T)} time points")
    
    # Calculate derivatives for the complete dataset
    print("Calculating derivatives for complete dataset...")
    dX = np.zeros_like(X)
    for i, t in enumerate(T):
        dX[i, :] = rossler_hoi(t, X[i, :], edge_list, triangle_list)
    
    if visualization:
        create_plots(T, X, N, edge_list, triangle_list)
        create_phase_diagram(T, X, N, edge_list, triangle_list)
    
    # Calculate some statistics
    results = {
        'T': T,
        'X': X,
        'dX': dX,  # Add derivatives to results
        'N': N,
        'edge_list': edge_list,
        'triangle_list': triangle_list,
        'simulation_time': simulation_time,
        'x_coords': X[:, :N],
        'y_coords': X[:, N:2*N],
        'z_coords': X[:, 2*N:],
        'dx_coords': dX[:, :N],  # x derivatives
        'dy_coords': dX[:, N:2*N],  # y derivatives
        'dz_coords': dX[:, 2*N:],  # z derivatives
    }
    
    return results


def create_plots(T, X, N, edge_list, triangle_list):
    """Create comprehensive visualization of the simulation results"""
    
    # Calculate derivatives for visualization
    print("Calculating derivatives for visualization...")
    dX = np.zeros_like(X)
    for i, t in enumerate(T):
        dX[i, :] = rossler_hoi(t, X[i, :], edge_list, triangle_list)
    
    fig = plt.figure(figsize=(24, 16))
    
    # Create a subplot layout with more space for derivative plots
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Time series for first few nodes (x coordinates)
    ax1 = fig.add_subplot(gs[0, 0])
    n_show = min(5, N)
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    for i in range(n_show):
        ax1.plot(T, X[:, i], color=colors[i], label=f'Node {i}', alpha=0.8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('x coordinate')
    ax1.legend()
    ax1.set_title('X Coordinates vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase portraits (x-y) for multiple nodes
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(min(3, N)):
        ax2.plot(X[:, i], X[:, N+i], color=colors[i], alpha=0.7, 
                linewidth=1.5, label=f'Node {i}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.set_title('Phase Portrait (x-y)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D trajectory projection (x-z)
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(min(3, N)):
        ax3.plot(X[:, i], X[:, 2*N+i], color=colors[i], alpha=0.7, 
                linewidth=1.5, label=f'Node {i}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.legend()
    ax3.set_title('Phase Portrait (x-z)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All coordinates for one node
    ax4 = fig.add_subplot(gs[1, 0])
    node_idx = 0
    ax4.plot(T, X[:, node_idx], 'b-', label='x', alpha=0.8)
    ax4.plot(T, X[:, N + node_idx], 'r-', label='y', alpha=0.8)
    ax4.plot(T, X[:, 2*N + node_idx], 'g-', label='z', alpha=0.8)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Coordinate')
    ax4.legend()
    ax4.set_title(f'All Coordinates - Node {node_idx}')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Long-term behavior
    ax5 = fig.add_subplot(gs[1, 1:])
    # Show longer time series for fewer nodes
    long_indices = slice(len(T)//2, None)  # Second half of simulation
    for i in range(min(2, N)):
        ax5.plot(T[long_indices], X[long_indices, i], 
                color=colors[i], alpha=0.8, label=f'Node {i}')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('x coordinate')
    ax5.legend()
    ax5.set_title('Long-term Dynamics (Second Half)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Derivative time series for first few nodes (dx/dt)
    ax6 = fig.add_subplot(gs[2, 0])
    for i in range(n_show):
        ax6.plot(T, dX[:, i], color=colors[i], label=f'Node {i}', alpha=0.8)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('dx/dt')
    ax6.legend()
    ax6.set_title('X Derivatives vs Time')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Derivative time series for first few nodes (dy/dt)
    ax7 = fig.add_subplot(gs[2, 1])
    for i in range(n_show):
        ax7.plot(T, dX[:, N+i], color=colors[i], label=f'Node {i}', alpha=0.8)
    ax7.set_xlabel('Time')
    ax7.set_ylabel('dy/dt')
    ax7.legend()
    ax7.set_title('Y Derivatives vs Time')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Derivative time series for first few nodes (dz/dt)
    ax8 = fig.add_subplot(gs[2, 2])
    for i in range(n_show):
        ax8.plot(T, dX[:, 2*N+i], color=colors[i], label=f'Node {i}', alpha=0.8)
    ax8.set_xlabel('Time')
    ax8.set_ylabel('dz/dt')
    ax8.legend()
    ax8.set_title('Z Derivatives vs Time')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Node dynamics heatmap
    ax9 = fig.add_subplot(gs[3, :])
    # Create heatmap data: nodes vs time, color represents x amplitude
    x_coords = X[:, :N]  # Extract x coordinates for all nodes
    
    # Create the heatmap
    im = ax9.imshow(x_coords.T, aspect='auto', cmap='RdBu_r', 
                   extent=[T[0], T[-1], 0, N-1], interpolation='bilinear')
    
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Node Index')
    ax9.set_title('Node Dynamics Heatmap (X Coordinate Amplitude)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
    cbar.set_label('X Coordinate')
    
    # Set y-axis ticks to show node indices
    if N <= 20:
        ax9.set_yticks(range(0, N, max(1, N//10)))
    else:
        ax9.set_yticks(range(0, N, N//10))
    
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Roessler Network Simulation - {N} Nodes (Including Derivatives)', fontsize=16, y=0.97)
    plt.savefig('simple_rossler_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_phase_diagram(T, X, N, edge_list, triangle_list):
    """
    Create a unified phase diagram showing all results and their derivatives in one plot
    """
    print("Creating unified phase diagram...")
    
    # Calculate exact derivatives using the system equations
    dX_exact = np.zeros_like(X)
    for i, t in enumerate(T):
        dX_exact[i, :] = rossler_hoi(t, X[i, :], edge_list, triangle_list)
    
    # Extract coordinates for all nodes
    x_coords = X[:, :N]
    y_coords = X[:, N:2*N]
    z_coords = X[:, 2*N:]
    
    # Extract derivative coordinates for all nodes
    dx_coords = dX_exact[:, :N]
    dy_coords = dX_exact[:, N:2*N]
    dz_coords = dX_exact[:, 2*N:]
    
    # Create a single large phase diagram
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate colors for all nodes
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    
    # Plot all variables and their derivatives in one unified phase space
    # We'll use different markers and alpha values to distinguish different relationships
    
    # Plot x vs dx/dt for all nodes
    for i in range(N):
        ax.plot(x_coords[:, i], dx_coords[:, i], 
               color=colors[i], alpha=0.6, linewidth=1.2, 
               label=f'x-dx/dt Node {i}' if i < 5 else '')
    
    # Plot y vs dy/dt for all nodes (with different line style)
    for i in range(N):
        ax.plot(y_coords[:, i], dy_coords[:, i], 
               color=colors[i], alpha=0.5, linewidth=1.0, linestyle='--',
               label=f'y-dy/dt Node {i}' if i < 3 else '')
    
    # Plot z vs dz/dt for all nodes (with different line style)
    for i in range(N):
        ax.plot(z_coords[:, i], dz_coords[:, i], 
               color=colors[i], alpha=0.4, linewidth=0.8, linestyle=':',
               label=f'z-dz/dt Node {i}' if i < 2 else '')
    
    # Add some cross-variable relationships
    # Plot x vs dy/dt (showing coupling between variables)
    for i in range(min(3, N)):
        ax.plot(x_coords[:, i], dy_coords[:, i], 
               color=colors[i], alpha=0.3, linewidth=0.8, linestyle='-.',
               label=f'x-dy/dt Node {i}' if i < 2 else '')
    
    ax.set_xlabel('Variable Values')
    ax.set_ylabel('Derivative Values')
    ax.set_title(f'Unified Phase Diagram - All Variables and Derivatives\n{N} Nodes Rossler Network')
    ax.grid(True, alpha=0.3)
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.6, label='x vs dx/dt'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.5, linestyle='--', label='y vs dy/dt'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.4, linestyle=':', label='z vs dz/dt'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.3, linestyle='-.', label='x vs dy/dt (coupling)'),
        Line2D([0], [0], color='white', linewidth=0, label=f'Colors: {N} different nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('unified_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Simple Roessler Network Simulation")
    
    # Run simulation
    results = simple_rossler_simulation(tmax=30, dt=0.05, visualization=True)
    
    # Save data
    np.savez('simple_rossler_data.npz', **results)
    
    print(f"Simulation completed. Data saved to 'simple_rossler_data.npz'")
    print(f"Saved data includes: X (states), dX (derivatives), T (time), and network structure")
    print(f"Plots saved as 'simple_rossler_results.png' and 'unified_phase_diagram.png'")


