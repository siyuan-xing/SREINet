"""
Simple Kuramoto Network Simulation with Higher-Order Interactions

This is a simulation program for Kuramoto oscillators on a network with both
pairwise (first-order) and triangular (second-order) interactions.

Based on the equation:
θ̇ᵢ = ω + γ₁/(k⁽¹⁾) ∑ⱼ₌₁ⁿ Aᵢⱼ sin(θⱼ - θᵢ) + γ₂/(k⁽²⁾) ∑ⱼ,ₖ₌₁ⁿ Bᵢⱼₖ 1/2 sin(θⱼ + θₖ - 2θᵢ)

Where γ₁ = 1-α, γ₂ = α, α ∈ [0,1]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

from kuramoto_hoi import kuramoto_hoi, calculate_order_parameter, calculate_local_order_parameter
from network_loader import load_zachary_network


def simple_kuramoto_simulation(tmax=50, dt=0.01, omega=1.0, alpha=0.5, visualization=True):
    """
    Run a simple Kuramoto network simulation with higher-order interactions
    
    Parameters:
    -----------
    tmax : float
        Maximum simulation time
    dt : float
        Time step for output
    omega : float
        Natural frequency of oscillators
    alpha : float
        Balance parameter between first and second order interactions (0 ≤ α ≤ 1)
        α = 0: only first-order interactions
        α = 1: only second-order interactions
        α = 0.5: equal contribution
    visualization : bool
        Whether to create plots
        
    Returns:
    --------
    results : dict
        Simulation results
    """
    
    print("Starting simple Kuramoto network simulation...")
    print(f"Parameters: ω = {omega}, α = {alpha}")
    
    # Load network structure
    data, edge_list, triangle_list, N = load_zachary_network()
    
    # Generate random initial conditions (phase angles between 0 and 2π)
    np.random.seed(42)
    theta0 = 2 * np.pi * np.random.rand(N)
    
    print(f"Simulating {N} coupled Kuramoto oscillators...")
    print(f"Time span: 0 to {tmax} with dt = {dt}")
    print(f"Initial phases range: [{np.min(theta0):.3f}, {np.max(theta0):.3f}]")
    
    # Time vector
    T = np.arange(0, tmax + dt, dt)
    
    # Integrate the system
    start_time = time.time()
    sol = solve_ivp(lambda t, theta: kuramoto_hoi(t, theta, edge_list, triangle_list, omega, alpha),
                    [0, tmax], theta0, t_eval=T, method='DOP853', rtol=1e-8)
    
    if not sol.success:
        raise RuntimeError("Integration failed!")
    
    Theta = sol.y.T  # Transpose to match convention (time x variables)
    T = sol.t
    simulation_time = time.time() - start_time
    
    print(f"Integration completed in {simulation_time:.2f} seconds")
    print(f"Generated {len(T)} time points")
    
    # Calculate derivatives for the complete dataset
    print("Calculating derivatives and order parameters...")
    dTheta = np.zeros_like(Theta)
    order_params = np.zeros(len(T))
    avg_phases = np.zeros(len(T))
    local_order_params = np.zeros((len(T), N))
    
    for i, t in enumerate(T):
        dTheta[i, :] = kuramoto_hoi(t, Theta[i, :], edge_list, triangle_list, omega, alpha)
        order_params[i], avg_phases[i] = calculate_order_parameter(Theta[i, :])
        local_order_params[i, :] = calculate_local_order_parameter(Theta[i, :], edge_list)
    
    print(f"Final order parameter: R = {order_params[-1]:.4f}")
    print(f"Average order parameter: R_avg = {np.mean(order_params):.4f}")
    
    if visualization:
        create_kuramoto_plots(T, Theta, dTheta, order_params, avg_phases, local_order_params, 
                             N, edge_list, triangle_list, omega, alpha)
        create_phase_synchronization_analysis(T, Theta, order_params, local_order_params, 
                                            N, edge_list, triangle_list, alpha)
    
    # Calculate some statistics
    results = {
        'T': T,
        'Theta': Theta,
        'dTheta': dTheta,
        'N': N,
        'edge_list': edge_list,
        'triangle_list': triangle_list,
        'simulation_time': simulation_time,
        'order_parameters': order_params,
        'average_phases': avg_phases,
        'local_order_parameters': local_order_params,
        'omega': omega,
        'alpha': alpha,
        'final_order_parameter': order_params[-1],
        'average_order_parameter': np.mean(order_params),
    }
    
    return results


def create_kuramoto_plots(T, Theta, dTheta, order_params, avg_phases, local_order_params, 
                         N, edge_list, triangle_list, omega, alpha):
    """Create comprehensive visualization of the Kuramoto simulation results"""
    
    fig = plt.figure(figsize=(24, 16))
    
    # Create a subplot layout
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Phase evolution for first few oscillators
    ax1 = fig.add_subplot(gs[0, 0])
    n_show = min(8, N)
    colors = plt.cm.hsv(np.linspace(0, 1, n_show))
    for i in range(n_show):
        # Unwrap phases to show continuous evolution
        unwrapped_phases = np.unwrap(Theta[:, i])
        ax1.plot(T, unwrapped_phases, color=colors[i], label=f'Osc {i}', alpha=0.8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Phase (unwrapped)')
    ax1.legend()
    ax1.set_title('Phase Evolution (Unwrapped)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase evolution on unit circle (wrapped)
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(n_show):
        ax2.plot(T, Theta[:, i], color=colors[i], label=f'Osc {i}', alpha=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_ylim([0, 2*np.pi])
    ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax2.legend()
    ax2.set_title('Phase Evolution (Wrapped)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Global order parameter evolution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(T, order_params, 'b-', linewidth=2, label='Order Parameter R')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Order Parameter R')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.set_title(f'Global Synchronization (α = {alpha})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Phase derivatives (angular velocities)
    ax4 = fig.add_subplot(gs[1, 0])
    for i in range(n_show):
        ax4.plot(T, dTheta[:, i], color=colors[i], label=f'Osc {i}', alpha=0.8)
    ax4.axhline(y=omega, color='black', linestyle='--', alpha=0.7, label=f'Natural freq ω = {omega}')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('dθ/dt (angular velocity)')
    ax4.legend()
    ax4.set_title('Angular Velocities')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Average phase evolution  
    ax5 = fig.add_subplot(gs[1, 1])
    unwrapped_avg_phase = np.unwrap(avg_phases)
    ax5.plot(T, unwrapped_avg_phase, 'r-', linewidth=2, label='Average Phase')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Average Phase (unwrapped)')
    ax5.legend()
    ax5.set_title('Average Phase Evolution')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Local order parameters heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(local_order_params.T, aspect='auto', cmap='viridis', 
                   extent=[T[0], T[-1], 0, N-1], interpolation='bilinear')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Oscillator Index')
    ax6.set_title('Local Order Parameters')
    cbar = plt.colorbar(im, ax=ax6, shrink=0.8)
    cbar.set_label('Local R')
    
    # Plot 7: Phase portrait on unit circle (Poincaré plot)
    ax7 = fig.add_subplot(gs[2, 0], projection='polar')
    # Show trajectories on unit circle for several oscillators
    for i in range(min(5, N)):
        ax7.plot(Theta[:, i], np.ones_like(Theta[:, i]), 
                color=colors[i], alpha=0.7, linewidth=2, label=f'Osc {i}')
    ax7.set_ylim([0, 1.2])
    ax7.set_title('Phase Trajectories (Unit Circle)')
    ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Plot 8: Final state visualization
    ax8 = fig.add_subplot(gs[2, 1], projection='polar')
    final_phases = Theta[-1, :]
    ax8.scatter(final_phases, np.ones(N), c=np.arange(N), cmap='hsv', s=100, alpha=0.8)
    ax8.set_ylim([0, 1.2])
    ax8.set_title(f'Final State (R = {order_params[-1]:.3f})')
    
    # Plot 9: Synchronization metrics over time
    ax9 = fig.add_subplot(gs[2, 2])
    # Calculate phase coherence (standard deviation of phases)
    phase_coherence = np.zeros(len(T))
    for i, t in enumerate(T):
        # Calculate circular standard deviation
        mean_phase = avg_phases[i]
        phase_diffs = np.angle(np.exp(1j * (Theta[i, :] - mean_phase)))
        phase_coherence[i] = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phase_diffs)))))
    
    ax9_twin = ax9.twinx()
    line1 = ax9.plot(T, order_params, 'b-', linewidth=2, label='Order Parameter R')
    line2 = ax9_twin.plot(T, phase_coherence, 'r-', linewidth=2, label='Phase Coherence')
    
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Order Parameter R', color='blue')
    ax9_twin.set_ylabel('Phase Coherence', color='red')
    ax9.set_title('Synchronization Metrics')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax9.legend(lines, labels, loc='upper right')
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Long-term dynamics heatmap
    ax10 = fig.add_subplot(gs[3, :])
    # Show phase evolution for all oscillators
    im = ax10.imshow(Theta.T, aspect='auto', cmap='hsv', 
                    extent=[T[0], T[-1], 0, N-1], interpolation='bilinear')
    ax10.set_xlabel('Time')
    ax10.set_ylabel('Oscillator Index')
    ax10.set_title('All Phase Evolutions')
    cbar = plt.colorbar(im, ax=ax10, shrink=0.8)
    cbar.set_label('Phase (radians)')
    
    plt.suptitle(f'Kuramoto Network - {N} Oscillators (ω={omega}, α={alpha})', fontsize=16, y=0.97)
    plt.savefig('simple_kuramoto_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_phase_synchronization_analysis(T, Theta, order_params, local_order_params, 
                                         N, edge_list, triangle_list, alpha):
    """
    Create detailed phase synchronization analysis plots
    """
    print("Creating phase synchronization analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Order parameter distribution over time
    ax1.hist(order_params, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(order_params), color='red', linestyle='--', linewidth=2, 
               label=f'Mean R = {np.mean(order_params):.3f}')
    ax1.set_xlabel('Order Parameter R')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Order Parameter Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Local vs global order parameter correlation
    final_local_order = local_order_params[-1, :]
    ax2.scatter(final_local_order, [order_params[-1]] * N, alpha=0.7, s=60)
    ax2.set_xlabel('Local Order Parameter')
    ax2.set_ylabel('Global Order Parameter')
    ax2.set_title('Local vs Global Synchronization (Final State)')
    ax2.grid(True, alpha=0.3)
    
    # Add node degree information
    degrees = np.zeros(N)
    for edge in edge_list:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
    
    # Color points by degree
    scatter = ax2.scatter(final_local_order, [order_params[-1]] * N, 
                         c=degrees, cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter, ax=ax2, label='Node Degree')
    
    # 3. Phase difference evolution
    if N >= 2:
        phase_diffs = np.zeros((len(T), len(edge_list)))
        for i, t in enumerate(T):
            for j, edge in enumerate(edge_list):
                node1, node2 = edge[0], edge[1]
                # Calculate wrapped phase difference
                diff = Theta[i, node2] - Theta[i, node1]
                phase_diffs[i, j] = np.angle(np.exp(1j * diff))
        
        # Show evolution of first few phase differences
        n_edges_show = min(10, len(edge_list))
        colors_edges = plt.cm.tab10(np.linspace(0, 1, n_edges_show))
        
        for j in range(n_edges_show):
            edge = edge_list[j]
            ax3.plot(T, phase_diffs[:, j], color=colors_edges[j], alpha=0.7,
                    label=f'Edge {edge[0]}-{edge[1]}')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Phase Difference (radians)')
        ax3.set_title('Pairwise Phase Differences')
        ax3.set_ylim([-np.pi, np.pi])
        ax3.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax3.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        if n_edges_show <= 5:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Synchronization onset analysis
    # Find when the system becomes synchronized (R > threshold)
    sync_threshold = 0.7
    sync_indices = np.where(order_params > sync_threshold)[0]
    
    if len(sync_indices) > 0:
        sync_onset = T[sync_indices[0]]
        ax4.plot(T, order_params, 'b-', linewidth=2, label='Order Parameter')
        ax4.axhline(sync_threshold, color='red', linestyle='--', 
                   label=f'Sync Threshold = {sync_threshold}')
        ax4.axvline(sync_onset, color='green', linestyle='--', 
                   label=f'Sync Onset = {sync_onset:.2f}')
        ax4.fill_between(T[sync_indices], 0, 1, alpha=0.2, color='green', 
                        label='Synchronized Region')
    else:
        ax4.plot(T, order_params, 'b-', linewidth=2, label='Order Parameter')
        ax4.axhline(sync_threshold, color='red', linestyle='--', 
                   label=f'Sync Threshold = {sync_threshold}')
        ax4.text(0.5, 0.5, 'No Synchronization\nDetected', 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Order Parameter R')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.set_title(f'Synchronization Analysis (α = {alpha})')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Phase Synchronization Analysis - Kuramoto Network', fontsize=14)
    plt.tight_layout()
    plt.savefig('kuramoto_synchronization_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Simple Kuramoto Network Simulation with Higher-Order Interactions")
    
    # Test different alpha values to see the effect of higher-order interactions
    alpha_values = [0.25, 0.5, 0.75, 1.0]
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"Running simulation with α = {alpha}")
        print(f"{'='*60}")
        
        # Run simulation
        results = simple_kuramoto_simulation(tmax=30, dt=0.05, omega=1.0, alpha=alpha, 
                                           visualization=(alpha == 0.5))  # Only visualize one case
        
        print(f"Results for α = {alpha}:")
        print(f"  Final order parameter: R = {results['final_order_parameter']:.4f}")
        print(f"  Average order parameter: R_avg = {results['average_order_parameter']:.4f}")
        
        # Save data for this alpha value
        filename = f'kuramoto_data_alpha_{alpha:.2f}.npz'
        np.savez(filename, **results)
        print(f"  Data saved to '{filename}'")
    
    print("\nSimulation completed for all α values!")
    print("Check the generated plots and data files for analysis.")
