

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
from sklearn.metrics import mean_squared_error

def coupled_duffing_oscillators(state, t, n_nodes, alpha, beta, gamma, delta, coupling_strength):
    """
    Coupled Duffing oscillators with very low damping
    For node i:
    dx_i/dt = y_i
    dy_i/dt = -delta*y_i - alpha*x_i - beta*x_i^3 + gamma*cos(omega*t) + coupling_term

    Coupling term = coupling_strength * sum_j(x_j - x_i) for all neighbors j
    """
    omega = 1.0  # driving frequency

    # Reshape state vector: [x1, y1, x2, y2, ..., xn, yn]
    x = state[::2]  # positions [x1, x2, ..., xn]
    y = state[1::2]  # velocities [y1, y2, ..., yn]

    dxdt = y.copy()
    dydt = np.zeros(n_nodes)

    for i in range(n_nodes):
        # Individual Duffing dynamics
        dydt[i] = -delta * y[i] - alpha * x[i] - beta * x[i]**3 + gamma * np.cos(omega * t)

        # Coupling term (all-to-all coupling)
        coupling_term = 0.0
        for j in range(n_nodes):
            if i != j:
                coupling_term += coupling_strength * (x[j] - x[i])
        dydt[i] += coupling_term

    # Interleave dx and dy: [dx1, dy1, dx2, dy2, ..., dxn, dyn]
    derivatives = np.zeros(2 * n_nodes)
    derivatives[::2] = dxdt
    derivatives[1::2] = dydt

    return derivatives

def generate_coupled_duffing_data(t, initial_conditions, n_nodes=4, alpha=1.0, beta=1.0,
                                  gamma=0.3, delta=1e-4, coupling_strength=0.1):
    """Generate synthetic data for coupled Duffing oscillators"""
    args = (n_nodes, alpha, beta, gamma, delta, coupling_strength)
    solution = odeint(coupled_duffing_oscillators, initial_conditions, t, args=args)
    return solution

def test_coupled_duffing_sindy():
    """Test SINDy on coupled Duffing oscillators with low damping"""
    print("=" * 60)
    print("Testing SINDy on Coupled Duffing Oscillators with Low Damping")
    print("=" * 60)

    # Parameters
    n_nodes = 4      # number of coupled oscillators
    alpha = 1.0      # linear restoring force
    beta = 1.0       # nonlinear restoring force
    gamma = 0.3      # driving amplitude
    delta = 1e-5     # very low damping coefficient
    coupling_strength = 0.1  # coupling strength

    print(f"System parameters:")
    print(f"  Number of nodes: {n_nodes}")
    print(f"  alpha (linear): {alpha}")
    print(f"  beta (cubic): {beta}")
    print(f"  gamma (driving): {gamma}")
    print(f"  delta (damping): {delta}")
    print(f"  coupling strength: {coupling_strength}")
    print()

    # Time points
    dt = 0.01
    t = np.arange(0, 30, dt)

    # Initial conditions: slightly different for each oscillator
    initial_conditions = []
    for i in range(n_nodes):
        initial_conditions.extend([0.1 * (i + 1), 0.0])  # [x1, y1, x2, y2, ...]

    # Generate data
    data = generate_coupled_duffing_data(t, initial_conditions, n_nodes, alpha, beta,
                                         gamma, delta, coupling_strength)

    # Extract positions only for SINDy analysis
    x_data = data[:, ::2]  # positions [x1, x2, x3, x4]

    # Add small amount of noise
    noise_level = 0.005
    x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

    print(f"Data shape: {x_noisy.shape}")
    print(f"Time points: {len(t)}")
    print()

    # Create SINDy model for coupled system
    # Use polynomial features and custom library to capture coupling terms
    poly_lib = ps.PolynomialLibrary(degree=3, include_interaction=True)

    # Try different optimizers
    optimizers = [
        ("STLSQ", ps.STLSQ(threshold=1e-6)),
        ("SR3", ps.SR3(threshold=1e-6, nu=0.02)),
    ]

    for opt_name, optimizer in optimizers:
        print(f"Testing with {opt_name} optimizer:")
        print("-" * 40)

        try:
            model = ps.SINDy(
                feature_library=poly_lib,
                optimizer=optimizer,
                differentiation_method=ps.FiniteDifference()
            )

            # Fit the model
            model.fit(x_noisy, t=t)

            # Print discovered equations
            print("Discovered equations:")
            model.print()
            print()

            # Get coefficients
            coeffs = model.coefficients()
            feature_names = model.get_feature_names()

            print("Non-zero coefficients:")
            for eq_idx in range(coeffs.shape[0]):
                print(f"Equation {eq_idx + 1}:")
                for feat_idx, name in enumerate(feature_names):
                    if abs(coeffs[eq_idx, feat_idx]) > 1e-4:
                        print(f"  {name}: {coeffs[eq_idx, feat_idx]:.6f}")
            print()

            # Test prediction
            x_pred = model.simulate(x_noisy[0], t)
            mse = mean_squared_error(x_data, x_pred)
            print(f"Mean Squared Error: {mse:.6f}")

            # Plot comparison for first two oscillators
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Time series for oscillator 1
            axes[0, 0].plot(t[:1500], x_data[:1500, 0], 'b-', label='True', linewidth=2)
            axes[0, 0].plot(t[:1500], x_pred[:1500, 0], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('x₁(t)')
            axes[0, 0].set_title(f'Oscillator 1 - {opt_name}')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Time series for oscillator 2
            axes[0, 1].plot(t[:1500], x_data[:1500, 1], 'b-', label='True', linewidth=2)
            axes[0, 1].plot(t[:1500], x_pred[:1500, 1], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('x₂(t)')
            axes[0, 1].set_title(f'Oscillator 2 - {opt_name}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Phase portrait oscillator 1
            y_true = data[:, 1]  # velocity of oscillator 1
            y_pred = model.differentiate(x_pred, t=dt)[:, 0] if x_pred.shape[1] > 0 else np.zeros_like(t)

            axes[1, 0].plot(x_data[:1500, 0], y_true[:1500], 'b-', label='True', linewidth=2)
            axes[1, 0].plot(x_pred[:1500, 0], y_pred[:1500], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
            axes[1, 0].set_xlabel('x₁(t)')
            axes[1, 0].set_ylabel('dx₁/dt')
            axes[1, 0].set_title(f'Phase Portrait 1 - {opt_name}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Coupling visualization: x1 vs x2
            axes[1, 1].plot(x_data[:1500, 0], x_data[:1500, 1], 'b-', label='True', linewidth=2)
            axes[1, 1].plot(x_pred[:1500, 0], x_pred[:1500, 1], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
            axes[1, 1].set_xlabel('x₁(t)')
            axes[1, 1].set_ylabel('x₂(t)')
            axes[1, 1].set_title(f'Coupling (x₁ vs x₂) - {opt_name}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'coupled_duffing_{opt_name.lower()}_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Plot all oscillators time series
            plt.figure(figsize=(15, 8))
            for i in range(n_nodes):
                plt.subplot(2, 2, i + 1)
                plt.plot(t[:1000], x_data[:1000, i], 'b-', label='True', linewidth=2)
                plt.plot(t[:1000], x_pred[:1000, i], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
                plt.xlabel('Time')
                plt.ylabel(f'x_{i+1}(t)')
                plt.title(f'Oscillator {i+1}')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'all_oscillators_{opt_name.lower()}.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"Error with {opt_name}: {str(e)}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*60 + "\n")

def test_different_node_counts():
    """Test SINDy performance with different numbers of coupled oscillators"""
    print("Testing SINDy with different numbers of coupled oscillators")
    print("=" * 60)

    node_counts = [3, 4, 5]
    results = {}

    for n_nodes in node_counts:
        print(f"\nTesting with {n_nodes} coupled oscillators:")
        print("-" * 40)

        # Parameters
        alpha, beta, gamma = 1.0, 1.0, 0.3
        delta = 1e-5
        coupling_strength = 0.1

        # Time points
        dt = 0.01
        t = np.arange(0, 20, dt)

        # Initial conditions
        initial_conditions = []
        for i in range(n_nodes):
            initial_conditions.extend([0.1 * (i + 1), 0.0])

        # Generate data
        data = generate_coupled_duffing_data(t, initial_conditions, n_nodes,
                                             alpha, beta, gamma, delta, coupling_strength)
        x_data = data[:, ::2]

        # Add noise
        noise_level = 0.005
        x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

        try:
            # Use STLSQ for this test
            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=3, include_interaction=True),
                optimizer=ps.STLSQ(threshold=0.02),
                differentiation_method=ps.FiniteDifference()
            )

            model.fit(x_noisy, t=t)
            x_pred = model.simulate(x_noisy[0], t)
            mse = mean_squared_error(x_data, x_pred)

            results[n_nodes] = mse
            print(f"MSE for {n_nodes} nodes: {mse:.6f}")

        except Exception as e:
            print(f"Error with {n_nodes} nodes: {str(e)}")
            results[n_nodes] = float('inf')

    print("\nSummary of results:")
    print("-" * 30)
    for nodes, mse in results.items():
        print(f"{nodes} nodes: MSE = {mse:.6f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Test coupled Duffing oscillators
    test_coupled_duffing_sindy()

    # Test different node counts
    test_different_node_counts()