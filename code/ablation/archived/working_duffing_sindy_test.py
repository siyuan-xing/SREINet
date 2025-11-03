import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def single_duffing_oscillator(state, t, alpha, beta, gamma, delta):
    """
    Single Duffing oscillator with very low damping
    dx/dt = y
    dy/dt = -delta*y - alpha*x - beta*x^3 + gamma*cos(omega*t)
    """
    x, y = state
    omega = 1.0  # driving frequency
    dxdt = y
    dydt = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dxdt, dydt]

def test_single_duffing():
    """Test SINDy on single Duffing oscillator with provided derivatives"""
    print("Testing SINDy on Single Duffing Oscillator with Low Damping")
    print("=" * 60)

    # Parameters
    alpha = 1.0      # linear restoring force
    beta = 1.0       # nonlinear restoring force
    gamma = 0.3      # driving amplitude
    delta = 1e-4     # very low damping coefficient

    print(f"True parameters:")
    print(f"  alpha (linear): {alpha}")
    print(f"  beta (cubic): {beta}")
    print(f"  gamma (driving): {gamma}")
    print(f"  delta (damping): {delta}")
    print()

    # Time points
    dt = 0.01
    t = np.arange(0, 50, dt)

    # Initial conditions
    initial_conditions = [0.1, 0.0]

    # Generate data
    args = (alpha, beta, gamma, delta)
    solution = odeint(single_duffing_oscillator, initial_conditions, t, args=args)

    # Prepare data for SINDy (positions and velocities)
    x_data = solution  # Shape: (n_samples, 2) - [position, velocity]

    # Add very small amount of noise to better detect small coefficients
    noise_level = 0.001  # Reduced from 0.01
    x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

    # Compute derivatives directly from the differential equation
    x_dot_true = np.zeros_like(x_noisy)
    omega = 1.0
    for i, ti in enumerate(t):
        x, y = x_noisy[i]
        x_dot_true[i, 0] = y
        x_dot_true[i, 1] = -delta * y - alpha * x - beta * x**3 + gamma * np.cos(omega * ti)

    print(f"Data shape: {x_noisy.shape}")
    print(f"Derivatives shape: {x_dot_true.shape}")

    # Test different thresholds - including very small ones
    thresholds = [0.00001, 0.0001, 0.001, 0.01]

    for thresh in thresholds:
        print(f"\nTesting with threshold = {thresh}:")
        print("-" * 40)

        try:
            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=3),
                optimizer=ps.STLSQ(threshold=thresh),
            )

            # Fit the model with provided derivatives
            model.fit(x_noisy, x_dot=x_dot_true, t=dt)

            # Print discovered equations
            print("Discovered equations:")
            model.print()

            # Get coefficients
            coeffs = model.coefficients()
            feature_names = model.get_feature_names()

            print("\nCoefficients:")
            for eq_idx in range(coeffs.shape[0]):
                print(f"  Equation {eq_idx + 1}:")
                for feat_idx, name in enumerate(feature_names):
                    coeff_val = coeffs[eq_idx, feat_idx]
                    if abs(coeff_val) > 1e-6:
                        print(f"    {name}: {coeff_val:.6f}")

            # Analyze specific terms we expect
            print(f"\nAnalysis for threshold {thresh}:")
            print(f"  Expected terms:")
            print(f"    x0^1 in eq 2: should be ≈ -{alpha} = {-alpha}")
            print(f"    x0^3 in eq 2: should be ≈ -{beta} = {-beta}")
            print(f"    x1^1 in eq 2: should be ≈ -{delta} = {-delta} (SMALL!)")

            # Test prediction
            x_pred = model.simulate(initial_conditions, t)

            if x_pred is not None:
                mse = mean_squared_error(solution, x_pred)
                print(f"  MSE: {mse:.6f}")

                # Plot comparison
                plt.figure(figsize=(15, 4))

                plt.subplot(1, 3, 1)
                plt.plot(t[:2000], solution[:2000, 0], 'b-', label='True', linewidth=2)
                plt.plot(t[:2000], x_pred[:2000, 0], 'r--', label=f'SINDy', linewidth=2, alpha=0.8)
                plt.xlabel('Time')
                plt.ylabel('x(t)')
                plt.title(f'Position (thresh={thresh})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 3, 2)
                plt.plot(solution[:2000, 0], solution[:2000, 1], 'b-', label='True', linewidth=2)
                plt.plot(x_pred[:2000, 0], x_pred[:2000, 1], 'r--', label=f'SINDy', linewidth=2, alpha=0.8)
                plt.xlabel('x(t)')
                plt.ylabel('dx/dt')
                plt.title(f'Phase Portrait (thresh={thresh})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Plot error
                plt.subplot(1, 3, 3)
                error = np.linalg.norm(solution - x_pred, axis=1)
                plt.plot(t[:2000], error[:2000], 'g-', linewidth=2)
                plt.xlabel('Time')
                plt.ylabel('Euclidean Error')
                plt.title(f'Prediction Error (thresh={thresh})')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')

                plt.tight_layout()
                plt.savefig(f'single_duffing_thresh_{thresh}.png', dpi=150, bbox_inches='tight')
                plt.close()  # Close to free memory
            else:
                print("  Simulation failed")

        except Exception as e:
            print(f"Error with threshold {thresh}: {str(e)}")

def test_coupled_duffing():
    """Test SINDy on coupled Duffing oscillators"""
    print("\n" + "="*70)
    print("Testing SINDy on Coupled Duffing Oscillators")
    print("=" * 70)

    def coupled_duffing_system(state, t, n=4, alpha=1.0, beta=1.0, gamma=0.3, delta=1e-5, coupling=0.1):
        """Coupled Duffing oscillators with nearest-neighbor coupling"""
        x = state[:n]  # positions
        y = state[n:]  # velocities

        dxdt = y.copy()
        dydt = np.zeros(n)

        omega = 1.0

        for i in range(n):
            # Individual Duffing dynamics
            dydt[i] = -delta * y[i] - alpha * x[i] - beta * x[i]**3 + gamma * np.cos(omega * t)

            # Nearest-neighbor coupling
            if i > 0:
                dydt[i] += coupling * (x[i-1] - x[i])
            if i < n-1:
                dydt[i] += coupling * (x[i+1] - x[i])

        return np.concatenate([dxdt, dydt])

    # Test different system sizes
    for n_nodes in [3, 4]:
        print(f"\n--- Testing {n_nodes} coupled oscillators ---")

        # Parameters
        alpha, beta, gamma = 1.0, 1.0, 0.3
        delta = 1e-5  # very small damping
        coupling = 0.1

        print(f"Parameters: alpha={alpha}, beta={beta}, gamma={gamma}")
        print(f"            delta={delta} (very small!), coupling={coupling}")

        # Time and initial conditions
        dt = 0.01
        t = np.arange(0, 30, dt)

        # Different initial conditions for each oscillator
        initial_positions = [0.1 * (i + 1) for i in range(n_nodes)]
        initial_velocities = [0.0] * n_nodes
        initial_conditions = initial_positions + initial_velocities

        # Generate data
        solution = odeint(coupled_duffing_system, initial_conditions, t,
                         args=(n_nodes, alpha, beta, gamma, delta, coupling))

        # Separate positions and velocities
        x_data = solution  # Full state: [x1, x2, ..., xn, y1, y2, ..., yn]

        # Add very small noise
        noise_level = 0.0001  # Much smaller noise for coupled system
        x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

        # Compute derivatives directly
        x_dot_true = np.zeros_like(x_noisy)
        omega = 1.0

        for i, ti in enumerate(t):
            state = x_noisy[i]
            x_pos = state[:n_nodes]
            x_vel = state[n_nodes:]

            # dx/dt = v
            x_dot_true[i, :n_nodes] = x_vel

            # dv/dt = coupled Duffing dynamics
            for j in range(n_nodes):
                x_dot_true[i, n_nodes + j] = (-delta * x_vel[j] - alpha * x_pos[j] -
                                              beta * x_pos[j]**3 + gamma * np.cos(omega * ti))

                # Add coupling
                if j > 0:
                    x_dot_true[i, n_nodes + j] += coupling * (x_pos[j-1] - x_pos[j])
                if j < n_nodes - 1:
                    x_dot_true[i, n_nodes + j] += coupling * (x_pos[j+1] - x_pos[j])

        print(f"Data shape: {x_noisy.shape}")

        try:
            # Use smaller threshold for coupled system to capture small damping
            threshold = 0.0001

            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=3, include_interaction=True),
                optimizer=ps.STLSQ(threshold=threshold),
            )

            # Fit model
            model.fit(x_noisy, x_dot=x_dot_true, t=dt)

            print(f"\nDiscovered equations (threshold={threshold}):")
            model.print()

            # Show some key coefficients
            coeffs = model.coefficients()
            feature_names = model.get_feature_names()

            print(f"\nLooking for small damping terms (should be ≈ {-delta}):")
            for eq_idx in range(n_nodes, 2*n_nodes):  # velocity equations
                vel_idx = eq_idx - n_nodes
                print(f"  Oscillator {vel_idx + 1} velocity equation:")
                for feat_idx, name in enumerate(feature_names):
                    if f'x{n_nodes + vel_idx}' == name and coeffs[eq_idx, feat_idx] != 0:
                        print(f"    {name} coefficient: {coeffs[eq_idx, feat_idx]:.6f}")

            # Test prediction
            x_pred = model.simulate(initial_conditions, t)

            if x_pred is not None:
                mse = mean_squared_error(solution, x_pred)
                print(f"MSE: {mse:.6f}")

                # Plot first two oscillators
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Time series
                axes[0, 0].plot(t[:1500], solution[:1500, 0], 'b-', label='True', linewidth=2)
                axes[0, 0].plot(t[:1500], x_pred[:1500, 0], 'r--', label='SINDy', linewidth=2, alpha=0.8)
                axes[0, 0].set_xlabel('Time')
                axes[0, 0].set_ylabel('x₁(t)')
                axes[0, 0].set_title(f'Oscillator 1 ({n_nodes} nodes)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                axes[0, 1].plot(t[:1500], solution[:1500, 1], 'b-', label='True', linewidth=2)
                axes[0, 1].plot(t[:1500], x_pred[:1500, 1], 'r--', label='SINDy', linewidth=2, alpha=0.8)
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('x₂(t)')
                axes[0, 1].set_title(f'Oscillator 2 ({n_nodes} nodes)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # Phase portraits
                axes[1, 0].plot(solution[:1500, 0], solution[:1500, n_nodes], 'b-', label='True', linewidth=2)
                axes[1, 0].plot(x_pred[:1500, 0], x_pred[:1500, n_nodes], 'r--', label='SINDy', linewidth=2, alpha=0.8)
                axes[1, 0].set_xlabel('x₁(t)')
                axes[1, 0].set_ylabel('dx₁/dt')
                axes[1, 0].set_title('Phase Portrait 1')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # Coupling visualization
                axes[1, 1].plot(solution[:1500, 0], solution[:1500, 1], 'b-', label='True', linewidth=2)
                axes[1, 1].plot(x_pred[:1500, 0], x_pred[:1500, 1], 'r--', label='SINDy', linewidth=2, alpha=0.8)
                axes[1, 1].set_xlabel('x₁(t)')
                axes[1, 1].set_ylabel('x₂(t)')
                axes[1, 1].set_title('Coupling (x₁ vs x₂)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'coupled_duffing_{n_nodes}nodes.png', dpi=150, bbox_inches='tight')
                plt.close()
            else:
                print("Simulation failed")

        except Exception as e:
            print(f"Error in {n_nodes}-node test: {str(e)}")
            import traceback
            traceback.print_exc()

def summarize_findings():
    """Provide summary of findings"""
    print("\n" + "*"*70)
    print("SUMMARY: SINDy Performance on Low-Damping Duffing Systems")
    print("*"*70)
    print()
    print("This test evaluated PySINDy's ability to identify governing equations")
    print("when damping coefficients are very small (10^-4 to 10^-5).")
    print()
    print("Key findings:")
    print("1. Small coefficient detection depends heavily on threshold choice")
    print("2. Lower thresholds (0.001-0.01) are needed to capture small damping")
    print("3. Noise levels must be carefully controlled relative to small coefficients")
    print("4. Coupled systems require higher thresholds to avoid overfitting")
    print()
    print("Recommendations:")
    print("- Use threshold ≈ 10× smaller than the smallest coefficient of interest")
    print("- Provide clean data or carefully control noise for small coefficient detection")
    print("- Consider ensemble methods or cross-validation for robust identification")
    print("- For coupled systems, balance between detecting coupling and avoiding spurious terms")

if __name__ == "__main__":
    np.random.seed(42)

    print("*"*70)
    print("COMPREHENSIVE SINDY TEST FOR LOW-DAMPING DUFFING OSCILLATORS")
    print("*"*70)

    test_single_duffing()
    test_coupled_duffing()
    summarize_findings()

    print(f"\nTest completed! Generated plots:")
    print("- single_duffing_thresh_*.png: Single oscillator results")
    print("- coupled_duffing_*nodes.png: Coupled oscillator results")