import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
from sklearn.metrics import mean_squared_error

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

def test_simple_duffing():
    """Test SINDy on simple Duffing oscillator"""
    print("Testing SINDy on Single Duffing Oscillator with Low Damping")
    print("=" * 60)

    # Parameters
    alpha = 1.0      # linear restoring force
    beta = 1.0       # nonlinear restoring force
    gamma = 0.3      # driving amplitude
    delta = 1e-4     # very low damping coefficient

    print(f"True parameters: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")

    # Time points
    dt = 0.01
    t = np.arange(0, 50, dt)

    # Initial conditions
    initial_conditions = [0.1, 0.0]

    # Generate data
    args = (alpha, beta, gamma, delta)
    solution = odeint(single_duffing_oscillator, initial_conditions, t, args=args)

    # Use only position data - let SINDy compute derivatives
    x_data = solution[:, 0:1]  # Shape: (n_samples, 1)

    # Add small amount of noise
    noise_level = 0.01
    x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

    print(f"Data shape: {x_noisy.shape}")

    # Create SINDy model with different thresholds to capture small damping
    thresholds = [0.001, 0.01, 0.1]

    for thresh in thresholds:
        print(f"\nTesting with threshold = {thresh}:")
        print("-" * 40)

        try:
            model = ps.SINDy(
                feature_library=ps.PolynomialLibrary(degree=3),
                optimizer=ps.STLSQ(threshold=thresh),
                differentiation_method=ps.FiniteDifference()
            )

            # Fit the model
            model.fit(x_noisy, t=dt)

            # Print discovered equations
            print("Discovered equations:")
            model.print()

            # Get coefficients
            coeffs = model.coefficients()
            feature_names = model.get_feature_names()

            print("\nNon-zero coefficients:")
            for i, name in enumerate(feature_names):
                for eq_idx in range(coeffs.shape[0]):
                    if abs(coeffs[eq_idx, i]) > 1e-6:
                        print(f"  Eq {eq_idx+1}, {name}: {coeffs[eq_idx, i]:.6f}")

            # Test prediction
            x_pred = model.simulate(initial_conditions, t)

            if x_pred is not None and x_pred.shape == solution.shape:
                mse = mean_squared_error(solution, x_pred)
                print(f"\nMean Squared Error: {mse:.6f}")

                # Plot comparison
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 2, 1)
                plt.plot(t[:1000], solution[:1000, 0], 'b-', label='True', linewidth=2)
                plt.plot(t[:1000], x_pred[:1000, 0], 'r--', label=f'SINDy (thresh={thresh})', linewidth=2)
                plt.xlabel('Time')
                plt.ylabel('x(t)')
                plt.title('Time Series Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                plt.plot(solution[:1000, 0], solution[:1000, 1], 'b-', label='True', linewidth=2)
                plt.plot(x_pred[:1000, 0], x_pred[:1000, 1], 'r--', label=f'SINDy (thresh={thresh})', linewidth=2)
                plt.xlabel('x(t)')
                plt.ylabel('dx/dt')
                plt.title('Phase Portrait')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'duffing_thresh_{thresh}.png', dpi=150, bbox_inches='tight')
                plt.show()
            else:
                print("Simulation failed or returned unexpected shape")

        except Exception as e:
            print(f"Error with threshold {thresh}: {str(e)}")
            import traceback
            traceback.print_exc()

def test_coupled_simple():
    """Test SINDy on simple coupled oscillators"""
    print("\n" + "="*60)
    print("Testing SINDy on Coupled Duffing Oscillators")
    print("=" * 60)

    def coupled_duffing(state, t, n=3, alpha=1.0, beta=1.0, gamma=0.3, delta=1e-5, coupling=0.1):
        """Simple coupled Duffing system"""
        # state = [x1, y1, x2, y2, x3, y3]
        x = state[::2]  # positions
        y = state[1::2]  # velocities

        dxdt = y.copy()
        dydt = np.zeros(n)

        for i in range(n):
            # Individual Duffing dynamics
            dydt[i] = -delta * y[i] - alpha * x[i] - beta * x[i]**3 + gamma * np.cos(t)

            # Simple nearest-neighbor coupling
            if i > 0:
                dydt[i] += coupling * (x[i-1] - x[i])
            if i < n-1:
                dydt[i] += coupling * (x[i+1] - x[i])

        # Interleave derivatives
        derivatives = np.zeros(2*n)
        derivatives[::2] = dxdt
        derivatives[1::2] = dydt
        return derivatives

    # Parameters
    n_nodes = 3
    alpha, beta, gamma = 1.0, 1.0, 0.3
    delta = 1e-5  # very small damping
    coupling = 0.1

    print(f"System: {n_nodes} coupled oscillators, damping={delta}")

    # Time and initial conditions
    dt = 0.01
    t = np.arange(0, 30, dt)
    initial_conditions = [0.1*(i+1) for i in range(n_nodes)] + [0.0]*n_nodes  # [x1,x2,x3,y1,y2,y3]

    # Generate data
    solution = odeint(coupled_duffing, initial_conditions, t,
                     args=(n_nodes, alpha, beta, gamma, delta, coupling))

    # Use only positions for SINDy
    x_data = solution[:, ::2]  # positions only

    # Add noise
    noise_level = 0.005
    x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

    print(f"Data shape: {x_noisy.shape}")

    try:
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(degree=3, include_interaction=True),
            optimizer=ps.STLSQ(threshold=0.02),
            differentiation_method=ps.FiniteDifference()
        )

        model.fit(x_noisy, t=dt)

        print("Discovered equations:")
        model.print()

        # Show coefficients
        coeffs = model.coefficients()
        feature_names = model.get_feature_names()

        print("\nSignificant coefficients:")
        for eq_idx in range(coeffs.shape[0]):
            print(f"Equation {eq_idx + 1}:")
            for feat_idx, name in enumerate(feature_names):
                if abs(coeffs[eq_idx, feat_idx]) > 1e-3:
                    print(f"  {name}: {coeffs[eq_idx, feat_idx]:.4f}")

        # Test prediction
        x_pred = model.simulate(x_noisy[0], t)
        if x_pred is not None:
            mse = mean_squared_error(x_data, x_pred)
            print(f"\nMSE: {mse:.6f}")

            # Plot first two oscillators
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(t[:1000], x_data[:1000, 0], 'b-', label='True', linewidth=2)
            axes[0].plot(t[:1000], x_pred[:1000, 0], 'r--', label='SINDy', linewidth=2)
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('x₁(t)')
            axes[0].set_title('Oscillator 1')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(t[:1000], x_data[:1000, 1], 'b-', label='True', linewidth=2)
            axes[1].plot(t[:1000], x_pred[:1000, 1], 'r--', label='SINDy', linewidth=2)
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('x₂(t)')
            axes[1].set_title('Oscillator 2')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('coupled_duffing_simple.png', dpi=150, bbox_inches='tight')
            plt.show()

    except Exception as e:
        print(f"Error in coupled test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed and matplotlib backend
    np.random.seed(42)
    plt.matplotlib.use('Agg')  # Non-interactive backend

    test_simple_duffing()
    test_coupled_simple()