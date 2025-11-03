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

def generate_duffing_data(t, initial_conditions, alpha=1.0, beta=1.0, gamma=0.3, delta=1e-4):
    """Generate synthetic data for single Duffing oscillator"""
    args = (alpha, beta, gamma, delta)
    solution = odeint(single_duffing_oscillator, initial_conditions, t, args=args)
    return solution

def test_single_duffing_sindy():
    """Test SINDy on single Duffing oscillator with low damping"""
    print("=" * 60)
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
    data = generate_duffing_data(t, initial_conditions, alpha, beta, gamma, delta)
    x_data = data  # Both position and velocity

    # Add small amount of noise
    noise_level = 0.01
    x_noisy = x_data + noise_level * np.random.normal(size=x_data.shape)

    # Create SINDy model
    # Use polynomial features up to degree 3 to capture x^3 term
    poly_lib = ps.PolynomialLibrary(degree=3)

    # Try different optimizers
    optimizers = [
        ("STLSQ", ps.STLSQ(threshold=0.01)),
        ("SR3", ps.SR3(threshold=0.01, nu=0.01))
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

            print("Coefficients:")
            for i, name in enumerate(feature_names):
                if abs(coeffs[0, i]) > 1e-6:  # Only show non-zero coefficients
                    print(f"  {name}: {coeffs[0, i]:.6f}")
            print()

            # Test prediction
            x_pred = model.simulate(initial_conditions, t)
            mse = mean_squared_error(x_data, x_pred)
            print(f"Mean Squared Error: {mse:.6f}")

            # Plot comparison
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.plot(t[:1000], x_data[:1000, 0], 'b-', label='True', linewidth=2)
            plt.plot(t[:1000], x_pred[:1000, 0], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('x(t)')
            plt.title(f'Time Series Comparison - {opt_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(x_data[:1000, 0], x_data[:1000, 1], 'b-', label='True', linewidth=2)
            plt.plot(x_pred[:1000, 0], x_pred[:1000, 1], 'r--', label=f'SINDy ({opt_name})', linewidth=2)
            plt.xlabel('x(t)')
            plt.ylabel('dx/dt')
            plt.title(f'Phase Portrait - {opt_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'single_duffing_{opt_name.lower()}_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

        except Exception as e:
            print(f"Error with {opt_name}: {str(e)}")

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Test single Duffing oscillator
    test_single_duffing_sindy()