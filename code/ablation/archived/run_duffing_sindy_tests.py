"""
Comprehensive test suite for PySINDy performance on Duffing oscillators with very low damping coefficients.

This script tests:
1. Single Duffing oscillator with damping coefficient 10^-4
2. Coupled Duffing oscillators (3-5 nodes) with damping coefficient 10^-5

The goal is to evaluate if SINDy can correctly identify the governing equations
when some coefficients are very small.
"""

import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from duffing_low_damping_sindy_test import test_single_duffing_sindy
from coupled_duffing_low_damping_sindy_test import test_coupled_duffing_sindy, test_different_node_counts

def main():
    """Run comprehensive SINDy tests on low-damping Duffing oscillators"""
    print("*" * 80)
    print("COMPREHENSIVE SINDY TEST FOR LOW-DAMPING DUFFING OSCILLATORS")
    print("*" * 80)
    print()
    print("This test suite evaluates PySINDy's ability to identify governing equations")
    print("when damping coefficients are very small (10^-4 to 10^-5).")
    print()
    print("Test scenarios:")
    print("1. Single Duffing oscillator (damping = 10^-4)")
    print("2. Coupled Duffing oscillators with 3-5 nodes (damping = 10^-5)")
    print()
    print("Expected challenges:")
    print("- Very small damping terms may be below SINDy's threshold detection")
    print("- Numerical precision issues with small coefficients")
    print("- Increased sensitivity to noise")
    print()

    # Set up matplotlib for better plots
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10

    # Set random seed for reproducibility
    np.random.seed(42)

    try:
        # Test 1: Single Duffing Oscillator
        print("\n" + "=" * 80)
        print("TEST 1: SINGLE DUFFING OSCILLATOR")
        print("=" * 80)
        test_single_duffing_sindy()

    except Exception as e:
        print(f"Error in single Duffing test: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Test 2: Coupled Duffing Oscillators
        print("\n" + "=" * 80)
        print("TEST 2: COUPLED DUFFING OSCILLATORS")
        print("=" * 80)
        test_coupled_duffing_sindy()

    except Exception as e:
        print(f"Error in coupled Duffing test: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Test 3: Different node counts
        print("\n" + "=" * 80)
        print("TEST 3: SCALABILITY TEST (DIFFERENT NODE COUNTS)")
        print("=" * 80)
        test_different_node_counts()

    except Exception as e:
        print(f"Error in scalability test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "*" * 80)
    print("TEST SUITE COMPLETED")
    print("*" * 80)
    print()
    print("Summary of findings:")
    print("- Check the generated plots for visual comparison of SINDy vs true dynamics")
    print("- Look at the discovered equations to see if small damping terms were identified")
    print("- MSE values indicate the quality of the SINDy approximation")
    print()
    print("Key questions to evaluate:")
    print("1. Did SINDy identify the very small damping coefficients?")
    print("2. How does performance degrade with system size (more coupled oscillators)?")
    print("3. Which optimizer (STLSQ, SR3, FROLS) performs best for small coefficients?")
    print("4. What threshold values work best for detecting small terms?")

if __name__ == "__main__":
    main()