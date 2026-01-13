# Import os library to access environment variables and operating system functionalities
import os

# Set environment variable to avoid duplicate library errors, specifically for OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set TensorFlow log level to '2' to suppress most messages except warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append("../utilities")

# Import necessary libraries for deep learning and numerical computation
import tensorflow as tf
import numpy as np
import DataGenerator as DG


# Specify the model type
MY_MODEL = 'dnls_split'  # discrete nonlinear Schrödinger system

DNLS_SITES = 10  # Number of lattice sites (state dimension = 2 * DNLS_SITES)
STATE_DIM = DNLS_SITES * 2
# Generate N random numbers from a normal distribution (mean=0, std=1)

num_traj = 1
data_initial_conditions = DG.DataGenerator.generate_initial_conditions(STATE_DIM, num_traj) #np.random.normal(1, 2, STATE_DIM)

#data_initial_conditions = np.random.normal(0.0, 1.0, STATE_DIM)
data_T =  60.0       # Length of the data (T)
data_dt = 0.001       # Resolution of the data (dt)
myDG = DG.DataGenerator(data_initial_conditions,T=data_T, dt=data_dt, derivative_mode="exact")
t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG.generate_dataset_by_model_name(MY_MODEL, method='DOP853')

print("Guess highest order polynomial:", guess_highest_order_polynomial)

#plot x_train and dx_train
import matplotlib.pyplot as plt
print(x_train.shape)

#Shuffle data
training_input, training_output = x_train, dx_train
# Optionally, save a copy of the training input for simulation
training_input_for_sim = training_input.copy() # Saving a copy of training input for simulation (Before shuffling)

#training_input, training_output = myDG.shuffleData(training_input, training_output)

print("Training input shape:", training_input.shape)
print("Training output shape:", training_output.shape)


import pysindy as ps 

import time

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2), optimizer=ps.STLSQ(threshold=0.01))

start = time.time()

model.fit(training_input, t=t_arr, x_dot=training_output)

t_end = time.time()

print("Training time:", t_end - start)

model.print()

# =============================================================================
# Comprehensive Error Analysis
# =============================================================================

print("\n" + "="*60)
print("COMPREHENSIVE ERROR ANALYSIS")
print("="*60)

# 1. Coefficient Analysis and Sparsity
print("\n1. COEFFICIENT ANALYSIS AND SPARSITY")
print("-" * 40)

# Get the coefficient matrix
coefficients = model.coefficients()
print(f"Coefficient matrix shape: {coefficients.shape}")

# Calculate sparsity
total_coefficients = coefficients.size
nonzero_coefficients = np.count_nonzero(coefficients)
sparsity = 1 - (nonzero_coefficients / total_coefficients)
print(f"Total coefficients: {total_coefficients}")
print(f"Non-zero coefficients: {nonzero_coefficients}")
print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")

# Show non-zero coefficients
nonzero_indices = np.where(coefficients != 0)
print(f"\nNon-zero coefficient positions:")
for i, (row, col) in enumerate(zip(nonzero_indices[0], nonzero_indices[1])):
    print(f"  Row {row}, Col {col}: {coefficients[row, col]:.6f}")

# 2. Training Data Prediction Error
print("\n2. TRAINING DATA PREDICTION ERROR")
print("-" * 40)

# Predict derivatives using the fitted model
dx_pred = model.predict(training_input)
dx_true = training_output

# Calculate prediction errors
mse = np.mean((dx_pred - dx_true)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(dx_pred - dx_true))
max_error = np.max(np.abs(dx_pred - dx_true))

print(f"Mean Squared Error (MSE): {mse:.8f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.8f}")
print(f"Mean Absolute Error (MAE): {mae:.8f}")
print(f"Maximum Absolute Error: {max_error:.8f}")

# 3. Trajectory Prediction Error
print("\n3. TRAJECTORY PREDICTION ERROR")
print("-" * 40)

# Simulate the identified system

# Use the fitted model to simulate
x_sim = model.simulate(training_input[0], t_arr)
    
# Calculate trajectory errors
traj_mse = np.mean((x_sim - training_input)**2)
traj_rmse = np.sqrt(traj_mse)
traj_mae = np.mean(np.abs(x_sim - training_input))
    
print(f"Trajectory MSE: {traj_mse:.8f}")
print(f"Trajectory RMSE: {traj_rmse:.8f}")
print(f"Trajectory MAE: {traj_mae:.8f}")
    
# 4. Model Complexity Analysis
print("\n5. MODEL COMPLEXITY ANALYSIS")
print("-" * 40)

# Count active terms per equation
active_terms_per_eq = np.count_nonzero(coefficients, axis=1)
print(f"Active terms per equation:")
for i in range(min(10, STATE_DIM)):  # Show first 10 equations
    print(f"  Equation {i}: {active_terms_per_eq[i]} active terms")

avg_active_terms = np.mean(active_terms_per_eq)
print(f"\nAverage active terms per equation: {avg_active_terms:.2f}")

# 7. Summary Statistics
print("\n7. SUMMARY STATISTICS")
print("-" * 40)
print(f"Model Type: {type(model).__name__}")
print(f"Feature Library: {type(model.feature_library).__name__}")
print(f"Optimizer: {type(model.optimizer).__name__}")
print(f"Training Time: {t_end - start:.4f} seconds")
print(f"Data Points: {len(t_arr)}")
print(f"Time Range: {t_arr[0]:.3f} to {t_arr[-1]:.3f}")
print(f"Time Step: {data_dt:.6f}")
print(f"Overall Model Performance: RMSE = {rmse:.8f}, Sparsity = {sparsity*100:.2f}%")

print("\n" + "="*60)
print("ERROR ANALYSIS COMPLETE")
print("="*60)
