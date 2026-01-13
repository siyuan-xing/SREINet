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

# Tensowflow version should be 2.13.0 or
# Print a formatted message to the terminal to display current time, date, and TensorFlow version
print("**********************************************")
print("*                                            *")
print("*    Tensorflow Version: " + tf.__version__ + "              *")
print("*                                            *")
print("*******************************|***************")

from packaging import version
if version.parse(tf.__version__) > version.parse('2.10.0'):
    print("Error: Tensorflow version is higher than 2.10. Please use Tensorflow 2.10 or lower.")
    sys.exit()


# Specify the model type
MY_MODEL = 'lorenz_96' # discrete sine Gordon model 

# Generate data based on the specified model type
node_num = 100
C = 2.0 #coupling strength

#initial conditions for Simulation
xn = np.linspace(-node_num/2/np.sqrt(C), node_num/2/np.sqrt(C), node_num)

#x0 = np.tanh(xn/np.sqrt(1-v**2))  
num_traj = 1
v = np.linspace(0.0, 0.99, num_traj)
data_initial_conditions= []
for i in range(num_traj):
        x0 =  np.tanh(xn/np.sqrt(1-v[i]**2))
        d_x0 = 1/np.sqrt(1-v[i]**2) * (1 - np.tanh(xn/np.sqrt(1-v[i]**2))**2)
        new_IC = np.concatenate((x0, d_x0), axis=0) # add velocity
        data_initial_conditions.append(new_IC)

data_T =  500.0       # Length of the data (T)
data_dt = 0.05       # Resolution of the data (dt)
myDG = DG.DataGenerator(data_initial_conditions,T=data_T, dt=data_dt)

t_arr, x_train, dx_train, guess_highest_order_polynomial = myDG.generate_dataset_by_model_name(MY_MODEL, C, node_num)

print("Guess highest order polynomial:", guess_highest_order_polynomial)

#plot x_train and dx_train
import matplotlib.pyplot as plt
print(x_train.shape)
#plt.scatter(x_train, dx_train)
#plt.show()

#Shuffle data
training_input, training_output = x_train, dx_train
# Optionally, save a copy of the training input for simulation
training_input_for_sim = training_input.copy() # Saving a copy of training input for simulation (Before shuffling)

training_input, training_output = myDG.shuffleData(training_input, training_output)

print("Training input shape:", training_input.shape)
print("Training output shape:", training_output.shape)


import pysindy as ps 

import time

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3), optimizer=ps.STLSQ(threshold=0.02))

start = time.time()

model.fit(training_input, t=t_arr, x_dot=training_output)

t_end = time.time()

print("Training time:", t_end - start)

model.print()