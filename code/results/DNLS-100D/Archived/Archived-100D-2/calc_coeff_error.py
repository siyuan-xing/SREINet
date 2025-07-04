import os
import numpy as np
import re

# Step 1: Define the directory paths and corresponding noise levels

# Step 2: Initialize an empty list to store accuracies for each noise level
all_accuracies = []

# Define the real coefficients (absolute values) - small to large
real_coeff = np.array([1.0,1.0, 2.0, 2.0, 4.0])

# Step 3: Loop through each noise level directory
    # Initialize a list to store accuracies for the current noise level

accuracies = []
    
    # Step 4: Loop through each file in the current directory

filename = 'output1.txt'
        
# Step 5: Read the file content
with open(filename, 'r') as file:  
    data_string = file.read()

    # Step 6: Extract coefficients from each line using regular expressions
coefficients = []
lines = data_string.strip().split('\n')

    # this method will only capture the absolute value of the coefficients, but it is fine.
for line in lines:
        # Extract the constant term (first number after the equal sign)
        #constant = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line).group()
        # Find all floating-point numbers before the 'x' character (coefficients)
    match = re.search(r'=(.*)', line)
    if match:
        rhs = match.group(1)
                # Extract coefficients from the right-hand side
        coeffs = re.findall(r"([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)(?=\s*[\*\s]*[ab])", line)
        coeffs = [float(c) for c in coeffs]
        #sort the coefficients
        coeffs=sorted(coeffs)
        coefficients.append(coeffs)  # Convert to float
        
        # Combine the constant term with the rest of the coefficients
    #all_coeffs =  [np.abs(float(c)) for c in coeffs]
    #coefficients.append(all_coeffs)

# convert the list of coefficients into a numpy array (matrix)
coeff_matrix = np.array(coefficients)
print("average coefficient error: ", np.mean(np.abs(coeff_matrix - real_coeff)))

"""
        #extract  number integer from node_num
num = int(re.search(r'\d+', node_num).group())

    # Step 8: Convert the list of coefficients into a numpy array (matrix)
coeff_matrix_fist_half_dim = np.array(coefficients[:num])

max_len = max(len(lst) for lst in coefficients[num:])

coeff_matrix_second_half_dim = np.array([lst + [np.nan] * (max_len - len(lst)) for lst in coefficients[num:]])


#coeff_matrix_second_half_dim = np.array(coefficients[num:])


 # Step 9: Calculate the accuracy of the coefficients
# Calculate the relative error for each coefficient and average them
accuracy_first_half_dim = np.abs(coeff_matrix_fist_half_dim) / 1.0 # ad-hoc solution
accuracy_second_half_dim = np.abs(coeff_matrix_second_half_dim) / 2.0 # ad-hoc solution



accuracy_first_half_dim = accuracy_first_half_dim.flatten()
accuracy_second_half_dim =  accuracy_second_half_dim.flatten()
accuracy_second_half_dim = accuracy_second_half_dim[~np.isnan(accuracy_second_half_dim)]

accuracies += list(accuracy_first_half_dim) + list(accuracy_second_half_dim)
    
    # Step 11: Append the accuracies for the current noise level to the main list
all_accuracies.append(accuracies)

"""

"""
node_number = ['20', '40', '60', '80', '100']



params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 600,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    'axes.titlesize': 14,
    'font.size': 12, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'Helvetica',
}
import matplotlib
matplotlib.rcParams.update(params)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette('Reds')



# Create a DataFrame from the accuracies to use with Seaborn
import pandas as pd

# Flatten the data and create labels for each noise level
flattened_accuracies = [accuracy for sublist in all_accuracies for accuracy in sublist]
node_labels = [level for level, accuracies in zip(node_number, all_accuracies) for _ in accuracies]

# Create a DataFrame to hold the data for Seaborn
data = pd.DataFrame({
    'Node Number': node_labels,
    'Performance Metric': flattened_accuracies
})


summary_stats = data.groupby('Node Number')['Performance Metric'].describe(percentiles=[0.25, 0.5, 0.75])

# This will print the summary statistics for each noise level including percentiles
print(summary_stats)

# You can access specific percentiles like this, for example:
for node_num in summary_stats.index:
    median = summary_stats.loc[node_num, '50%']
    q1 = summary_stats.loc[node_num, '25%']
    q3 = summary_stats.loc[node_num, '75%']
    print(f"Noise Level {node_num}: Q1 = {q1}, Median = {median}, Q3 = {q3}")


from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    return f'{100 * y:.0f}%'


# Plot the boxplot using Seaborn
plt.figure(figsize=(5, 4))
ax = sns.boxplot(x='Node Number', y='Performance Metric', data=data, showfliers=False, width=0.5)
#sns.swarmplot(x="Noise Level", y="Performance Metric", data=data, color=".25", size=4)
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

# Set title and labels
#plt.title('Accuracy at Different Noise Levels')
plt.xlabel('Number of Dimensions')
plt.ylabel('Absolute Coefficient Ratio')

plt.tight_layout()
plt.savefig('Dimensions_versus_accuracy.png', format='png', dpi=1200)
"""