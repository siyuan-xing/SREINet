import re

import numpy as np
import scipy.io


def parse_equation_coeffs(line: str) -> dict:
    rhs = line.split("=", 1)[1].strip()
    parts = re.split(r"\s(?=[+-])", rhs)
    coeffs = {}
    for part in parts:
        term = part.strip().replace(" ", "")
        if not term:
            continue
        if term.startswith("+"):
            term = term[1:]
        if "x" not in term:
            coeffs["const"] = coeffs.get("const", 0.0) + float(term)
            continue
        if "*" in term:
            coeff_str, var_str = term.split("*", 1)
            coeff = float(coeff_str)
        else:
            if term.startswith("-"):
                coeff = -1.0
                var_str = term[1:]
            else:
                coeff = 1.0
                var_str = term
        match = re.match(r"(x\d+)(?:\^(\d+))?$", var_str)
        if not match:
            raise ValueError(f"Unexpected term format: {term}")
        var = match.group(1)
        power = int(match.group(2)) if match.group(2) else 1
        key = f"{var}^{power}" if power != 1 else var
        coeffs[key] = coeffs.get(key, 0.0) + coeff
    return coeffs


def load_true_coeffs(edge_list: np.ndarray) -> list:
    n_nodes = int(edge_list.max()) + 1
    neighbors = [[] for _ in range(n_nodes)]
    degrees = np.zeros(n_nodes, dtype=int)
    for i, j in edge_list:
        neighbors[i].append(j)
        neighbors[j].append(i)
        degrees[i] += 1
        degrees[j] += 1

    coupling_strength = 0.05
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    r = 0.05
    s = 4.0
    x0 = -1.618
    I = 3.2

    coeffs_by_dim = []
    for dim_index in range(3 * n_nodes):
        coeffs = {}
        if dim_index < n_nodes:
            i = dim_index
            coeffs["const"] = I
            coeffs[f"x{i+1}^2"] = b
            coeffs[f"x{i+1}^3"] = -a
            coeffs[f"x{n_nodes + i + 1}"] = 1.0
            coeffs[f"x{2 * n_nodes + i + 1}"] = -1.0
            coeffs[f"x{i+1}"] = coeffs.get(f"x{i+1}", 0.0) - coupling_strength * degrees[i]
            for j in neighbors[i]:
                key = f"x{j+1}"
                coeffs[key] = coeffs.get(key, 0.0) + coupling_strength
        elif dim_index < 2 * n_nodes:
            i = dim_index - n_nodes
            coeffs["const"] = c
            coeffs[f"x{i+1}^2"] = -d
            coeffs[f"x{n_nodes + i + 1}"] = -1.0
        else:
            i = dim_index - 2 * n_nodes
            coeffs["const"] = -r * s * x0
            coeffs[f"x{i+1}"] = r * s
            coeffs[f"x{2 * n_nodes + i + 1}"] = -r
        coeffs_by_dim.append(coeffs)
    return coeffs_by_dim


def main() -> None:
    with open("output1.txt", "r") as file:
        lines = [line for line in file.read().strip().split("\n") if line.strip()]

    pred_coeffs = [parse_equation_coeffs(line) for line in lines]

    mat = scipy.io.loadmat("data.mat")
    edge_list = mat["edge_list"]
    true_coeffs = load_true_coeffs(edge_list)

    if len(pred_coeffs) != len(true_coeffs):
        raise ValueError(
            f"Equation count mismatch: predicted={len(pred_coeffs)}, true={len(true_coeffs)}"
        )

    all_errors = []
    for pred, true in zip(pred_coeffs, true_coeffs):
        keys = set(pred.keys()) | set(true.keys())
        all_errors.extend([abs(pred.get(k, 0.0) - true.get(k, 0.0)) for k in keys])

    print("average coefficient error:", np.mean(all_errors))


if __name__ == "__main__":
    main()

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
