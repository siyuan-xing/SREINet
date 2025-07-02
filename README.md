# Data-driven discovery of spatiotemporal dynamical systems with sparse interpretable neural networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10.0-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

SREINet (Sparse Regression Embedded Interpretable Network) is a machine-learning framework that is specifically designed for learning governing equations of spatiotemporal dynamical systems. This framework addresses the fundamental challenge of the "curse of dimensionality" that plagues existing approaches like sparse optimization, symbolic regression, and Kolmogorov-Arnold networks.

### Key Innovation

SREINet integrates an interpretable neural network incorporating the matrix formulation of sparse regression with a specially designed sparsity-promoting pruning scheme. The framework reduces computational complexity of matrix-formulation from $O[m(n+p)!/(n!p!)]$ to $O(m_bpn²)$, where $m$ is the number of data points, $m_b$ is the mini-batch size, $p$ is the order of nonlinearity, and $n$ is the system dimension.

### Main Contributions

- **Scalability**: Capable of accurately finding governing equations of spatiotemporal systems of more than 100 dimensions with a personal laptop and potentially over 1000 dimensions with cloud computing
- **Extrapolation**: Generates correct coherent structures from untrained data
- **Robustness**: Maintains high accuracy against intermittent noise and incomplete data
- **Interpretability**: Direct recovery of governing equations through forward propagation
- **Efficiency**: Consistent computational cost as data volume increases

## 🏗️ Network Architecture


SREINet employs an interpretable neural network architecture that combines:

1. **Matrix Formulation of Sparse Regression**: Incorporates sparse regression principles directly into the network structure
2. **Sparsity-Promoting Pruning**: Periodic, piecewise-continuous plateau schedule for computational efficiency
3. **Interpretable Structure**: Direct mapping from network weights to governing equations
4. **Dimensionality Reduction**: Reduces space complexity from O[m(n+p)!/(n!p!)] to O(mpn²)

### Network Structure

### Figure 1: Schematic illustration of the structure of SREINet and the process of using it to discover governing equations.

![Figure 1](figs/Figure%209.png)

The neural network architecture consists of an input layer, multiple hidden layers, and an output layer. The input is the state variables $\boldsymbol{x}=[x_1(t),x_2(t),\dots,x_n(t)]^T$, and the output is the time derivative $\dot{x}_i$ for each dimension.

**Activation Functions**: Each layer uses univariate candidate functions:
$$\boldsymbol{\Phi}^{[j]} = \left[\varphi_1^{[j]},\varphi_2^{[j]},\dots,\varphi_{k}^{[j]}\right]^T$$

where $\varphi^{[j]}_i(\zeta) \in \{1,\zeta, \sin(\zeta), \sin(2\zeta),\dots, \cos(\zeta), \cos(2\zeta),\dots\}$

**Hidden Layer Output**: The $j$th hidden layer output is:
$$\boldsymbol{y}^{[j]}=\boldsymbol{\Phi}^{[j]}(\boldsymbol{x}) \odot \boldsymbol{W}^{[j-1]} \boldsymbol{\cdot} \boldsymbol{y}^{[j-1]}$$

where $\odot$ denotes elementwise Hadamard product and $\boldsymbol{\cdot}$ is matrix multiplication.

**Network Output**: The final output is:
$$\hat{\dot{x}}_i=\mathcal{N}(\boldsymbol{x})=\boldsymbol{1}_{1\times k} \boldsymbol{\cdot} \boldsymbol{\Phi}^{[p]}(\boldsymbol{x}) \odot \boldsymbol{W}^{[p-1]} \cdots \boldsymbol{W}^{[2]} \boldsymbol{\cdot} \boldsymbol{\Phi}^{[2]}(\boldsymbol{x}) \odot \boldsymbol{W}^{[1]} \boldsymbol{\cdot} \boldsymbol{\Phi}^{[1]}(\boldsymbol{x})$$

where $p$ is the network depth (highest order of nonlinearity) and $\boldsymbol{1}_{1\times k}$ represents fixed unit weights of the output layer.

**Key Features**:
- No composition of activation functions
- Explicit form can be written as sum of all combinations up to $p$th order
- Direct interpretability through forward propagation


### Figure 2: Performance Comparison
![Figure 2](figs/Figure%202.png)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.10.0 (specific version required)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/siyuan-xing/SREINet.git
   cd SREINet
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```python
   import tensorflow as tf
   print(f"TensorFlow version: {tf.__version__}")
   # Should output: TensorFlow version: 2.10.0
   ```

### Required Packages

The following packages are automatically installed via `requirements.txt`:

- **tensorflow==2.10.0** - Deep learning framework (specific version required)
- **numpy>=1.19.0** - Numerical computing
- **matplotlib>=3.3.0** - Plotting and visualization
- **scipy>=1.7.0** - Scientific computing
- **pandas>=1.3.0** - Data manipulation
- **scikit-learn>=1.0.0** - Machine learning utilities
- **openpyxl>=3.0.0** - Excel file handling
- **packaging>=20.0** - Version comparison utilities

### Manual Installation (if needed)

If you prefer to install packages manually:

```bash
pip install tensorflow==2.10.0
pip install numpy matplotlib scipy pandas scikit-learn
pip install openpyxl packaging
```

## 📚 Usage

### Data Download

For the triple pendulum experimental data:

```bash
cd code/experiment_data
python download_data.py
```

This will download ~65MB of experimental data to `code/experiment_data/TriplePendulum_Data/`.

### Running Examples

All examples are provided as Jupyter notebooks in the `examples/` directory. 

1. **Examples**:
   - `SREINet_Lorenz96.ipynb` - Lorenz 96 system (100D)
   - `SREINet_Kuramoto.ipynb` - Kuramoto model (60D)
   - `SREINet_Phi_4.ipynb` - Phi-4 discrete field theory
   - `SREINet_DNLS.ipynb` - Discrete Nonlinear Schrödinger equation
   - `SREINet_AL.ipynb` - Ablowitz-Ladik system
   - `SREINet_FPU.ipynb` - Fermi-Pasta-Ulam chain
   - `SREINet_triple_pendulum.ipynb` - Experimental triple pendulum data

## 📁 Project Structure

```
SREINet/
├── utilities/                 # Core implementation
│   ├── SREINet.py            # Main network implementation
│   ├── DataGenerator.py      # Data generation utilities
│   ├── Model_zoo.py          # Predefined dynamical systems
│   ├── loss.py               # Loss functions
│   └── pruning_schedule.py   # Pruning utilities
├── examples/                  # Jupyter notebook examples
│   ├── SREINet_Lorenz96.ipynb
│   ├── SREINet_Kuramoto.ipynb
│   ├── SREINet_Phi_4.ipynb
│   └── ...
├── results/                   # Training results and outputs
│   ├── AL-128D/
│   ├── DNLS-100D/
│   ├── Kuramoto-60D/
│   └── ...
├── data/                      # Experimental data
└── requirements.txt           # Python dependencies
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Siyuan (Simon) Xing**, California Polytechnic State University - San Luis Obispo
- **Qingyu Han**, California Polytechnic State University - San Luis Obispo
- **Efstathios G. Charalampidis**, San Diego State University
- **Ying-Cheng Lai**, Arizona State University

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{sreinet2025,
  title={Data-driven discovery of spatiotemporal dynamical systems with sparse interpretable neural networks},
  author={Xing, Siyuan and Han, Qingyu and Charalampidis, Efstathios G. and Lai, Ying-Cheng},
  journal={Nature Communications},
  year={2025},
  doi={TBD}
}
```