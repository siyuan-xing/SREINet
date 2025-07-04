# Data-driven discovery of spatiotemporal dynamical systems with sparse interpretable neural networks

[![Python 3.10.13](https://img.shields.io/badge/python-3.10.13-blue.svg)](https://www.python.org/downloads/)
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

**Key Features**:
- No composition of activation functions
- Explicit form can be written as sum of all combinations up to $p$th order
- Direct interpretability through forward propagation


### Figure 2: Performance Comparison
![Figure 2](figs/Figure%202.png)

## 🚀 Installation

### Prerequisites

- Python 3.10.13
- TensorFlow 2.10.0 (specific version required)

### Setup

#### Using Conda 

1. **Create and activate conda environment**:
   ```bash
   conda create -n sreinet python=3.10.13
   conda activate sreinet
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/siyuan-xing/SREINet.git
   cd SREINet
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```python
   import tensorflow as tf
   print(f"TensorFlow version: {tf.__version__}")
   # Should output: TensorFlow version: 2.10.0
   ```


### Required Packages

The following packages are automatically installed via `requirements.txt`:

- **tensorflow==2.10.0** - Deep learning framework (specific version required)
- **numpy>=1.19.0<1.24.0** - Numerical computing
- **matplotlib>=3.5.0** - Plotting and visualization
- **scipy>=1.7.0** - Scientific computing
- **pandas>=1.3.0** - Data manipulation
- **scikit-learn>=1.0.0** - Machine learning utilities
- **openpyxl>=3.0.0** - Excel file handling
- **packaging>=21.0** - Version comparison utilities
- **seaborn>=0.11.0** -  Data visualization
- **ipykernel>=6.0.0** - Jupyter kernel for Python 


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

<!---
## 📚 Citation

If you use this code in your research, please cite:


```bibtex
@article{sreinet2025,
  title={Data-driven discovery of spatiotemporal dynamical systems with sparse interpretable neural networks},
  author={Xing, Siyuan and Han, Qingyu and Charalampidis, Efstathios G. and Lai, Ying-Cheng},
  journal={Arxiv},
  year={2025},
  doi={TBD}
}
```
-->