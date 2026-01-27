# Data-driven discovery of high-dimensional dynamical systems with sparse interpretable neural networks

[![Python 3.10.13](https://img.shields.io/badge/python-3.10.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10-2.15](https://img.shields.io/badge/tensorflow-2.10--2.15-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

SREINet (Sparse Regression Embedded Interpretable Network) is an interpretable and end-to-end machine-learning framework that is designed for learning governing equations of high-dimensional dynamical systems. This framework addresses the fundamental challenge of the "curse of dimensionality" that constrains existing approaches like sparse optimization, symbolic regression, and Kolmogorov-Arnold networks.

### Main Contributions

- **Scalability**: Capable of accurately finding governing equations of high-dimensional systems of more than 100 dimensions with a personal laptop and potentially over 1000 dimensions with cloud computing
- **Extrapolation**: Generates correct coherent structures from untrained data
- **Robustness**: Maintains high accuracy against intermittent noise and incomplete data
- **Interpretability**: Direct recovery of governing equations through forward propagation
- **Efficiency**: Consistent computational cost as data volume increases

### Network Structure

SREINet is a neuro-symbolic framework designed to recover explicit governing equations of nonlinear dynamical systems. By replacing standard activation functions with a set of atomic basis functions and eliminating recursive composition, the network's forward pass essentially performs symbolic construction. It assembles multivariate library functions through a structured sequence of additions and multiplications, allowing the output to directly represent the system’s vector field. When combined with a specialized S-shaped periodic pruning strategy, SREINet can achieve exact recovery of the underlying governing equations.

### Figure 1: Schematic illustration of the structure of SREINet and the process of using it to discover governing equations.

![Figure 1](figs/Figure_11.png)

### Figure 2: Performance Comparison

![Figure 2](figs/Figure_2.png)

## 🚀 Installation

### Prerequisites

- Python 3.10.13
- TensorFlow 2.10.0 to 2.15.0 (inclusive)

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
   # Should output: TensorFlow version between 2.10.0 and 2.15.0
   ```

### Required Packages

The following packages are automatically installed via `requirements.txt`:

- **tensorflow>=2.10.0,<=2.15.0** - Deep learning framework (tested versions)
- **numpy>=1.19.0,<1.24.0** - Numerical computing
- **matplotlib>=3.5.0** - Plotting and visualization
- **scipy>=1.7.0** - Scientific computing
- **pandas>=1.3.0** - Data manipulation
- **scikit-learn>=1.0.0** - Machine learning utilities
- **openpyxl>=3.0.0** - Excel file handling
- **packaging>=21.0** - Version comparison utilities
- **seaborn>=0.11.0** - Data visualization
- **ipykernel>=6.0.0** - Jupyter kernel for Python
- **scikit-image>=0.24.0** - Image processing and computer vision
- **cmocean>=4.0.3** - Oceanographic colormaps for matplotlib
- **pysindy<=2.0.0** - PySINDy library for Ablation 

### 📚 Usage

### Data Download

For the experimental data

```bash
cd code/empirical_data
python download_data.py
```

**Data Source**: The experimental data is from the [MultiArm-Pendulum repository](https://github.com/dynamicslab/MultiArm-Pendulum) by Kaheman et al. (2022).

### Running Examples

All experiments are provided as Jupyter notebooks in the `code` directory (e.g. `code\examples` and `code\empirical_data`).

1. **Examples**:
   - `SREINet_Lorenz96.ipynb` - Lorenz 96 system (100D)
   - `SREINet_Kuramoto.ipynb` - Kuramoto model (60D)
   - `SREINet_Phi_4.ipynb` - Phi-4 system (100D)
   - `SREINet_DNLS.ipynb` - Discrete Nonlinear Schrödinger equation (100D)
   - `SREINet_AL.ipynb` - Ablowitz-Ladik system (128D)
   - `SREINet_FPU.ipynb` - Fermi-Pasta-Ulam chain
   - `SREINet_Phi_4_noise_effect.ipynb` - Phi-4 with noise
   - `hindmarsh_rose_network/` - Hindmarsh-Rose network notebooks (75D)
2. **Experimental notebooks** (in `code/empirical_data/`):
   - `SREINet_triple_pendulum*.ipynb`
   - one additional experiment with the double-pendulum system that is not presented in the paper.

## 📁 Project Structure

```
SREINet/
├── code/                                 # Main code directory
│   ├── utilities/                        # Core implementation
│   │   ├── SREINet.py                     # Main network implementation
│   │   ├── DataGenerator.py               # Data generation utilities
│   │   ├── Model_zoo.py                   # Predefined dynamical systems
│   │   ├── loss.py                        # Loss functions
│   │   ├── pruning_scheduler.py           # Pruning utilities
│   │   ├── sreinet_trainer.py             # Training helpers
│   │   └── ...                            # Other utilities
│   ├── examples/                         # Jupyter notebook examples
│   │   ├── SREINet_Lorenz96.ipynb
│   │   ├── SREINet_Kuramoto.ipynb
│   │   ├── SREINet_Phi_4.ipynb
│   │   ├── SREINet_DNLS.ipynb
│   │   ├── SREINet_AL.ipynb
│   │   ├── SREINet_FPU.ipynb
│   │   ├── SREINet_Phi_4_noise_effect.ipynb
│   │   └── hindmarsh_rose_network/
│   ├── empirical_data/                    # Experimental data + notebooks
│   │   ├── download_data.py
│   │   ├── TriplePendulum_Data/
│   │   ├── DoublePendulum_Data/
│   │   └── SREINet_triple_pendulum_*.ipynb
│   ├── ablation/                         # Ablation studies
│   ├── missing_library_functions/        # Missing-library experiments
│   ├── SREINet-extension (nonseparable fns) # Nonseparable extensions
│   ├── balancing weights/                # Weight balancing experiments
│   └── Archived/                          # Archived experiments
├── results/                               # Training results and outputs
│   ├── AL_128D/
│   ├── DNLS_100D/
│   ├── Kuramoto_60D/
│   ├── Lorenz_96_100D/
│   ├── Phi4/
│   ├── Hindmarsh_rose_network-75D/
│   ├── training_dynamics/
│   ├── training_outputs/
│   └── ...
├── data/                                  # Source data for figures
├── figs/                                  # Generated figures
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
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
