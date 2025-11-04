# Hindmarsh–Rose Network Example

This folder provides utilities to simulate networks of coupled Hindmarsh–Rose
neurons on random Erdős–Rényi graphs.

## Contents

- `hindmarsh_rose_model.py`: single-neuron dynamics with cubic nonlinearities.
- `coupling.py`: diffusive coupling helper applied to the membrane potential.
- `network_generator.py`: sampling and adjacency helpers for ER graphs.
- `simulation.py`: wrappers around `solve_ivp` and quick-look plots.
- `run_hindmarsh_rose.py`: command-line interface for batch simulations.
- `HindmarshRose_network_simulation.ipynb`: interactive simulation notebook.
- `SREINet_HindmarshRose_network.ipynb`: system identification workflow using SREINet.

## Quick start

```python
from hindmarsh_rose_network import simulate_hindmarsh_rose_network

result = simulate_hindmarsh_rose_network(visualize=True)
```

Or run from the CLI:

```bash
python -m hindmarsh_rose_network.run_hindmarsh_rose --visualize
```
