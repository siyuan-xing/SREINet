# Lorenz and Lotka–Volterra Network Examples

This folder mirrors the structure of `rossler_network_with_high_order_interaction`,
but focuses on (primarily) pairwise-coupled dynamical systems over homogeneous Erdős–Rényi
networks, while exposing optional simplicial interactions when available.

## Contents

- `network_generator.py`: helper utilities to sample connected Erdős–Rényi graphs
  and visualise their structure.
- `coupling.py`: reusable diffusive coupling terms for pairwise and triangle-based interactions.
- `lorenz_network.py`: Lorenz oscillator dynamics with diffusive coupling on the
  x-coordinate.
- `lotka_volterra_network.py`: Predator–prey dynamics with diffusive coupling plus
  optional three-body (triangle) interactions.
- `lorenz_hoi_network.py`: Lorenz dynamics including three-body interactions.
- `higher_order_network.py`: helper utilities to sample simplicial structures.
- `simulations/lorenz_simulation.py`, `simulations/lotka_volterra_simulation.py`:
  dedicated simulation workflows for each dynamical system.
- `lorenz_hoi_simulation.py`: Lorenz dynamics with pairwise plus three-body
  interactions, alongside `run_lorenz_hoi.py` for CLI usage.
- `simulations/run_lorenz.py`, `simulations/run_lotka_volterra.py`: command-line
  entry points so you can run simulations directly with `python -m ...`.
- `run_lorenz_hoi.py`: command-line interface for the higher-order Lorenz model.
- `SREINet_Lorenz_network.ipynb`, `SREINet_LotkaVolterra_network.ipynb`: notebooks
  that generate data with the helper functions for SREINet training or analysis.
- `simulations/Lorenz_network_simulation.ipynb`, `simulations/LotkaVolterra_network_simulation.ipynb`:
  standalone notebooks for exploring the raw network dynamics with editable
  coupling strengths and graph structures.
- `simulations/Lorenz_HOI_network_simulation.ipynb`: notebook showcasing higher-order Lorenz dynamics.

## Quick start

```python
from lorenz_lotka_network import (
    simulate_lorenz_network,
    simulate_lorenz_higher_order_network,
)

result = simulate_lorenz_network(visualize=True)
hoi_result = simulate_lorenz_higher_order_network(visualize=True)
```

Or run directly from the command line:

```bash
python -m lorenz_lotka_network.simulations.run_lorenz --visualize
python -m lorenz_lotka_network.simulations.run_lotka_volterra --visualize
python -m lorenz_lotka_network.run_lorenz_hoi --visualize
```

Each `SimulationResult` exposes the time grid (`time`), state trajectories
(`state`), derivatives (`derivatives`), and the network structure (`edge_list`,
`adjacency`, `degrees`).
