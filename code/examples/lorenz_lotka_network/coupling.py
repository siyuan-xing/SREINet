"""Helper functions for network coupling terms."""

from __future__ import annotations

import numpy as np


def diffusive_coupling(values: np.ndarray, edge_list: np.ndarray) -> np.ndarray:
    """
    Compute diffusive coupling for a scalar variable on each node.

    The returned vector corresponds to (A - D) @ values, which is the
    action of the graph Laplacian on the state.
    """

    coupling = np.zeros_like(values)
    for i, j in edge_list:
        diff = values[j] - values[i]
        coupling[i] += diff
        coupling[j] -= diff
    return coupling

