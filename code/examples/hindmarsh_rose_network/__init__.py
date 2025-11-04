"""Hindmarsh–Rose network simulation package."""

from .simulation import (
    SimulationResult,
    simulate_hindmarsh_rose_network,
    plot_overview as plot_hindmarsh_rose_overview,
)

__all__ = [
    "SimulationResult",
    "simulate_hindmarsh_rose_network",
    "plot_hindmarsh_rose_overview",
]

