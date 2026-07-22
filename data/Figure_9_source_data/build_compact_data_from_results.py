"""Generate the compact state arrays plotted in manuscript Figure 9."""

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load_generator(source: Path):
    """Load a result directory's DataGenerator without cross-directory collisions."""
    for name in ("DataGenerator", "Model_zoo"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(source))
    try:
        module = import_module("DataGenerator")
    finally:
        sys.path.pop(0)
    for name in ("DataGenerator", "Model_zoo"):
        sys.modules.pop(name, None)
    return module.DataGenerator


def load_function(path: Path, function_name: str):
    spec = spec_from_file_location(f"figure9_{path.parent.name}_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def save(name: str, time: np.ndarray, truth: np.ndarray, prediction: np.ndarray, **metadata) -> None:
    output = HERE / f"{name}_plot_data.npz"
    payload = {
        "time": np.asarray(time).ravel(),
        "truth": np.asarray(truth),
        "prediction": np.asarray(prediction),
    }
    payload.update({key: np.asarray(value) for key, value in metadata.items()})
    np.savez_compressed(output, **payload)
    print(f"Wrote {output.relative_to(REPO)} ({output.stat().st_size / 1024**2:.1f} MiB)")


def phi4_kink() -> None:
    source = REPO / "results/Phi4"
    generator = load_generator(source)
    node_num, coupling, end_time, dt = 50, 2.0, 500.0, 0.05
    positions = np.linspace(-node_num / 2 / np.sqrt(coupling), node_num / 2 / np.sqrt(coupling), node_num)
    displacement = np.tanh(positions)
    velocity = 1 - np.tanh(positions) ** 2
    initial = np.concatenate((displacement, velocity))
    time, truth, _, _ = generator(initial, T=end_time, dt=dt).generate_dataset_by_model_name(
        "discrete_phi_quartic", coupling, node_num
    )
    recovered_ode = load_function(source / "phi_ode.py", "phi4_ode")
    t_eval = np.arange(0, end_time, dt)
    solution = solve_ivp(recovered_ode, (0, end_time), truth[0], t_eval=t_eval)
    if not solution.success:
        raise RuntimeError(solution.message)
    save("phi4_kink", time, truth[:, :node_num], solution.y[:node_num].T, model="discrete phi4 kink")


def dnls_gaussian() -> None:
    source = REPO / "results/DNLS_100D"
    generator = load_generator(source)
    node_num, coupling, end_time, dt = 50, 2.0, 180.0, 0.01
    positions = np.linspace(-node_num / 2 / np.sqrt(coupling), node_num / 2 / np.sqrt(coupling), node_num)
    initial_complex = np.exp(-(positions**2) / (2 * 2.0**2)).astype(complex)
    time, truth_complex, _, _ = generator(initial_complex, T=end_time, dt=dt).generate_dataset_by_model_name(
        "dnls", coupling, node_num
    )
    recovered_ode = load_function(source / "DNLS_ode.py", "DNLS_ode")
    initial = np.concatenate((initial_complex.real, initial_complex.imag))
    t_eval = np.arange(0, end_time, dt)
    solution = solve_ivp(recovered_ode, (0, end_time), initial, t_eval=t_eval)
    if not solution.success:
        raise RuntimeError(solution.message)
    prediction = np.abs(solution.y[:node_num].T + 1j * solution.y[node_num:].T)
    save("dnls_gaussian", time, np.abs(truth_complex), prediction, model="DNLS Gaussian")


def abl_case(name: str, initial_complex: np.ndarray, start_time: float, duration: float) -> None:
    source = REPO / "results/AL_128D"
    generator = load_generator(source)
    node_num, dt = 64, 0.01
    time, truth_complex, _, _ = generator(initial_complex, T=duration, dt=dt).generate_dataset_by_model_name(
        "abolowitz_ladik", node_num, method="BDF"
    )
    time = time.ravel() + start_time
    recovered_ode = load_function(source / "AL_ode.py", "AL_ode")
    initial = np.concatenate((initial_complex.real, initial_complex.imag))
    end_time = start_time + duration
    t_eval = np.arange(start_time, end_time, dt)
    solution = solve_ivp(recovered_ode, (start_time, end_time), initial, t_eval=t_eval, method="BDF")
    if not solution.success:
        raise RuntimeError(solution.message)
    prediction = np.abs(solution.y[:node_num].T + 1j * solution.y[node_num:].T)
    save(name, time, np.abs(truth_complex), prediction, model=name.replace("_", " "))


def abl_peregrine() -> None:
    node_num = 64
    indices = np.arange(-node_num / 2, node_num / 2)
    start = -4.0
    initial = 1 / np.sqrt(2) * (1 - 6 * (1 + 1j * start) / (1 + 2 * indices**2 + 1.5 * start**2))
    abl_case("abl_peregrine", initial, start, 8.0)


def abl_breather() -> None:
    node_num = 64
    indices = np.arange(-node_num / 2, node_num / 2)
    start, omega = -5.0, 4.0
    theta = -np.arcsinh(omega)
    radius = np.arccosh((2 + np.cosh(theta)) / 3)
    factor = -omega / np.sqrt(3) / np.sinh(radius)
    initial = 1 / np.sqrt(2) * (
        np.cos(omega * start + 1j * theta) + factor * np.cosh(radius * indices)
    ) / (np.cos(omega * start) + factor * np.cosh(radius * indices))
    abl_case("abl_breather", initial, start, 10.0)


if __name__ == "__main__":
    phi4_kink()
    dnls_gaussian()
    abl_peregrine()
    abl_breather()
