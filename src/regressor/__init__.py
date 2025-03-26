"""A simple package for simulating and fitting regression data."""

__version__ = "0.1.0"

from regressor.simulate import (
    simulate_linear,
    simulate_polynomial,
    simulate_exponential,
)
from regressor.fit import (
    fit_linear,
    fit_polynomial,
    fit_exponential,
)

__all__ = [
    "simulate_linear",
    "simulate_polynomial",
    "simulate_exponential",
    "fit_linear",
    "fit_polynomial",
    "fit_exponential",
]
