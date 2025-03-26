"""Functions for simulating regression data."""

import numpy as np
from typing import Tuple, Optional, Union, List


def simulate_linear(
    n_samples: int = 100,
    slope: float = 2.0,
    intercept: float = 1.0,
    noise_level: float = 1.0,
    x_range: Tuple[float, float] = (0, 10),
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate data from a linear model: y = slope * x + intercept + noise.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    slope : float, default=2.0
        Slope of the linear relationship.
    intercept : float, default=1.0
        Intercept of the linear relationship.
    noise_level : float, default=1.0
        Standard deviation of the Gaussian noise.
    x_range : tuple, default=(0, 10)
        Range of the x values.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    x : ndarray of shape (n_samples,)
        Input data.
    y : ndarray of shape (n_samples,)
        Output data, with noise.
    """
    if random_state is not None:
        np.random.seed(random_state)

    x = np.random.uniform(x_range[0], x_range[1], size=n_samples)
    noise = np.random.normal(0, noise_level, size=n_samples)
    y = slope * x + intercept + noise

    return x, y


def simulate_polynomial(
    n_samples: int = 100,
    coefficients: List[float] = [1.0, 2.0, 1.0],
    noise_level: float = 1.0,
    x_range: Tuple[float, float] = (0, 10),
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate data from a polynomial model: y = sum(coef[i] * x^i) + noise.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    coefficients : list, default=[1.0, 2.0, 1.0]
        Coefficients of the polynomial, starting from the constant term.
        For example, [1, 2, 3] corresponds to 1 + 2x + 3x^2.
    noise_level : float, default=1.0
        Standard deviation of the Gaussian noise.
    x_range : tuple, default=(0, 10)
        Range of the x values.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    x : ndarray of shape (n_samples,)
        Input data.
    y : ndarray of shape (n_samples,)
        Output data, with noise.
    """
    if random_state is not None:
        np.random.seed(random_state)

    x = np.random.uniform(x_range[0], x_range[1], size=n_samples)
    y = np.zeros_like(x)

    for i, coef in enumerate(coefficients):
        y += coef * x**i

    noise = np.random.normal(0, noise_level, size=n_samples)
    y += noise

    return x, y


def simulate_exponential(
    n_samples: int = 100,
    amplitude: float = 1.0,
    decay: float = 0.5,
    noise_level: float = 0.1,
    x_range: Tuple[float, float] = (0, 10),
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate data from an exponential model: y = amplitude * exp(decay * x) + noise.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples to generate.
    amplitude : float, default=1.0
        Amplitude of the exponential.
    decay : float, default=0.5
        Decay rate of the exponential.
    noise_level : float, default=0.1
        Standard deviation of the Gaussian noise.
    x_range : tuple, default=(0, 10)
        Range of the x values.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    x : ndarray of shape (n_samples,)
        Input data.
    y : ndarray of shape (n_samples,)
        Output data, with noise.
    """
    if random_state is not None:
        np.random.seed(random_state)

    x = np.random.uniform(x_range[0], x_range[1], size=n_samples)
    noise = np.random.normal(0, noise_level, size=n_samples)
    y = amplitude * np.exp(decay * x) + noise

    return x, y
