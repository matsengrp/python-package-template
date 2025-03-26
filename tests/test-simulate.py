"""Tests for the simulation functions."""

import numpy as np
import pytest
from regressor.simulate import (
    simulate_linear,
    simulate_polynomial,
    simulate_exponential,
)


def test_simulate_linear_shape():
    """Test that simulate_linear returns arrays of the correct shape."""
    n_samples = 100
    x, y = simulate_linear(n_samples=n_samples)
    assert x.shape == (n_samples,)
    assert y.shape == (n_samples,)


def test_simulate_linear_reproducibility():
    """Test that simulate_linear with a fixed random_state is reproducible."""
    x1, y1 = simulate_linear(random_state=42)
    x2, y2 = simulate_linear(random_state=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_simulate_linear_parameters():
    """Test that simulate_linear respects the provided parameters."""
    slope = 3.0
    intercept = -1.0
    x_range = (5, 10)
    
    x, y = simulate_linear(
        slope=slope,
        intercept=intercept,
        noise_level=0,  # No noise for easy testing
        x_range=x_range,
        random_state=42
    )
    
    # Check x range
    assert x.min() >= x_range[0]
    assert x.max() <= x_range[1]
    
    # Check that y = slope * x + intercept (approximately, due to numerical precision)
    expected_y = slope * x + intercept
    np.testing.assert_allclose(y, expected_y)


def test_simulate_polynomial_shape():
    """Test that simulate_polynomial returns arrays of the correct shape."""
    n_samples = 150
    x, y = simulate_polynomial(n_samples=n_samples)
    assert x.shape == (n_samples,)
    assert y.shape == (n_samples,)


def test_simulate_polynomial_reproducibility():
    """Test that simulate_polynomial with a fixed random_state is reproducible."""
    x1, y1 = simulate_polynomial(random_state=42)
    x2, y2 = simulate_polynomial(random_state=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_simulate_polynomial_parameters():
    """Test that simulate_polynomial respects the provided parameters."""
    coefficients = [1.0, 2.0, 3.0]  # 1 + 2x + 3x²
    x_range = (0, 5)
    
    x, y = simulate_polynomial(
        coefficients=coefficients,
        noise_level=0,  # No noise for easy testing
        x_range=x_range,
        random_state=42
    )
    
    # Check x range
    assert x.min() >= x_range[0]
    assert x.max() <= x_range[1]
    
    # Check that y follows the polynomial (approximately, due to numerical precision)
    expected_y = coefficients[0] + coefficients[1] * x + coefficients[2] * x**2
    np.testing.assert_allclose(y, expected_y)


def test_simulate_exponential_shape():
    """Test that simulate_exponential returns arrays of the correct shape."""
    n_samples = 120
    x, y = simulate_exponential(n_samples=n_samples)
    assert x.shape == (n_samples,)
    assert y.shape == (n_samples,)


def test_simulate_exponential_reproducibility():
    """Test that simulate_exponential with a fixed random_state is reproducible."""
    x1, y1 = simulate_exponential(random_state=42)
    x2, y2 = simulate_exponential(random_state=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_simulate_exponential_parameters():
    """Test that simulate_exponential respects the provided parameters."""
    amplitude = 2.0
    decay = 0.5
    x_range = (0, 3)
    
    x, y = simulate_exponential(
        amplitude=amplitude,
        decay=decay,
        noise_level=0,  # No noise for easy testing
        x_range=x_range,
        random_state=42
    )
    
    # Check x range
    assert x.min() >= x_range[0]
    assert x.max() <= x_range[1]
    
    # Check that y = amplitude * exp(decay * x) (approximately, due to numerical precision)
    expected_y = amplitude * np.exp(decay * x)
    np.testing.assert_allclose(y, expected_y)
