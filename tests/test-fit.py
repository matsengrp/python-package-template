"""Tests for the fitting functions."""

import numpy as np
import pytest
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


def test_fit_linear_perfect():
    """Test fitting a perfect linear relationship."""
    # Create perfectly linear data (no noise)
    x = np.linspace(0, 10, 100)
    slope = 2.0
    intercept = -1.0
    y = slope * x + intercept
    
    # Fit model
    result = fit_linear(x, y)
    
    # Check parameters
    np.testing.assert_allclose(result['params'][0], intercept, rtol=1e-10)
    np.testing.assert_allclose(result['params'][1], slope, rtol=1e-10)
    
    # Check goodness of fit
    assert result['r_squared'] > 0.999  # R² should be very close to 1
    assert result['rmse'] < 1e-10  # RMSE should be very close to 0


def test_fit_linear_with_noise():
    """Test fitting linear data with noise."""
    # Generate data with known parameters and some noise
    slope = 3.5
    intercept = 2.0
    x, y = simulate_linear(
        n_samples=1000,
        slope=slope,
        intercept=intercept,
        noise_level=1.0,
        random_state=42
    )
    
    # Fit model
    result = fit_linear(x, y)
    
    # Parameters should be close but not exact due to noise
    np.testing.assert_allclose(result['params'][0], intercept, rtol=0.1)
    np.testing.assert_allclose(result['params'][1], slope, rtol=0.1)
    
    # R² should be high but not perfect
    assert result['r_squared'] > 0.9


def test_fit_polynomial_perfect():
    """Test fitting a perfect polynomial relationship."""
    # Create perfectly polynomial data (no noise)
    x = np.linspace(0, 5, 100)
    coefs = [1.0, -2.0, 0.5]  # 1 - 2x + 0.5x²
    y = coefs[0] + coefs[1] * x + coefs[2] * x**2
    
    # Fit model
    result = fit_polynomial(x, y, degree=2)
    
    # Check parameters
    np.testing.assert_allclose(result['params'], coefs, rtol=1e-10)
    
    # Check goodness of fit
    assert result['r_squared'] > 0.999  # R² should be very close to 1
    assert result['rmse'] < 1e-10  # RMSE should be very close to 0


def test_fit_polynomial_with_noise():
    """Test fitting polynomial data with noise."""
    # Generate data with known parameters and some noise
    coefs = [3.0, 1.0, -0.5]  # 3 + x - 0.5x²
    x, y = simulate_polynomial(
        n_samples=1000,
        coefficients=coefs,
        noise_level=2.0,
        random_state=42
    )
    
    # Fit model
    result = fit_polynomial(x, y, degree=2)
    
    # Parameters should be close but not exact due to noise
    np.testing.assert_allclose(result['params'], coefs, rtol=0.2)
    
    # R² should be high but not perfect
    assert result['r_squared'] > 0.8


def test_fit_exponential_perfect():
    """Test fitting a perfect exponential relationship."""
    # Create perfectly exponential data (no noise)
    x = np.linspace(0, 3, 100)
    amplitude = 2.0
    decay = 0.5
    y = amplitude * np.exp(decay * x)
    
    # Fit model
    result = fit_exponential(x, y)
    
    # Check parameters
    np.testing.assert_allclose(result['params'][0], amplitude, rtol=1e-3)
    np.testing.assert_allclose(result['params'][1], decay, rtol=1e-3)
    
    # Check goodness of fit
    assert result['r_squared'] > 0.999  # R² should be very close to 1
    assert result['rmse'] < 1e-3  # RMSE should be very close to 0


def test_fit_exponential_with_noise():
    """Test fitting exponential data with noise."""
    # Generate data with known parameters and some noise
    amplitude = 1.5
    decay = 0.3
    x, y = simulate_exponential(
        n_samples=1000,
        amplitude=amplitude,
        decay=decay,
        noise_level=0.2,
        random_state=42
    )
    
    # Fit model
    result = fit_exponential(x, y)
    
    # Parameters should be close but not exact due to noise
    np.testing.assert_allclose(result['params'][0], amplitude, rtol=0.2)
    np.testing.assert_allclose(result['params'][1], decay, rtol=0.2)
    
    # R² should be high but not perfect
    assert result['r_squared'] > 0.8