"""Functions for fitting regression models."""

import numpy as np
from scipy import optimize
from typing import Tuple, Dict, Callable, Any, Optional, Union


def fit_linear(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Fit a linear regression model: y = slope * x + intercept.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input data.
    y : ndarray of shape (n_samples,)
        Output data.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params': array of shape (2,) with [intercept, slope]
        - 'r_squared': coefficient of determination (R^2)
        - 'rmse': root mean squared error
    """
    # Add constant term to x for intercept
    X = np.vstack([np.ones_like(x), x]).T
    
    # Solve the linear system using least squares
    params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    # Extract parameters
    intercept, slope = params
    
    # Calculate R^2 and RMSE
    y_pred = intercept + slope * x
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    return {
        'params': np.array([intercept, slope]),
        'r_squared': r_squared,
        'rmse': rmse,
    }


def fit_polynomial(
    x: np.ndarray, y: np.ndarray, degree: int = 2
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Fit a polynomial regression model: y = sum(coef[i] * x^i).

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input data.
    y : ndarray of shape (n_samples,)
        Output data.
    degree : int, default=2
        Degree of the polynomial.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params': array of shape (degree+1,) with polynomial coefficients
                   starting with the constant term
        - 'r_squared': coefficient of determination (R^2)
        - 'rmse': root mean squared error
    """
    # Fit polynomial using numpy's polyfit
    coeffs = np.polyfit(x, y, degree)
    
    # Reverse coefficients to match the convention
    # np.polyfit returns highest degree first, we want constant term first
    coeffs = coeffs[::-1]
    
    # Calculate R^2 and RMSE
    y_pred = np.polyval(coeffs[::-1], x)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    return {
        'params': coeffs,
        'r_squared': r_squared,
        'rmse': rmse,
    }


def fit_exponential(
    x: np.ndarray, y: np.ndarray, p0: Optional[Tuple[float, float]] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Fit an exponential regression model: y = amplitude * exp(decay * x).

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        Input data.
    y : ndarray of shape (n_samples,)
        Output data.
    p0 : tuple, optional
        Initial guess for the parameters (amplitude, decay).
        If None, defaults to (1.0, 1.0).

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'params': array of shape (2,) with [amplitude, decay]
        - 'r_squared': coefficient of determination (R^2)
        - 'rmse': root mean squared error
    """
    if p0 is None:
        p0 = (1.0, 0.1)
    
    # Define the exponential function
    def exp_func(x, amplitude, decay):
        return amplitude * np.exp(decay * x)
    
    # Fit exponential using scipy's curve_fit
    params, pcov = optimize.curve_fit(exp_func, x, y, p0=p0)
    amplitude, decay = params
    
    # Calculate R^2 and RMSE
    y_pred = exp_func(x, amplitude, decay)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    return {
        'params': np.array([amplitude, decay]),
        'r_squared': r_squared,
        'rmse': rmse,
    }
