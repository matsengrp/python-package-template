"""Basic usage example for the regressor package."""

import numpy as np
import matplotlib.pyplot as plt
from regressor import (
    simulate_linear,
    simulate_polynomial,
    simulate_exponential,
    fit_linear,
    fit_polynomial,
    fit_exponential,
)


def plot_fit(x, y, y_pred, title):
    """Helper function for plotting data and fitted model."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label="Data")
    plt.plot(np.sort(x), y_pred[np.argsort(x)], 'r-', label="Fitted model")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def linear_example():
    """Example of simulating and fitting linear data."""
    print("Linear Regression Example")
    print("-" * 50)
    
    # Simulate data
    x, y = simulate_linear(
        n_samples=100,
        slope=2.5,
        intercept=1.0,
        noise_level=2.0,
        random_state=42
    )
    
    # Fit model
    result = fit_linear(x, y)
    
    # Print results
    print(f"True parameters: Intercept = 1.0, Slope = 2.5")
    print(f"Fitted parameters: Intercept = {result['params'][0]:.4f}, Slope = {result['params'][1]:.4f}")
    print(f"R² = {result['r_squared']:.4f}")
    print(f"RMSE = {result['rmse']:.4f}")
    
    # Predict using fitted parameters
    y_pred = result['params'][0] + result['params'][1] * x
    
    # Plot results
    plot_fit(x, y, y_pred, "Linear Regression")
    print()


def polynomial_example():
    """Example of simulating and fitting polynomial data."""
    print("Polynomial Regression Example")
    print("-" * 50)
    
    # Simulate data
    x, y = simulate_polynomial(
        n_samples=100,
        coefficients=[1.0, 0.5, 2.0],  # y = 1 + 0.5x + 2x²
        noise_level=3.0,
        random_state=42
    )
    
    # Fit model
    result = fit_polynomial(x, y, degree=2)
    
    # Print results
    print(f"True parameters: [1.0, 0.5, 2.0]")
    print(f"Fitted parameters: {', '.join([f'{p:.4f}' for p in result['params']])}")
    print(f"R² = {result['r_squared']:.4f}")
    print(f"RMSE = {result['rmse']:.4f}")
    
    # Predict using fitted parameters
    y_pred = sum(coef * x**i for i, coef in enumerate(result['params']))
    
    # Plot results
    plot_fit(x, y, y_pred, "Polynomial Regression (degree 2)")
    print()


def exponential_example():
    """Example of simulating and fitting exponential data."""
    print("Exponential Regression Example")
    print("-" * 50)
    
    # Simulate data
    x, y = simulate_exponential(
        n_samples=100,
        amplitude=2.0,
        decay=0.3,
        noise_level=0.5,
        random_state=42
    )
    
    # Fit model
    result = fit_exponential(x, y)
    
    # Print results
    print(f"True parameters: Amplitude = 2.0, Decay = 0.3")
    print(f"Fitted parameters: Amplitude = {result['params'][0]:.4f}, Decay = {result['params'][1]:.4f}")
    print(f"R² = {result['r_squared']:.4f}")
    print(f"RMSE = {result['rmse']:.4f}")
    
    # Predict using fitted parameters
    y_pred = result['params'][0] * np.exp(result['params'][1] * x)
    
    # Plot results
    plot_fit(x, y, y_pred, "Exponential Regression")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run examples
    linear_example()
    polynomial_example()
    exponential_example()
