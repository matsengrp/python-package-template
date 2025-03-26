# Regressor

[![Tests](https://github.com/yourusername/regressor/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/regressor/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/regressor.svg)](https://badge.fury.io/py/regressor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple package for simulating and fitting regression data.

## Features

- Simulate data from various regression models:
  - Linear regression
  - Polynomial regression
  - Exponential regression
- Fit regression models to data with easy-to-use API
- Get detailed regression statistics (R², RMSE)

## Installation

```bash
pip install regressor
```

Or using pixi:

```bash
pixi add regressor
```

## Usage

Here's a simple example of simulating and fitting linear regression data:

```python
import numpy as np
import matplotlib.pyplot as plt
from regressor import simulate_linear, fit_linear

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
print(f"Fitted parameters: Intercept = {result['params'][0]:.4f}, Slope = {result['params'][1]:.4f}")
print(f"R² = {result['r_squared']:.4f}")
print(f"RMSE = {result['rmse']:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label="Data")
plt.plot(
    np.sort(x),
    result['params'][0] + result['params'][1] * np.sort(x),
    'r-',
    label="Fitted model"
)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

See the `examples` directory for more usage examples.

## Development

This project uses pixi for dependency management.

### Setup for development

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/regressor.git
   cd regressor
   ```

2. Install development dependencies with pixi
   ```bash
   pixi install
   ```

### Running tests

```bash
pixi run test
```

### Linting and formatting

```bash
pixi run lint
```

### Type checking

```bash
pixi run type-check
```

### Building the package

```bash
pixi run build
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
