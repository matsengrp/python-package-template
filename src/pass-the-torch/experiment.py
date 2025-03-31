
# src/mean_estimator/experiment.py
import torch
import matplotlib.pyplot as plt
from .model import MeanEstimator
from .data import PointsDataset
from .trainer import MeanTrainer


def generate_points(num_points=1000, true_mean=1.0, std=0.5):
    """Generate random points with a given mean.
    
    Args:
        num_points: Number of points to generate.
        true_mean: The true mean of the distribution.
        std: Standard deviation of the distribution.
        
    Returns:
        Tensor of shape (num_points,) with the generated points.
    """
    return torch.normal(mean=true_mean, std=std, size=(num_points,))


def run_experiment(learning_rates=[0.01, 0.1, 0.5], epochs=100, initial_guess=0.0, true_mean=1.0):
    """Run experiments with different learning rates.
    
    Args:
        learning_rates: List of learning rates to try.
        epochs: Number of epochs to train for.
        initial_guess: Initial guess for the mean.
        true_mean: True mean of the generated data.
        
    Returns:
        Dictionary mapping learning rates to loss histories.
    """
    points = generate_points(true_mean=true_mean)
    dataset = PointsDataset(points)
    
    results = {}
    
    for lr in learning_rates:
        model = MeanEstimator(initial_guess=initial_guess)
        trainer = MeanTrainer(model, dataset, lr=lr)
        
        losses = trainer.train(epochs=epochs)
        final_mean = model().item()
        
        results[lr] = {
            "losses": losses,
            "final_mean": final_mean,
            "true_mean": true_mean
        }
        
        print(f"Learning rate {lr}: Final mean = {final_mean:.4f} (True mean = {true_mean:.4f})")
        
    plot_comparison(results, epochs)
    return results


def plot_comparison(results, epochs):
    """Plot loss curves for different learning rates.
    
    Args:
        results: Results dictionary from run_experiment.
        epochs: Number of epochs trained.
    """
    plt.figure(figsize=(12, 8))
    
    for lr, result in results.items():
        plt.plot(range(epochs), result["losses"], label=f"LR = {lr}, Final Mean = {result['final_mean']:.4f}")
    
    plt.title("Loss Comparison for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale often helps visualize convergence
    plt.show()

