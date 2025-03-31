# tests/test_mean_estimator.py
import torch
import numpy as np
from nose.tools import assert_almost_equal, assert_greater
from mean_estimator.model import MeanEstimator
from mean_estimator.data import PointsDataset
from mean_estimator.trainer import MeanTrainer
from mean_estimator.experiment import generate_points


def test_mean_estimator_init():
    """Test MeanEstimator initialization."""
    model = MeanEstimator(initial_guess=0.5)
    assert_almost_equal(model().item(), 0.5, places=5)


def test_points_dataset():
    """Test PointsDataset functionality."""
    points = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    dataset = PointsDataset(points)
    
    assert len(dataset) == 5
    assert dataset[0].item() == 1.0
    assert dataset[4].item() == 5.0


def test_generate_points():
    """Test that generate_points produces expected distribution."""
    points = generate_points(num_points=10000, true_mean=2.0, std=0.1)
    
    # Mean should be close to 2.0
    sample_mean = torch.mean(points).item()
    assert_almost_equal(sample_mean, 2.0, places=1)


def test_training_converges():
    """Test that the model converges to the true mean."""
    true_mean = 3.0
    points = torch.tensor([true_mean] * 100)  # Deterministic points for testing
    dataset = PointsDataset(points)
    
    model = MeanEstimator(initial_guess=0.0)  # Start far from true mean
    trainer = MeanTrainer(model, dataset, lr=0.1)
    
    initial_loss = trainer.train_epoch()
    for _ in range(50):  # Train for 50 epochs
        trainer.train_epoch()
    final_loss = trainer.losses[-1]
    
    # Final loss should be lower than initial loss
    assert_greater(initial_loss, final_loss)
    
    # Final mean estimate should be close to true mean
    final_mean = model().item()
    assert_almost_equal(final_mean, true_mean, places=1)

