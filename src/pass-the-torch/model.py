# src/mean_estimator/model.py
import torch
import torch.nn as nn


class MeanEstimator(nn.Module):
    """Simple model to estimate the mean of a distribution."""
    
    def __init__(self, initial_guess=0.0):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(initial_guess, dtype=torch.float32))
        
    def forward(self, x=None):
        """Return the current mean estimate.
        
        Args:
            x: Ignored. The model doesn't require input for prediction.
            
        Returns:
            The current mean estimate.
        """
        return self.mean


