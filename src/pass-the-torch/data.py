# src/mean_estimator/data.py
import torch
from torch.utils.data import Dataset


class PointsDataset(Dataset):
    """Dataset for points with a known distribution."""
    
    def __init__(self, points):
        """Initialize with a tensor of points.
        
        Args:
            points: Tensor of shape (N,) containing the data points.
        """
        self.points = points
        
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx]

