# src/mean_estimator/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class MeanTrainer:
    """Trainer for the mean estimation model."""
    
    def __init__(self, model, dataset, batch_size=32, lr=0.01):
        """Initialize the trainer.
        
        Args:
            model: The MeanEstimator model.
            dataset: The PointsDataset.
            batch_size: Batch size for training.
            lr: Learning rate for SGD.
        """
        self.model = model
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        
    def train_epoch(self):
        """Train for one epoch.
        
        Returns:
            Average loss for the epoch.
        """
        epoch_losses = []
        for batch in self.dataloader:
            # Get the current mean estimate
            pred_mean = self.model()
            
            # Calculate loss (MSE between each point and the estimated mean)
            loss = self.loss_fn(pred_mean.expand_as(batch), batch)
            
            # Backpropagate and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.losses.append(avg_loss)
        return avg_loss
        
    def train(self, epochs=100):
        """Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train for.
            
        Returns:
            List of losses for each epoch.
        """
        for _ in range(epochs):
            self.train_epoch()
        return self.losses
    
    def plot_losses(self, title="Training Loss"):
        """Plot the training losses.
        
        Args:
            title: Title for the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.show()

