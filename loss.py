import torch
import torch.nn as nn

class FlowPredictionLoss(nn.Module):
    def __init__(self):
        super(FlowPredictionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        Compute the MSE loss between predictions and targets.

        Args:
            predictions (list of Data): List of predicted graph states for each timestep.
            targets (list of Data): List of actual graph states for each timestep.

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        loss = 0
        for pred, target in zip(predictions, targets):
            loss += self.mse_loss(pred.x, target.x)
        return loss / len(predictions)
