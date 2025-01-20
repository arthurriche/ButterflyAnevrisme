import torch
import torch.nn as nn

class FlowPredictionLoss(nn.Module):
    def __init__(self):
        super(FlowPredictionLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')  # Change to 'sum' for accumulation

    def forward(self, predictions, targets):
        """
        Compute the MSE loss between predictions and targets.

        Args:
            predictions (list of Data): List of predicted graph states for each timestep.
            targets (list of Data): List of actual graph states for each timestep.

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        total_loss = torch.tensor(0.0, device=predictions[0].x.device, requires_grad=True)
        for pred, target in zip(predictions, targets):
            total_loss = total_loss + self.mse_loss(pred.x, target.x)
        return total_loss / len(predictions)  # Return tensor instead of float
