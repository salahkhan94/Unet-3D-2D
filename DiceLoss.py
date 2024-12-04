# DiceLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss implementation for multi-class segmentation.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Computes the Dice Loss between the logits and the targets.

        Args:
            logits (torch.Tensor): The raw output from the model of shape (N, C, D, H, W).
            targets (torch.Tensor): The ground truth labels of shape (N, D, H, W).

        Returns:
            torch.Tensor: The Dice Loss.
        """
        num_classes = logits.shape[1]
        # Apply softmax to logits to get probabilities
        probs = F.softmax(logits, dim=1)
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # Shape: (N, D, H, W, C)
        # Move class dimension to match probs
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # Shape: (N, C, D, H, W)
        # Flatten tensors
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        # Calculate Dice score
        intersection = (probs_flat * targets_flat).sum(-1)
        union = probs_flat.sum(-1) + targets_flat.sum(-1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        # Calculate mean Dice loss over classes
        dice_loss = 1 - dice_score.mean()
        return dice_loss
