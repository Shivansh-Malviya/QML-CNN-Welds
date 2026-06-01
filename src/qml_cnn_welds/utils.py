"""Utility functions for QML-CNN-Welds."""

from __future__ import annotations

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the proportion of correct class predictions."""
    preds = predictions.argmax(dim=1)
    return float((preds == targets).sum().item() / targets.size(0))
