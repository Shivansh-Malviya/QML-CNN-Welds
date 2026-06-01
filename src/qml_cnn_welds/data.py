"""Data loading utilities for compact image-classification runs."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_digits_dataloaders(
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders for the digits proxy dataset."""
    digits = load_digits()
    features = digits.images.astype(np.float32)
    targets = digits.target.astype(np.int64)

    features = features / 16.0
    features = np.expand_dims(features, 1)

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        targets,
        test_size=test_size,
        stratify=targets,
        random_state=random_state,
    )

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

    generator = None
    if random_state is not None:
        generator = torch.Generator()
        generator.manual_seed(random_state)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
