"""Training utilities for QML-CNN-Welds."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import yaml

from .data import load_digits_dataloaders
from .models import HybridQNN, SimpleCNN


def train_from_config(config_path: str) -> dict[str, float]:
    """Train a configured model and return final train/validation metrics."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    batch_size = int(cfg.get("batch_size", 32))
    num_epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("learning_rate", 1e-3))
    seed = int(cfg.get("seed", 42))
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "cnn")).lower()
    num_classes = int(model_cfg.get("num_classes", 10))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader = load_digits_dataloaders(batch_size=batch_size, random_state=seed)

    if model_type == "cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif model_type == "hybrid":
        model = HybridQNN(n_qubits=4, n_layers=2, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc = 0.0
    val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
        )

    return {"train_accuracy": float(train_acc), "val_accuracy": float(val_acc)}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """Return classification accuracy for a dataloader."""
    model.eval()
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total if total > 0 else 0.0
