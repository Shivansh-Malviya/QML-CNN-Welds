"""Tests for the demonstration training pipeline."""

import sys
import pathlib

import yaml

# Add the src directory to the Python path so qml_cnn_welds can be imported without installation.
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

from qml_cnn_welds.train import train_from_config


def test_train_from_config(tmp_path) -> None:
    """Ensure that training runs and returns reasonable metrics."""
    config = {
        "batch_size": 16,
        "epochs": 1,
        "learning_rate": 0.01,
        "model": {"type": "cnn", "num_classes": 10},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    metrics = train_from_config(str(config_path))
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["val_accuracy"] <= 1.0