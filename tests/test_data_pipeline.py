"""Tests for the data loading utilities."""

import sys
import pathlib

# Add the src directory to the Python path so qml_cnn_welds can be imported without installation.
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

import qml_cnn_welds.data as data


def test_digits_loader_shapes() -> None:
    train_loader, val_loader = data.load_digits_dataloaders(batch_size=16)
    # Check that we get at least one batch
    X_batch, y_batch = next(iter(train_loader))
    assert X_batch.shape[1:] == (1, 8, 8)
    assert y_batch.ndim == 1
    assert len(y_batch) <= 16