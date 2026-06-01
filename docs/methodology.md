# Methodology

The repository keeps a compact methodology: load a small image dataset, train a baseline CNN, report train and validation accuracy, and keep a separate optional path for hybrid quantum-classical experimentation.

## Data

The default run uses the `digits` dataset from scikit-learn. It contains 1,797 grayscale images with 8 by 8 pixels and ten class labels. The dataset is used as a proxy for image classification rather than as weld data.

Pixel values are normalized to the range `[0, 1]`, reshaped to `(batch_size, 1, 8, 8)`, and split with an 80/20 stratified train-validation split. The split seed is fixed by default for stable checks.

## Classical CNN

`qml_cnn_welds.models.SimpleCNN` defines the default model. It uses two convolutional layers, ReLU activations, max pooling, and two fully connected layers. The architecture is intentionally small enough for CPU execution while still exercising the full training and evaluation path.

The model is a baseline pipeline component, not a weld-specific production architecture. Real weld imagery would require a dataset-specific loader, image-size handling, class mapping, augmentation strategy, and a separate evaluation protocol.

## Hybrid Quantum-Classical Model

`qml_cnn_welds.models.HybridQNN` defines an optional PennyLane-backed model. A linear preprocessing layer maps the flattened image to a low-dimensional vector, a parameterized quantum circuit produces expectation values, and a final linear layer maps those values to class logits.

The hybrid model documents the experimental QML direction from the source work. It is disabled in the default configuration and is not used to claim quantum advantage.

## Training

`qml_cnn_welds.train.train_from_config` reads a YAML configuration, builds the selected model, trains with Adam and cross-entropy loss, and returns final train and validation accuracy. The default configuration uses the classical CNN path.
