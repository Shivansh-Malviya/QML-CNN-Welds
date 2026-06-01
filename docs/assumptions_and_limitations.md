# Assumptions and Limitations

QML-CNN-Welds is a research-code showcase. The retained implementation is intentionally modest and should not be read as a complete weld-inspection product.

## Assumptions

1. The default dataset is a compact public proxy, not weld imagery.
2. Input images in the default path are grayscale, fixed-size, and small.
3. The baseline CNN is intentionally lightweight.
4. The default split is train-validation only; there is no independent weld test set.
5. The optional quantum model path is experimental and dependency-gated.

## Limitations

1. Weld imagery and Kaggle dataset files are not versioned.
2. Validation metrics on the digits proxy dataset do not measure weld-defect performance.
3. The repository does not claim quantum advantage.
4. The code does not include deployment infrastructure, model registry handling, data governance, or production monitoring.
5. The hybrid model path is a compact implementation sketch rather than a tuned QML architecture.

These boundaries keep the public repository aligned with what the retained code demonstrates: a small, inspectable image-classification pipeline with a documented QML extension point.
