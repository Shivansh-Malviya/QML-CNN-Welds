# Project Lineage

This project comes from a Womanium Quantum+AI weld-classification effort involving classical CNN experiments, TIG weld-defect labels, Kaggle-hosted weld imagery, and hybrid quantum-classical model exploration.

## Source Material Represented

- TensorFlow weld-classification notebooks using TIG aluminium weld imagery and six weld-quality labels.
- PennyLane and PyTorch experiments combining convolutional feature extraction with quantum circuit layers.
- Smaller QML task notebooks covering variational classifiers, quanvolution-style experiments, and optimizer demonstrations.
- Womanium challenge README material that established the project context.

## Public Repository Boundary

The public package keeps the model-pipeline structure and QML experimentation direction. It does not version the Kaggle datasets, challenge PDFs, rendered notebook outputs, local filesystem paths, package-install cells, or environment-specific artifacts.

The default executable path uses a public proxy dataset so the package can be checked without bundling weld imagery. This makes the repository suitable as a compact research-code presentation while avoiding overclaiming real weld-classification performance.
