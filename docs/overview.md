# Project Overview

QML-CNN-Welds explores image-classification pipelines connected to TIG weld-defect detection. The codebase presents a compact package around data loading, CNN model construction, training, evaluation, and an optional hybrid quantum-classical model path.

Weld inspection is a quality-control task in which image or sensor observations are classified into acceptable and defective categories. The source project work used weld-oriented labels such as good weld, burn through, contamination, lack of fusion, misalignment, and lack of penetration. The public package keeps that modeling direction while using a compact public proxy dataset for the default executable path.

The default CNN run uses the scikit-learn digits dataset. That choice keeps the repository lightweight and runnable without distributing weld imagery or Kaggle-bound dataset files. The proxy dataset preserves the shape of an image-classification workflow, but its validation accuracy is not evidence of weld-inspection performance.

The hybrid model path shows how a small PennyLane quantum circuit can be connected to a PyTorch model. It is included as an experimental extension point and is not presented as a performance improvement over the classical baseline.
