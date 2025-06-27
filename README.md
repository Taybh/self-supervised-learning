# Self-Supervised Learning

This repository provides utilities and experiments for self-supervised and semi-supervised learning on the CIFAR-10 dataset, with a focus on flexible data loading, augmentation, and support for both supervised and unsupervised training paradigms.

## Features

- **Flexible Dataset Handling:**  
  Utilities for creating labelled, unlabelled, and augmented datasets using PyTorch’s `Subset`, `ConcatDataset`, and custom wrappers.
- **Custom Transform Wrapper:**  
  The `AddTransform` class allows dynamic application of transformations to any dataset.
- **Augmentation Pipelines:**  
  Supports advanced augmentations including `RandAugment`, `RandomErasing`, and standard normalization.
- **Supervised and Unsupervised DataLoaders:**  
  Functions to easily create DataLoaders for both supervised and unsupervised (semi-supervised) training.
- **Reproducibility:**  
  Deterministic dataset splits and random seeds for consistent experiments.


## Repository Structure
├── Training_Unsupervised_Data_Augmentation.ipynb # Main notebook for unsupervised training experiments
├── create_datasets.py # Dataset creation and DataLoader utilities
├── transforms.py # Custom transform wrappers and augmentation logic
├── networks.ipynb # Model architectures and training scripts


## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- numpy

Install dependencies with:

```bash
pip install torch torchvision numpy
```

### Usage

#### 1. Clone the repository

```bash
git clone https://github.com/Taybh/self-supervised-learning.git
cd self-supervised-learning
```

#### 2. Prepare the Data

The code will automatically download CIFAR-10 to `~/workspace/data` or `./data` as needed.

#### 3. Run Experiments

- **Unsupervised/Semi-supervised Training:**  
  Use the `Training_Unsupervised_Data_Augmentation.ipynb` notebook to run experiments with different data splits and augmentations.

- **Custom Dataset Utilities:**  
  Import and use the dataset utilities in your own scripts:
  ```python
  from create_datasets import cifar10_unsupervised_dataloaders, cifar10_supervised_dataloaders
  from transforms import AddTransform
  ```

## Dataset Types Explained

- **Raw Dataset:**  
  The original CIFAR-10 dataset from torchvision.
- **Subset:**  
  A selection of indices from the original dataset, used to create labelled and unlabelled splits.
- **ConcatDataset:**  
  Combines multiple datasets (e.g., labelled and unlabelled) into one.
- **AddTransform:**  
  A custom Dataset wrapper that applies a given transform to each sample on-the-fly.
- **DataLoader:**  
  Provides efficient batching, shuffling, and multiprocessing for any dataset.

## Acknowledgements

This project uses the method presented in [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848) and leverages the [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) libraries.

## License

This project is for research and educational purposes. See individual files for license details if provided.
