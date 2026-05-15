# Iris Species Classifier — Simple ANN

A feedforward neural network built with PyTorch that classifies iris flowers into three species based on sepal and petal measurements.

## Overview

This project trains a 2-hidden-layer artificial neural network on the classic [Iris dataset](https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv). The model achieved 30/30 correct predictions on the test set (100% test accuracy).

## Model Architecture

```
Input (4 features) → FC1 (8 neurons, ReLU) → FC2 (8 neurons, ReLU) → Output (3 classes)
```

| Layer | Type | Size |
|-------|------|------|
| Input | — | 4 (sepal length, sepal width, petal length, petal width) |
| Hidden 1 | Linear + ReLU | 8 neurons |
| Hidden 2 | Linear + ReLU | 8 neurons |
| Output | Linear | 3 (setosa, versicolor, virginica) |

## Dataset

The [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) contains 150 samples across 3 species with 4 measurements each (in cm).

| Label | Species |
|-------|---------|
| 0 | Iris setosa |
| 1 | Iris versicolor |
| 2 | Iris virginica |

80/20 train/test split with `random_state=42`.

## Requirements

```
torch
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install torch pandas matplotlib scikit-learn
```

## Usage

Run all cells in `simpleANN.ipynb` top to bottom. The notebook will:

1. Load and preprocess the Iris dataset
2. Split into train/test sets
3. Train the model for 800 epochs
4. Plot the loss curve
5. Evaluate on the test set
6. Run predictions on custom out-of-sample test cases

## Training

- Optimizer: Adam (lr=0.001)
- Loss function: CrossEntropyLoss
- Epochs: 800
- Random seed: 41

## Results

The model correctly classifies all 30 test samples. The loss curve shows steady convergence over 800 epochs.

## Custom Predictions

To predict a new flower, pass a tensor of 4 measurements:

```python
import torch

sample = torch.tensor([5.1, 3.5, 1.4, 0.2], dtype=torch.float32)
labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

with torch.no_grad():
    out = model(sample)
    pred = torch.argmax(out).item()
    print(labels[pred])
```

## Notes

- The model uses a closed-world assumption — it will always output one of the three training classes even for flower species it has never seen.
- Hidden layer size of 8 was chosen arbitrarily and works well for this small dataset.
