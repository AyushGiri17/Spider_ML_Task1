# Fashion MNIST Classification using PyTorch

## Overview

This project implements a custom Neural Network using PyTorch for image classification on the Fashion-MNIST dataset.

The objective is to classify grayscale clothing images into one of 10 fashion categories. The model is trained using a complete deep learning pipeline including data preprocessing, dataset creation, training, evaluation, model saving, prediction generation, and visualization of training metrics.

---

## Dataset

Fashion-MNIST consists of 28×28 grayscale images represented as 784 pixel values.

Number of classes: 10

Classes:

- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

## Project Features

- Custom PyTorch Dataset implementation
- DataLoader based mini-batch training
- Custom Neural Network architecture
- Shared feature extraction layer
- Multi-branch architecture
- Skip connection in the left branch
- SGD optimizer
- Cross Entropy Loss
- Training and evaluation pipeline
- Accuracy and loss tracking
- Model weight saving
- Test prediction generation
- Accuracy and loss visualization

---

## Model Architecture

Input Layer:

784 Features

↓

Shared Hidden Layer:

784 → 16

↓

Split into Two Branches

### Left Branch

16 → 8

8 → 8

Skip Connection:

Output = Layer2 Output + Layer1 Output

### Right Branch

16 → 12

12 → 8

### Concatenation

Left Output (8)

+

Right Output (8)

↓

16 Features

### Output Layer

16 → 10

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 100 |
| Batch Size | 32 |
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Loss Function | CrossEntropyLoss |
| Random Seed | 42 |

---

## Results

Final Test Accuracy:

**86.63%**

Final Test Loss:

**0.2640**

---

## Generated Files

### Model Weights

Stored inside:

```text
save_models/fashion_mnist_model.pth
```

Contains the trained neural network weights.

### Submission File

Stored as:

```text
submission.csv
```

Contains:

```text
Id,Label
```

where:

- Id = Sample Index
- Label = Predicted Class

### Training Plots

Stored inside:

```text
plots/training_plots.png
```

Contains:

- Loss vs Epoch
- Accuracy vs Epoch

---

## Project Structure

```text
base_task/
│
├── train.py
├── README.md
├── submission.csv
│
├── save_models/
│   └── fashion_mnist_model.pth
│
└── plots/
    └── training_plots.png
```

---

## How to Run

Install dependencies:

```bash
pip install torch pandas matplotlib
```

Run training:

```bash
python train.py
```

The script will:

1. Load the Fashion-MNIST dataset
2. Train the neural network
3. Evaluate model performance
4. Save model weights
5. Generate submission.csv
6. Save training plots

---

## Libraries Used

- Python
- PyTorch
- Pandas
- Matplotlib
- NumPy

---

## Author

Ayush

Spider ML Task 1 – Base Task Submission