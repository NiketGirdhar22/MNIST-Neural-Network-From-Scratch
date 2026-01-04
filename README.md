# MNIST Neural Network From Scratch

A fully-connected neural network implemented **from scratch using NumPy**, trained on the MNIST digit classification dataset.  
This project focuses on understanding the **core mechanics of neural networks** without relying on deep learning frameworks such as PyTorch or TensorFlow.

---

## Project Overview

This repository implements a simple feedforward neural network with:

- Manual **forward propagation**
- **ReLU** and **Softmax** activations
- **Cross-entropy loss**
- Full **backpropagation**
- **Gradient descent** optimization
- Training and evaluation on the MNIST dataset
- Loss and accuracy tracking

The goal is not state-of-the-art accuracy, but **deep conceptual understanding**.

---

## Model Architecture

Input Layer : 784 neurons (28×28 image flattened)
Hidden Layer : 10 neurons (ReLU)
Output Layer : 10 neurons (Softmax)

> This minimal architecture intentionally highlights learning limitations and optimization behavior.

---

## Mathematical Components Implemented

- Linear transformations
- ReLU activation and derivative
- Softmax with numerical stability
- One-hot encoding
- Categorical cross-entropy loss
- Backpropagation through all layers
- Parameter updates using gradient descent

---

## Example Output

```yaml
Iteration: 0   | Loss: 15.61 | Accuracy: 0.09
Iteration: 200 | Loss: 1.01  | Accuracy: 0.71
Iteration: 600 | Loss: 0.58  | Accuracy: 0.82
Iteration: 980 | Loss: 0.49  | Accuracy: 0.85
```

---

### Key Learnings

- Why proper weight initialization matters
- How gradients flow through layers
- The effect of learning rate on stability
- Why model capacity limits accuracy
- How loss and accuracy behave differently

---

### Possible Improvements

- Increase hidden layer size
- Mini-batch gradient descent
- L2 regularization
- Deeper architectures
- Comparison with PyTorch implementation
- Confusion matrix and per-class accuracy

---

## Acknowledgements

- MNIST Dataset
- NumPy & Matplotlib
Inspired by the need to truly understand neural networks

---