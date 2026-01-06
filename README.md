# MNIST Neural Network From Scratch

A fully-connected neural network implemented **entirely from scratch using NumPy**, trained on the MNIST digit classification dataset.

This project intentionally avoids deep learning frameworks such as **PyTorch** or **TensorFlow** to focus on understanding the **core mathematics and mechanics** behind neural networks.

---

## Project Motivation

Most modern ML workflows rely on high-level frameworks that abstract away the learning process.

While this is great for productivity, it often hides:

* how forward propagation actually works
* how gradients flow backward
* how architectural choices affect learning

This project was built to **understand neural networks from first principles**, not to chase state-of-the-art accuracy.

---

## Project Structure & Phases

The project was built incrementally in **two clear phases**, with evaluation happening within each phase.

---

## Phase 1: Single Hidden Layer Neural Network

The first phase focuses on building the **simplest working neural network** for MNIST.

### Architecture

* Input layer: **784 neurons** (28×28 image flattened)
* One hidden layer
* ReLU activation
* Output layer: **10 neurons** (Softmax)

### Key Characteristics

* Forward propagation implemented manually
* Backpropagation fully derived by hand
* Explicit weight and bias matrices
* One-hot encoding + cross-entropy loss
* Gradient descent optimization
* No abstractions, no modularization

The goal in this phase was **correctness and intuition**, not flexibility.

### Result

* Training & evaluation performed within this phase
* **Accuracy achieved: ~86%**

This validated that the math and gradient flow were correct.


The handwritten notes document:

* Input representation
* Forward propagation
* Backpropagation with matrix dimensions
* Gradient descent updates

They directly map to the implemented code.

Relevant notebook:

* `nn-implementation.ipynb`

---

## Phase 2 — Modular & Scalable Architecture

After validating the single hidden layer network, the code was refactored into a **fully modular neural network implementation**.

### Key Refactor

The architecture is now defined using a single configuration variable:

```python
layer_dims = [784, 128, 64, 10]
```

### What This Enables

* Arbitrary number of hidden layers
* No hardcoded layer logic
* Automatic parameter initialization
* Generic forward propagation loop
* Depth-agnostic backpropagation
* Clean separation of concerns

Adding or removing layers requires **no changes to the training or backpropagation code**.

This phase transforms the implementation from a fixed demo into a **general-purpose feedforward neural network**.

### Result

* Training & evaluation performed within this phase
* **Accuracy achieved: ~92% on the dev set**

This improvement comes purely from increased model capacity and better structure — **no framework magic involved**.

Relevant notebook:

* `nn-implementation-modular.ipynb`

---

## Mathematical Components Implemented

Across both phases, the following were implemented manually:

* Linear transformations (W · X + b)
* ReLU activation and derivative
* Softmax with numerical stability
* One-hot encoding
* Categorical cross-entropy loss
* Backpropagation through all layers
* Gradient descent parameter updates
* He initialization (modular phase)

All operations are fully vectorized using NumPy.

---

## Key Learnings

* Why explicit matrix dimensions matter
* How gradients propagate through layers
* Why Softmax + Cross-Entropy simplifies gradients
* How architectural flexibility improves performance
* Why frameworks feel “magical” only until you build this once

---

## Possible Extensions

* Mini-batch gradient descent
* L2 regularization
* Learning rate scheduling
* Deeper architectures
* Confusion matrix & error analysis
* PyTorch reimplementation for comparison

---

## Final Note

This project reinforced a simple idea:

> Neural networks are not magic.
> They are carefully chained matrix operations with gradients flowing backward.

Building one from scratch makes every deep learning paper and API far easier to reason about.

