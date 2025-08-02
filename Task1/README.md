**From Scratch Neural-Net Regression with NumPy Only**

__________________________________________________________________________________________________________________________________________________________________________________________
**Overview**

This project implements a fully connected feedforward neural network from scratch using only NumPy for a regression task. The model is trained on a synthetic cubic function with added noise, demonstrating effective learning through forward and backward propagation.

__________________________________________________________________________________________________________________________________________________________________________________________
**Features Implemented**

Custom Dense Layers with weights and biases
Activation Functions: ReLU and Sigmoid (ReLU used in model)
Loss Function: Mean Squared Error (MSE)
Optimizer: Manual Stochastic Gradient Descent (SGD)
Gradient Backpropagation: Implemented manually for all layers
Training Visualization: Loss curve and prediction vs. ground truth
Evaluation Metric: R² Score

__________________________________________________________________________________________________________________________________________________________________________________________
**Neural Network Architecture**

Input Dimension: 1 (univariate input)

__________________________________________________________________________________________________________________________________________________________________________________________
**Architecture:**

Input (1) → Dense(64) → ReLU → Dense(64) → ReLU → Dense(1) → Output

Output Dimension: 1 (regression target)

__________________________________________________________________________________________________________________________________________________________________________________________
**Dataset**

We generate a synthetic dataset using the cubic function:

__________________________________________________________________________________________________________________________________________________________________________________________
**Training Details**

Epochs: 900
Learning Rate: 0.01
Batch Size: Full batch (all data used each step)
Loss Function: MSE

__________________________________________________________________________________________________________________________________________________________________________________________
**Forward & Backward Pass**

Each layer: Stores input during forward pass.

Computes gradients with respect to weights, biases, and inputs during backpropagation.

Updates parameters using SGD

__________________________________________________________________________________________________________________________________________________________________________________________
**Results - Convergence**

The training loss consistently decreased, indicating proper gradient flow and convergence. Final MSE loss dropped significantly, and predictions closely matched the true function.

__________________________________________________________________________________________________________________________________________________________________________________________
**Evaluation**

R² Score: ~0.99, indicating excellent fit.

__________________________________________________________________________________________________________________________________________________________________________________________
**Convergence Analysis**

The model learns effectively within 900 epochs. ReLU was chosen to introduce non-linearity while avoiding vanishing gradients. Proper weight initialization and learning rate were crucial for stable convergence.

__________________________________________________________________________________________________________________________________________________________________________________________
**Requirements**

pip install numpy matplotlib notebook

__________________________________________________________________________________________________________________________________________________________________________________________
**Key Learnings**
Building neural networks from scratch enhances understanding of:
Gradient computation.
Parameter updates.
The role of activations and loss functions.
Visualization helps monitor convergence and debug learning issues.



