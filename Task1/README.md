From Scratch Neural-Net Regression with NumPy Only
Overview
This project implements a fully connected feedforward neural network from scratch using only NumPy for a regression task. The model is trained on a synthetic cubic function with added noise, demonstrating effective learning through forward and backward propagation.

Features Implemented
Custom Dense Layers with weights and biases

Activation Functions: ReLU and Sigmoid (ReLU used in model)

Loss Function: Mean Squared Error (MSE)

Optimizer: Manual Stochastic Gradient Descent (SGD)

Gradient Backpropagation: Implemented manually for all layers

Training Visualization: Loss curve and prediction vs. ground truth

Evaluation Metric: R² Score

Neural Network Architecture
Input Dimension: 1 (univariate input)

Architecture:
Input (1) → Dense(64) → ReLU → Dense(64) → ReLU → Dense(1) → Output

Output Dimension: 1 (regression target)

Dataset
We generate a synthetic dataset using the cubic function:

Training Details
Epochs: 900

Learning Rate: 0.01

Batch Size: Full batch (all data used each step)

Loss Function: MSE

Forward & Backward Pass
Each layer: Stores input during forward pass.

Computes gradients with respect to weights, biases, and inputs during backpropagation.

Updates parameters using SGD

Results - Convergence
The training loss consistently decreased, indicating proper gradient flow and convergence. Final MSE loss dropped significantly, and predictions closely matched the true function.

Evaluation
R² Score: ~0.99, indicating excellent fit.

Convergence Analysis
The model learns effectively within 900 epochs. ReLU was chosen to introduce non-linearity while avoiding vanishing gradients. Proper weight initialization and learning rate were crucial for stable convergence.

Requirements
bash
Copy
Edit
pip install numpy matplotlib notebook
Key Learnings
Building neural networks from scratch enhances understanding of:

Gradient computation.

Parameter updates.

The role of activations and loss functions.

Visualization helps monitor convergence and debug learning issues.

