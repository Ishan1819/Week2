\_From Scratch Neural-Net Regression with NumPy Only\*

_Overview_
This project implements a fully connected feedforward neural network from scratch using only NumPy for a regression task. The model is trained on a synthetic cubic function with added noise, demonstrating effective learning through forward and backward propagation.

_Features Implemented_
Custom Dense Layers with weights and biases
Activation Functions: ReLU and Sigmoid (ReLU used in model)
Loss Function: Mean Squared Error (MSE)
Optimizer: Manual Stochastic Gradient Descent (SGD)
Gradient Backpropagation: Implemented manually for all layers
Training Visualization: Loss curve and prediction vs. ground truth
Evaluation Metric: R² Score

_Neural Network Architecture_
Input Dimension: 1 (univariate input)

_Architecture_:
Input (1) → Dense(64) → ReLU → Dense(64) → ReLU → Dense(1) → Output
Output Dimension: 1 (regression target)

_Dataset_
We generate a synthetic dataset using the cubic function:

_Training Details_ -
Epochs: 900
Learning Rate: 0.01
Batch Size: Full batch (all data used each step)
Loss Function: MSE

_Forward & Backward Pass_ -
Each layer: Stores input during forward pass. Computes gradients with respect to weights, biases, and inputs during backpropagation. Updates parameters using SGD

_Results_ -
Convergence
The training loss consistently decreased, indicating proper gradient flow and convergence.
Final MSE loss dropped significantly, and predictions closely matched the true function.

_Evaluation_
R² Score: ~0.99, indicating excellent fit.

_Convergence Analysis_
The model learns effectively within 900 epochs.
ReLU was chosen to introduce non-linearity while avoiding vanishing gradients.
Proper weight initialization and learning rate were crucial for stable convergence.

_Requirements_
pip install numpy matplotlib notebook

_Key Learnings_
Building neural networks from scratch enhances understanding of:
Gradient computation.
Parameter updates.
The role of activations and loss functions.
Visualization helps monitor convergence and debug learning issues.
