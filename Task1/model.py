# model.py

import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_output, learning_rate):
        self.d_weights = np.dot(self.input.T, d_output)
        self.d_biases = np.sum(d_output, axis=0, keepdims=True)
        self.d_input = np.dot(d_output, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
        return self.d_input


class ReLU:
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, d_output):
        return d_output * (self.input > 0)


class Sigmoid:
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, d_output):
        return d_output * self.output * (1 - self.output)


class MSELoss:
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        return np.mean((prediction - target) ** 2)

    def backward(self):
        return 2 * (self.prediction - self.target) / self.target.size
