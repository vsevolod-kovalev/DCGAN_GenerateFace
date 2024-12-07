import numpy as np
from constants import *

class Dense:
    def __init__(self, input_shape, num_neurons, activation):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.activation = activation
        self.num_neurons = num_neurons
        self.W = np.random.uniform(-0.1, 0.1, (num_neurons, self.input_size))
        self.B = np.zeros((num_neurons))
        self.Z = np.zeros((num_neurons))
        self.A = np.zeros_like(self.Z)
    def forward(self, input):
        self.X_flattened = np.ravel(input)
        self.Z = np.dot(self.W, self.X_flattened) + self.B
        match self.activation.lower():
            case 'relu':
                self.A, self.dA_dZ = np.maximum(0, self.Z), np.where(self.Z > 0, 1, 0)
            case 'lrelu' | 'leaky_relu':
                self.A, self.dA_dZ = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA), np.where(self.Z > 0, 1, LRELU_ALPHA)
            case 'sigmoid':
                sigmoid = 1 / (1 + np.exp(-self.Z))
                self.A, self.dA_dZ = sigmoid, sigmoid * (1 - sigmoid)
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A
    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        if not np.ndim(gradient) == 1:
            gradient = np.ravel(gradient)
        dL_dZ = gradient * self.dA_dZ
        dL_dW = np.outer(dL_dZ, self.X_flattened)
        dL_dB = dL_dZ
        W_delta_accumulated += dL_dW
        B_delta_accumulated += dL_dB
        updated_gradient = np.dot(self.W.T, dL_dZ)
        return updated_gradient.reshape(self.input_shape)