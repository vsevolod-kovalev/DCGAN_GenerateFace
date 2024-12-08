import numpy as np
from constants import *

class Dense:
    def __init__(self, batch_size, input_shape, num_neurons, activation):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.activation = activation
        self.num_neurons = num_neurons

        self.W = np.random.uniform(-0.1, 0.1, (self.input_size, num_neurons))
        # self.W = np.random.randn(self.input_size, self.num_neurons) * np.sqrt(2 / self.input_size)


        self.B = np.zeros((num_neurons))

        self.Z = np.zeros((batch_size, num_neurons))
        self.A = np.zeros_like(self.Z)

    def forward(self, input):
        # input shape: (batch_size, height, width, depth)
        self.X_flattened = input.reshape(self.batch_size, -1)

        # (batch_size, input_size) dot (input_size, num_neurons) -> (batch_size, num_neurons)
        self.Z = self.X_flattened.dot(self.W) + self.B

        match self.activation.lower():
            case 'relu':
                self.A = np.maximum(0, self.Z)
                self.dA_dZ = np.where(self.Z > 0, 1, 0)
            case 'lrelu' | 'leaky_relu':
                self.A = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA)
                self.dA_dZ = np.where(self.Z > 0, 1, LRELU_ALPHA)
            case 'sigmoid':
                self.Z = np.clip(self.Z, -500, 500)
                sigmoid = 1 / (1 + np.exp(-self.Z))
                self.A = sigmoid
                self.dA_dZ = sigmoid * (1 - sigmoid)
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A

    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        if not np.shape(gradient) == 2:
            gradient = gradient.reshape((self.batch_size, -1))
        # gradient shape: (batch_size, num_neurons)
        # dL_dZ: (batch_size, num_neurons)
        dL_dZ = gradient * self.dA_dZ
        # (input_size, batch_size) dot (batch_size, num_neurons) -> (input_size, num_neurons)
        dL_dW = self.X_flattened.T.dot(dL_dZ)
        dL_dB = np.sum(dL_dZ, axis=0)

        W_delta_accumulated += dL_dW
        B_delta_accumulated += dL_dB

        # dL_dX: (batch_size, input_size)
        # (batch_size, num_neurons) dot (num_neurons, input_size) -> (batch_size, input_size)
        updated_gradient = dL_dZ.dot(self.W.T)

        return updated_gradient.reshape((self.batch_size,) + self.input_shape)
