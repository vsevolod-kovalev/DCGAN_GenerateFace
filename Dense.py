import numpy as np
from constants import *

class Dense:
    def __init__(self, batch_size, input_shape, num_neurons, activation, frozen=False, batch_norm=False):
        self.frozen = frozen
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.activation = activation
        self.num_neurons = num_neurons
        self.W = np.random.randn(self.input_size, self.num_neurons) * np.sqrt(1 / self.input_size)
        self.B = np.zeros((num_neurons))

        self.Z = np.zeros((batch_size, num_neurons))
        self.A = np.zeros_like(self.Z)
    
        if self.batch_norm:
            self.Z_normalized = np.zeros_like(self.Z)
            self.Y = np.zeros_like(self.Z)
            self.scale = np.ones((num_neurons,))
            self.shift = np.zeros((num_neurons,))
            self.scale_grad = np.zeros_like(self.scale)
            self.shift_grad = np.zeros_like(self.shift)
            self.running_mean = np.zeros((num_neurons,))
            self.running_var = np.ones((num_neurons,))
            self.momentum = 0.99

    def forward(self, input):
        self.X_flattened = input.reshape(self.batch_size, -1)
        self.Z = self.X_flattened.dot(self.W) + self.B

        if self.batch_norm:
            if not self.frozen:
                # Compute mean and variance per feature
                self.batch_mean = np.mean(self.Z, axis=0)
                self.batch_variance = np.var(self.Z, axis=0)
                # Normalize Z
                self.Z_normalized = (self.Z - self.batch_mean) / np.sqrt(self.batch_variance + 1e-3)
                # Scale and shift
                self.Y = self.Z_normalized * self.scale + self.shift
                # Update running estimates
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_variance
                pre_activation = self.Y
            else:
                # Use running mean and variance for normalization
                Z_normalized = (self.Z - self.running_mean) / np.sqrt(self.running_var + 1e-3)
                Y = Z_normalized * self.scale + self.shift
                pre_activation = Y
        else:
            pre_activation = self.Z

        # Activation
        match self.activation.lower():
            case 'linear':
                self.A = pre_activation
                self.dA_dZ = np.ones_like(pre_activation)
            case 'relu':
                self.A = np.maximum(0, pre_activation)
                self.dA_dZ = np.where(pre_activation > 0, 1, 0)
            case 'lrelu' | 'leaky_relu':
                self.A = np.where(pre_activation > 0, pre_activation, pre_activation * LRELU_ALPHA)
                self.dA_dZ = np.where(pre_activation > 0, 1, LRELU_ALPHA)
            case 'sigmoid':
                sigmoid = 1 / (1 + np.exp(-pre_activation))
                self.A = sigmoid
                self.dA_dZ = sigmoid * (1 - sigmoid)
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A

    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        if not np.shape(gradient) == 2:
            gradient = gradient.reshape((self.batch_size, -1))
        
        dA_dZ = self.dA_dZ
        dL_dZ = gradient * dA_dZ

        if self.batch_norm:
            dL_dscale = np.sum(dL_dZ * self.Z_normalized, axis=0)
            dL_dshift = np.sum(dL_dZ, axis=0)
            
            if not self.frozen:
                self.scale_grad += dL_dscale
                self.shift_grad += dL_dshift

            dL_dZ_normalized = dL_dZ * self.scale
            dL_dvar = np.sum(dL_dZ_normalized * (self.Z - self.batch_mean) * -0.5 * (self.batch_variance + 1e-3)**(-1.5), axis=0)
            dL_dmean = np.sum(dL_dZ_normalized * -1 / np.sqrt(self.batch_variance + 1e-3), axis=0) + \
                       dL_dvar * np.mean(-2 * (self.Z - self.batch_mean), axis=0)
            dL_dZ = (dL_dZ_normalized / np.sqrt(self.batch_variance + 1e-3)) + \
                    (dL_dvar * 2 * (self.Z - self.batch_mean) / self.batch_size) + \
                    (dL_dmean / self.batch_size)

        dL_dW = self.X_flattened.T.dot(dL_dZ)
        dL_dB = np.sum(dL_dZ, axis=0)

        if not self.frozen:
            W_delta_accumulated += dL_dW
            B_delta_accumulated += dL_dB

        updated_gradient = dL_dZ.dot(self.W.T)

        return updated_gradient.reshape((self.batch_size,) + self.input_shape)
