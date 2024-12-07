import numpy as np
from constants import *

class TConv2D:
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, activation):
        self.activation = activation
        self.num_filters = num_filters
        self.padding = padding
        self.input_height, self.input_width, self.depth = input_shape
        self.kernel_size, self.stride = kernel_size, stride
        self.feature_map_width = (self.input_width - 1) * self.stride + self.kernel_size - 2 * self.padding
        self.feature_map_height = (self.input_height - 1) * self.stride + self.kernel_size - 2 * self.padding
        if not (np.modf(self.feature_map_width)[0] == 0.0 and np.modf(self.feature_map_height)[0] == 0.0):
            raise Exception("Error. Feature map size must be a integer.")
        self.W = np.random.uniform(-0.1, 0.1, (num_filters, kernel_size, kernel_size, self.depth))
        self.B = np.zeros((num_filters))
        self.feature_map_width, self.feature_map_height = int(self.feature_map_width), int(self.feature_map_height)
        self.Z = np.zeros((self.feature_map_height, self.feature_map_width, num_filters))
        self.Z_unpadded = np.zeros((self.feature_map_height + 2 * self.padding, self.feature_map_width + 2 * self.padding, num_filters))
        if self.padding > 0:
            self.dL_dZ_unpadded = np.zeros_like(self.Z_unpadded)
        else:
            self.dL_dZ_unpadded = None
        self.A = np.zeros_like(self.Z)
    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        dL_dZ = gradient * self.dA_dZ
        if self.padding > 0:
            self.dL_dZ_unpadded.fill(0)
            self.dL_dZ_unpadded[self.padding:-self.padding, self.padding:-self.padding, :] = dL_dZ
            dL_dZ_unpadded = self.dL_dZ_unpadded
        else:
            dL_dZ_unpadded = dL_dZ

        for k in range(self.num_filters):
            B_delta_accumulated[k] += np.sum(dL_dZ_unpadded[:, :, k])

        dX = np.zeros_like(self.input)
        for i in range(self.input_height):
            for j in range(self.input_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.kernel_size
                end_j = start_j + self.kernel_size
                input_pixel = self.input[i, j, :]
                for k in range(self.num_filters):
                    dZ_region = dL_dZ_unpadded[start_i:end_i, start_j:end_j, k]
                    W_delta_accumulated[k] += dZ_region[..., np.newaxis] * input_pixel[np.newaxis, np.newaxis, :]
                    dX[i, j, :] += np.sum(dZ_region[..., np.newaxis] * self.W[k], axis=(0, 1))
        return dX

    def forward(self, input):
        input = input.reshape((self.input_height, self.input_width, self.depth))
        self.input = input  
        self.Z_unpadded[:] = 0
        for i in range(self.input_height):
            for j in range(self.input_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.kernel_size
                end_j = start_j + self.kernel_size
                input_pixel = input[i, j, :].reshape((1, 1, -1)) 
                for k in range(self.num_filters):
                    self.Z_unpadded[start_i:end_i, start_j:end_j, k] += np.sum(
                        input_pixel * self.W[k],
                        axis=-1
                    )
        if self.padding > 0:
            self.Z = self.Z_unpadded[self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            self.Z = self.Z_unpadded
        for k in range(self.num_filters):
            self.Z[:, :, k] += self.B[k]
        match self.activation.lower():
            case 'relu':
                self.A, self.dA_dZ = np.maximum(0, self.Z), np.where(self.Z > 0, 1, 0)
            case 'lrelu' | 'leaky_relu':
                self.A, self.dA_dZ = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA), np.where(self.Z > 0, 1, LRELU_ALPHA)
            case 'tanh':
                self.A, self.dA_dZ = np.tanh(self.Z), 1 - np.power(np.tanh(self.Z), 2)
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A