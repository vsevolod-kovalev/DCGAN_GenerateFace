import numpy as np
from constants import *

class Conv2D:
    def __init__(self, batch_size, input_shape, num_filters, kernel_size, stride, padding, activation):
        self.activation = activation
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.padding = padding
        self.input_height, self.input_width, self.depth = input_shape
        self.kernel_size, self.stride = kernel_size, stride
        self.feature_map_width = (self.input_width + 2 * padding - kernel_size) / stride + 1
        self.feature_map_height = (self.input_height + 2 * padding - kernel_size) / stride + 1
        if not (np.modf(self.feature_map_width)[0] == 0.0 and np.modf(self.feature_map_height)[0] == 0.0):
            raise Exception("Error. Feature map size must be a integer.")
        # self.W = np.random.uniform(-0.1, 0.1, (num_filters, kernel_size, kernel_size, self.depth))
        std_dev = np.sqrt(2 / (self.kernel_size * self.kernel_size * self.depth))
        self.W = np.random.normal(0, std_dev, (num_filters, kernel_size, kernel_size, self.depth))

        self.B = np.zeros((num_filters))
        self.feature_map_width, self.feature_map_height = int(self.feature_map_width), int(self.feature_map_height)
        self.Z = np.zeros((batch_size, self.feature_map_height, self.feature_map_width, num_filters))
        self.A = np.zeros_like(self.Z)
    def forward(self, input):
        self.padded_input = np.pad(
            input,
            pad_width=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant',
            constant_values=0
        )
        for k in range(self.num_filters):
            filter = self.W[k]
            for i in range(self.feature_map_height):
                for j in range(self.feature_map_width):
                    region = self.padded_input[
                        :,
                        i * self.stride : i * self.stride + self.kernel_size,
                        j * self.stride : j * self.stride + self.kernel_size,
                        :
                    ]
                    self.Z[:, i, j, k] = np.sum(region * filter, axis=(1, 2, 3))
            self.Z[:, :, :, k] += self.B[k]

        match self.activation.lower():
            case 'relu':
                self.A = np.maximum(0, self.Z)
                self.dA_dZ = np.where(self.Z > 0, 1, 0)
            case 'lrelu' | 'leaky_relu':
                self.A = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA)
                self.dA_dZ = np.where(self.Z > 0, 1, LRELU_ALPHA)
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A

    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        dL_dZ = gradient * self.dA_dZ
        dX = np.zeros_like(self.padded_input)
        for k in range(self.num_filters):
            for i in range(self.feature_map_height):
                for j in range(self.feature_map_width):
                    region = self.padded_input[
                        :,
                        i * self.stride : i * self.stride + self.kernel_size,
                        j * self.stride : j * self.stride + self.kernel_size,
                        :
                    ]

                    W_delta_accumulated[k] += np.sum(dL_dZ[:, i, j, k][:, None, None, None] * region, axis=0)
                    dX[
                        :,
                        i * self.stride : i * self.stride + self.kernel_size,
                        j * self.stride : j * self.stride + self.kernel_size,
                        :
                    ] += dL_dZ[:, i, j, k][:, None, None, None] * self.W[k]

            B_delta_accumulated[k] += np.sum(dL_dZ[:, :, :, k])
        if self.padding > 0:
            updated_gradient = dX[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            updated_gradient = dX
        return updated_gradient
