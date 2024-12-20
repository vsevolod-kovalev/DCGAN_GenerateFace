import numpy as np
from constants import *

class TConv2D:
    def __init__(self, batch_size, input_shape, num_filters, kernel_size, stride, padding, activation, frozen=False, batch_norm=False):
        self.frozen = frozen
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_filters = num_filters
        self.padding = padding
        self.batch_size = batch_size
        self.input_height, self.input_width, self.depth = input_shape
        self.kernel_size, self.stride = kernel_size, stride
        self.feature_map_width = (self.input_width - 1) * self.stride + self.kernel_size - 2 * self.padding
        self.feature_map_height = (self.input_height - 1) * self.stride + self.kernel_size - 2 * self.padding
        
        if not (np.modf(self.feature_map_width)[0] == 0.0 and np.modf(self.feature_map_height)[0] == 0.0):
            raise Exception("Error. Feature map size must be a integer.")
        
        self.feature_map_width, self.feature_map_height = int(self.feature_map_width), int(self.feature_map_height)
        
        std_dev = np.sqrt(2 / (self.kernel_size * self.kernel_size * self.depth))
        self.W = np.random.normal(0, std_dev, (num_filters, kernel_size, kernel_size, self.depth))
        self.B = np.zeros((num_filters,))
        self.Z = np.zeros((batch_size, self.feature_map_height, self.feature_map_width, num_filters))
        self.Z_unpadded = np.zeros((batch_size, self.feature_map_height + 2 * self.padding, self.feature_map_width + 2 * self.padding, num_filters))
        self.A = np.zeros_like(self.Z)
        
        if self.padding > 0:
            self.dL_dZ_unpadded = np.zeros_like(self.Z_unpadded)
        else:
            self.dL_dZ_unpadded = None
        
        if self.batch_norm:
            self.scale = np.ones((num_filters,))
            self.shift = np.zeros((num_filters,))
            self.scale_grad = np.zeros_like(self.scale)
            self.shift_grad = np.zeros_like(self.shift)
            self.running_mean = np.zeros((num_filters,))
            self.running_var = np.ones((num_filters,))
            self.momentum = 0.99
            self.epsilon = 1e-3

    def forward(self, input):
        input = input.reshape((self.batch_size, self.input_height, self.input_width, self.depth))
        self.input = input
        self.Z_unpadded.fill(0)
        for i in range(self.input_height):
            for j in range(self.input_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.kernel_size
                end_j = start_j + self.kernel_size
                input_pixel = self.input[:, i, j, :].reshape((self.batch_size, 1, 1, self.depth))
                for k in range(self.num_filters):
                    self.Z_unpadded[:, start_i:end_i, start_j:end_j, k] += np.sum(
                        input_pixel * self.W[k], axis=-1
                    )
        if self.padding > 0:
            self.Z = self.Z_unpadded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            self.Z = self.Z_unpadded.copy()
        
        self.Z += self.B[np.newaxis, np.newaxis, np.newaxis, :]
        
        if self.batch_norm:
            if not self.frozen:
                self.batch_mean = np.mean(self.Z, axis=(0, 1, 2), keepdims=True)
                self.batch_variance = np.var(self.Z, axis=(0, 1, 2), keepdims=True)
                
                self.Z_normalized = (self.Z - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)
                
                self.Y = self.Z_normalized * self.scale + self.shift
                
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean.squeeze()
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_variance.squeeze()
                
                pre_activation = self.Y
            else:
                Z_normalized = (self.Z - self.running_mean.reshape(1, 1, 1, -1)) / np.sqrt(self.running_var.reshape(1, 1, 1, -1) + self.epsilon)
                Y = Z_normalized * self.scale + self.shift
                pre_activation = Y
        else:
            pre_activation = self.Z
        
        activation = self.activation.lower()
        if activation == 'relu':
            self.A = np.maximum(0, pre_activation)
            self.dA_dZ = np.where(pre_activation > 0, 1, 0)
        elif activation in ['lrelu', 'leaky_relu']:
            self.A = np.where(pre_activation > 0, pre_activation, pre_activation * LRELU_ALPHA)
            self.dA_dZ = np.where(pre_activation > 0, 1, LRELU_ALPHA)
        elif activation == 'tanh':
            self.A = np.tanh(pre_activation)
            self.dA_dZ = 1 - np.power(self.A, 2)
        else:
            raise Exception("Error. Unknown activation function.")
        
        return self.A

    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        if not np.ndim(gradient) == 4:
            raise Exception("Gradient must have 4 dimensions (batch_size, height, width, num_filters)")
        dL_dZ_final = gradient * self.dA_dZ
        if self.batch_norm:
            dscale = np.sum(dL_dZ_final * self.Z_normalized, axis=(0, 1, 2))
            dshift = np.sum(dL_dZ_final, axis=(0, 1, 2))
            
            if not self.frozen:
                self.scale_grad += dscale
                self.shift_grad += dshift
            
            dZ_normalized = dL_dZ_final * self.scale
            dvariance = np.sum(dZ_normalized * (self.Z - self.batch_mean) * -0.5 * np.power(self.batch_variance + self.epsilon, -1.5), axis=(0,1,2), keepdims=True)
            dmean = (np.sum(dZ_normalized * -1 / np.sqrt(self.batch_variance + self.epsilon), axis=(0,1,2), keepdims=True) +
                     dvariance * np.mean(-2 * (self.Z - self.batch_mean), axis=(0,1,2), keepdims=True))
            N = self.batch_size * self.feature_map_height * self.feature_map_width
            dZ = (dZ_normalized / np.sqrt(self.batch_variance + self.epsilon)) + \
                 (dvariance * 2 * (self.Z - self.batch_mean) / N) + \
                 (dmean / N)
            
            dL_dZ_final = dZ
        
        if self.padding > 0:
            self.dL_dZ_unpadded.fill(0)
            self.dL_dZ_unpadded[:, self.padding:-self.padding, self.padding:-self.padding, :] = dL_dZ_final
            dL_dZ_unpadded = self.dL_dZ_unpadded
        else:
            dL_dZ_unpadded = dL_dZ_final    
        if not self.frozen:
            B_delta_accumulated += np.sum(dL_dZ_unpadded, axis=(0,1,2))
    
        dX = np.zeros_like(self.input)
        
        for i in range(self.input_height):
            for j in range(self.input_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.kernel_size
                end_j = start_j + self.kernel_size
                for k in range(self.num_filters):
                    dZ_region = dL_dZ_unpadded[:, start_i:end_i, start_j:end_j, k]
                    
                    if not self.frozen:
                        input_pixel = self.input[:, i, j, :]
                        input_pixel_expanded = input_pixel[:, np.newaxis, np.newaxis, :]
                        
                        dW = np.sum(dZ_region[:, :, :, np.newaxis] * input_pixel_expanded, axis=(0, 1, 2))
                        W_delta_accumulated[k] += dW
                    
                    dX[:, i, j, :] += np.sum(dZ_region[:, :, :, np.newaxis] * self.W[k], axis=(1, 2))
        
        return dX
