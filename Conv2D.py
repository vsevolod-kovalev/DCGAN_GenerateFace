import numpy as np

LRELU_ALPHA = 0.2
class Conv2d:
    def __init__(self, input, num_filters, kernel_size, stride, padding, activation):
        self.activation = activation
        self.num_filters = num_filters
        self.padding = padding
        self.input_height, self.input_width, self.depth = np.shape(input)
        self.kernel_size, self.stride = kernel_size, stride
        self.feature_map_width = (self.input_width + 2 * padding - kernel_size) / stride + 1
        self.feature_map_height = (self.input_height + 2 * padding - kernel_size) / stride + 1
        if not (np.modf(self.feature_map_width)[0] == 0.0 and np.modf(self.feature_map_height)[0] == 0.0):
            raise Exception("Error. Feature map size must be a integer.")
        self.W = np.random.uniform(-0.1, 0.1, (num_filters, kernel_size, kernel_size, self.depth))
        self.B = np.zeros((num_filters))
        self.feature_map_width, self.feature_map_height = int(self.feature_map_width), int(self.feature_map_height)
        self.Z = np.zeros((self.feature_map_height, self.feature_map_width, num_filters))
        self.A = np.zeros_like(self.Z)
    def forward(self, input):
        padded_input = np.pad(
            input,
            pad_width=((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode='constant',
            constant_values=0
        )
        for i in range(self.feature_map_height):
            for j in range(self.feature_map_width):
                region = padded_input[
                    i * self.stride : i * self.stride + self.kernel_size,
                    j * self.stride : j * self.stride + self.kernel_size,
                ]
                for k in range(self.num_filters):
                    self.Z[i, j, k] = np.sum(self.W[k] * region) + self.B[k]
        match self.activation.lower():
            case 'relu':
                self.A = np.maximum(0, self.Z)
            case 'lrelu' | 'leaky_relu':
                self.A = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA)
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A
class TConv2d:
    def __init__(self, input, num_filters, kernel_size, stride, padding, activation):
        self.activation = activation
        self.num_filters = num_filters
        self.padding = padding
        self.input_height, self.input_width, self.depth = np.shape(input)
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
        self.A = np.zeros_like(self.Z)
def forward(self, input):
    self.Z_unpadded[:] = 0
    for i in range(self.input_height):
        for j in range(self.input_width):
            start_i = i * self.stride
            start_j = j * self.stride
            end_i = start_i + self.kernel_size
            end_j = start_j + self.kernel_size
            input_pixel = input[i, j, :].reshape((1, 1, -1)) 
            for k in range(self.num_filters):
                self.Z_unpadded[start_i:end_i, start_j:end_j, k] += (
                    input_pixel * self.W[k]
                )
    if self.padding > 0:
        self.Z = self.Z_unpadded[self.padding:-self.padding, self.padding:-self.padding, :]
    else:
        self.Z = self.Z_unpadded

    for k in range(self.num_filters):
        self.Z[:, :, k] += self.B[k]

    activation = self.activation.lower()
    if activation == 'relu':
        self.A = np.maximum(0, self.Z)
    elif activation in ('lrelu', 'leaky_relu'):
        self.A = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA)
    elif activation == 'tanh':
        self.A = np.tanh(self.Z)
    else:
        raise Exception("Error. Unknown activation function.")
    
    return self.A

class Dense:
    def __init__(self, input, num_neurons, activation):
        self.activation = activation
        self.num_neurons = num_neurons
        self.flattened_input = np.ravel(input)
        self.W = np.random.uniform(-0.1, 0.1, (num_neurons, len(self.flattened_input)))
        self.B = np.zeros((num_neurons))
        self.Z = np.zeros((num_neurons))
        self.A = np.zeros_like(self.Z)
    def forward(self, input):
        self.flattened_input = np.ravel(input)
        self.Z = np.dot(self.W, self.flattened_input) + self.B
        match self.activation.lower():
            case 'relu':
                self.A = np.maximum(0, self.Z)
            case 'lrelu' | 'leaky_relu':
                self.A = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA)
            case 'sigmoid':
                self.A = 1 / (1 + np.exp(-self.Z))
            case _:
                raise Exception("Error. Unknown activation function.")
        return self.A
input = np.random.uniform(0, 1, (100, 100, 3))
x = Conv2d(
        input=input, 
        num_filters=64,
        kernel_size=4,
        stride=2,
        padding=1,
        activation='lrelu'
    )
output = x.forward(input)
print(output, np.shape(output))