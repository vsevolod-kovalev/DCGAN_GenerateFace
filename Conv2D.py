import numpy as np

LRELU_ALPHA = 0.2
class Conv2d:
    def __init__(self, input, num_filters, kernel_size, stride, padding, activation):
        self.activation = activation
        self.num_filters = num_filters
        self.padding = padding
        self.input_height, self.input_width, self.depth = np.shape(input)
        self.kernel_size, self.stride = kernel_size, stride
        self.W = np.random.uniform(-0.1, 0.1, (num_filters, kernel_size, kernel_size, self.depth))
        self.B = np.zeros((num_filters))
        self.feature_map_width = (self.input_width + 2 * padding - kernel_size) / stride + 1
        self.feature_map_height = (self.input_height + 2 * padding - kernel_size) / stride + 1
        if not (np.modf(self.feature_map_width)[0] == 0.0 and np.modf(self.feature_map_width)[0] == 0.0):
            raise Exception("Error. Feature map size must be a integer.")
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
                region = padded_input[i : i + self.kernel_size, j : j + self.kernel_size]
                for k in range(self.num_filters):
                    filter = self.W[k]
                    self.Z[i, j, k] = np.sum(filter * region)
        match self.activation.lower():
            case 'relu':
                self.A = np.maximum(0, self.Z)
            case 'lrelu' | 'leaky_relu':
                self.A = np.where(self.Z > 0, self.Z, self.Z * LRELU_ALPHA)
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