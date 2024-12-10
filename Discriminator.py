import numpy as np
from Conv2D import Conv2D
from Dropout import Dropout
from Dense import Dense

class Discriminator:
    def __init__(self, batch_size):
        self.layers = [
            Conv2D(batch_size=batch_size, input_shape=(32, 32, 3), num_filters=32, kernel_size=4, stride=2, padding=1, activation="lrelu"),
            Conv2D(batch_size=batch_size, input_shape=(16, 16, 32), num_filters=64, kernel_size=4, stride=2, padding=1, activation="lrelu"),
            Conv2D(batch_size=batch_size, input_shape=(8, 8, 64), num_filters=128, kernel_size=4, stride=2, padding=1, activation="lrelu"),
            Conv2D(batch_size=batch_size, input_shape=(4, 4, 128), num_filters=256, kernel_size=4, stride=2, padding=1, activation="lrelu"),
            Dropout(dropout=0.3),
            Dense(batch_size=batch_size, input_shape=(2, 2, 256), num_neurons=1, activation="sigmoid")
        ]
        self.W_deltas = [np.zeros_like(layer.W) if hasattr(layer, 'W') else None for layer in self.layers]
        self.B_deltas = [np.zeros_like(layer.B) if hasattr(layer, 'B') else None for layer in self.layers]


    def applyDeltas(self, learning_rate=0.002):
        for layer_index in range(len(self.layers)):
            if hasattr(self.layers[layer_index], 'W'):
                self.layers[layer_index].W -= learning_rate * self.W_deltas[layer_index]
                self.layers[layer_index].B -= learning_rate * self.B_deltas[layer_index]
    def resetDeltas(self):
        for layer_index in range(len(self.layers)):
            if hasattr(self.layers[layer_index], 'W'):
                self.W_deltas[layer_index].fill(0)
                self.B_deltas[layer_index].fill(0)
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, gradient):
        current_gradient = gradient
        for layer_index in range(len(self.layers) -1, -1, -1):
            current_gradient = self.layers[layer_index].backward(current_gradient, self.W_deltas[layer_index], self.B_deltas[layer_index])
        return current_gradient

    def state_dict(self):
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(self.layers[i], 'W'):
                state[f"layer_{i}_W"] = layer.W
                state[f"layer_{i}_B"] = layer.B
        return state

    def load_state_dict(self, state):
        for i, layer in enumerate(self.layers):
            if hasattr(self.layers[i], 'W'):
                layer.W = state[f"layer_{i}_W"]
                layer.B = state[f"layer_{i}_B"]