import numpy as np
from TConv2D import TConv2D
from Dense import Dense

class Generator:
    def __init__(self, batch_size):
        self.layers = [
            Dense(batch_size=batch_size, input_shape=(1, 1, 100), num_neurons=4 * 4 * 128, activation="relu"),
            TConv2D(batch_size=batch_size, input_shape=(4, 4, 128), num_filters=64, kernel_size=4, stride=2, padding=1, activation="relu"),
            TConv2D(batch_size=batch_size, input_shape=(8, 8, 64), num_filters=32, kernel_size=4, stride=2, padding=1, activation="relu"),
            TConv2D(batch_size=batch_size, input_shape=(16, 16, 32), num_filters=16, kernel_size=4, stride=2, padding=1, activation="relu"),
            TConv2D(batch_size=batch_size, input_shape=(32, 32, 16), num_filters=8, kernel_size=4, stride=2, padding=1, activation="relu"),
            TConv2D(batch_size=batch_size, input_shape=(64, 64, 8), num_filters=3, kernel_size=3, stride=1, padding=1, activation="tanh")
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