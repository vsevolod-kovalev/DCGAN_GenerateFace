import numpy as np

class Dropout:
    def __init__(self, dropout):
        self.dropout = dropout
    def forward(self, input):
        self.mask = np.random.binomial(1, 1 - self.dropout, np.shape(input))
        return input * self.mask / (1 - self.dropout)
    def backward(self, gradient):
        return self.mask * gradient

