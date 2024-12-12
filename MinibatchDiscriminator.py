import numpy as np

class MinibatchDiscriminator:
    def __init__(self, input_shape, num_features, batch_size):
        self.input_shape = input_shape
        self.num_features = num_features
        self.batch_size = batch_size

        self.W = np.random.normal(
            0,
            np.sqrt(2 / np.prod(input_shape)),
            (np.prod(input_shape), num_features)
        )
        self.B = np.zeros((num_features,))

    def forward(self, input):
        self.input_flat = input.reshape(self.batch_size, -1)
        self.A = self.input_flat @ self.W + self.B

        A_expanded = self.A[:, None, :]
        pairwise_diff = np.abs(A_expanded - self.A[None, :, :])
        pairwise_distances = np.sum(pairwise_diff, axis=2)

        minibatch_features = np.sum(np.exp(-pairwise_distances), axis=1)

        self.output = self.A + minibatch_features[:, None] 
        return self.output

    def backward(self, gradient, W_delta_accumulated, B_delta_accumulated):
        dA = gradient
        W_delta_accumulated += self.input_flat.T @ dA
        B_delta_accumulated += np.sum(dA, axis=0)

        dX_flat = dA @ self.W.T
        dX = dX_flat.reshape((self.batch_size, *self.input_shape))
        return dX
