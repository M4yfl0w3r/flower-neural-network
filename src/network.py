import numpy as np


class DenseLayer:

    def __init__(self, num_inputs: int, num_neurons: int):
        self.weights: np.ndarray = np.random.rand(num_inputs, num_neurons)
        self.biases: np.ndarray = np.random.rand(1, num_neurons)

    def forward(self, input: np.ndarray) -> np.ndarray:
        output: np.ndarray = np.dot(input, self.weights) + self.biases
        return output