import numpy as np


class DenseLayer:

    def __init__(self, num_inputs: int, num_neurons: int):
        self.weights: np.ndarray = np.random.rand(num_inputs, num_neurons)
        self.biases: np.ndarray = np.random.rand(1, num_neurons)
        
        self.input: np.ndarray = None
        self.output: np.ndarray = None

        self.weights_gradient: np.ndarray = None
        self.biases_gradient: np.ndarray = None
        self.input_gradient: np.ndarray = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.output = np.dot(input, self.weights) + self.biases
        self.input = input
        return self.output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weights_gradient = np.dot(np.transpose(self.input), gradient)
        self.biases_gradient = np.sum(1 * gradient, axis = 0)
        self.input_gradient = np.dot(gradient, np.transpose(self.weights))
        return self.input_gradient