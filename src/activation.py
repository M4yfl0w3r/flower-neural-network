import numpy as np


class ReLU:
    
    def __init__(self):
        self.input: np.ndarray = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        output: np.ndarray = np.maximum(0, input)
        self.input = input
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        output: np.ndarray = np.where(self.input <= 0, 0, 1) * gradient
        return output


class Softmax:

    def __init__(self):
        self.output = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        exp_values: np.ndarray = np.exp(input.data)
        self.output = exp_values / np.sum(exp_values, axis = 0)
        return self.output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        softmax_gradient: list = []
        
        def grad_(input: np.ndarray) -> np.ndarray:
            s: np.ndarray = input.reshape(-1, 1)
            return np.diagflat(s) - np.dot(s, s.T)
        
        for output, incoming_grad in zip(self.output, gradient):
            softmax_gradient.append(np.dot(grad_(output), incoming_grad))

        return np.array(softmax_gradient)
