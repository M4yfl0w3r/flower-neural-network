import numpy as np

from layer import Layer


class ReLU(Layer):

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.forward_output = np.maximum(0, input)
        self.forward_input = input
        return self.forward_output 

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.backward_output = np.where(self.forward_input <= 0, 0, 1) * gradient
        return self.backward_output


class Softmax(Layer):

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.forward_input = input
        exp_values: np.ndarray = np.exp(input)
        self.forward_output = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        return self.forward_output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        softmax_gradient: list = []
        
        def grad_(input: np.ndarray) -> np.ndarray:
            s: np.ndarray = input.reshape(-1, 1)
            return np.diagflat(s) - np.dot(s, s.T)
        
        for output, incoming_grad in zip(self.forward_output, gradient):
            softmax_gradient.append(np.dot(grad_(output), incoming_grad))

        self.backward_output = np.array(softmax_gradient)
        return self.backward_output