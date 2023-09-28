import numpy as np


class Layer:
  
    def __init__(self):
        self.forward_input: np.ndarray = None
        self.forward_output: np.ndarray = None
        self.backward_input: np.ndarray = None
        self.backward_output: np.ndarray = None

    def forward(self, *args, **kwargs): 
        raise NotImplementedError()
  
    def backward(self, *args, **kwargs): 
        raise NotImplementedError()
 

class DenseLayer(Layer):

    def __init__(self, num_inputs: int, num_neurons: int):
        self.weights: np.ndarray = np.random.rand(num_inputs, num_neurons) - 0.5
        self.biases: np.ndarray = np.random.rand(1, num_neurons) - 0.5
        
        self.weights_gradient: np.ndarray = np.empty(shape = (num_inputs, num_neurons))
        self.biases_gradient: np.ndarray = np.empty(shape = (1, num_neurons)) 

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.forward_input = input
        self.forward_output = np.dot(input, self.weights) + self.biases
        return self.forward_output
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        self.weights_gradient = np.dot(np.transpose(self.forward_input), gradient)
        self.biases_gradient = np.sum(gradient, axis = 0, keepdims = True)
        self.backward_output = np.dot(gradient, np.transpose(self.weights))
        return self.backward_output
        
    def update_params(self, learning_rate: int):
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient