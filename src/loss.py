import numpy as np

from layer import Layer 


def convert_to_hot_one(label: int, num_labels: int) -> np.ndarray:
    hot_one: np.ndarray = np.zeros(shape = num_labels)
    hot_one[label] = 1.0
    return hot_one

EPSILON = 1e-5

class CategoricalCrossEntropy(Layer):

    def __init__(self):
        self.target_classes: list = []

    def forward(self, input: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.forward_input = np.clip(input, EPSILON, 1.0 - EPSILON)
        num_classes: int = len(np.unique(labels))
        self.target_classes = np.array([convert_to_hot_one(label, num_classes) for label in labels])
        self.forward_output = - self.target_classes * np.log(self.forward_input)
        return self.forward_output
    
    def backward(self, input: np.ndarray) -> np.ndarray:
        self.backward_output = - self.target_classes / input
        return self.backward_output / len(input)
