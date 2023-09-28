import numpy as np

from layer import Layer 


def convert_to_hot_one(label: int, num_labels: int) -> np.ndarray:
    hot_one: np.ndarray = np.zeros(shape = num_labels)
    hot_one[label] = 1.0
    return hot_one

EPSILON = 1e-5

class CategoricalCrossEntropy(Layer):

    def forward(self, input: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.forward_input = np.clip(input, EPSILON, 1.0 - EPSILON)
        num_classes: int = len(np.unique(labels))
        labels_as_hot_one: list = [convert_to_hot_one(label, num_classes) for label in labels]
        self.forward_output = - np.array(labels_as_hot_one) * np.log(self.forward_input)
        return self.forward_output
    
    def backward(self, input: np.ndarray, labels: np.ndarray) -> np.ndarray:
        target_classes: list = [convert_to_hot_one(label, 3) for label in labels]
        target_classes: np.ndarray = np.array(target_classes)
        self.backward_output = - target_classes / input
        return self.backward_output / len(input)
