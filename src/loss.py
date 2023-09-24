import numpy as np


def convert_to_hot_one(label: int, num_labels: int) -> np.ndarray:
    hot_one: np.ndarray = np.zeros(shape = num_labels)
    hot_one[label] = 1.0
    return hot_one

class CategoricalCrossEntropy:

    def forward(self, input: np.ndarray, labels: np.ndarray) -> np.ndarray:
        label_as_hot_one = convert_to_hot_one(labels, 3)
        output = - label_as_hot_one * np.log(input) 
        return output