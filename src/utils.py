import numpy as np


def model_accuracy(input: np.ndarray, target_labels: np.ndarray) -> float:
    predicted_labels: np.ndarray = np.argmax(input, axis = 1)
    output: float = np.sum(predicted_labels == target_labels) / len(input)
    return output

def model_loss(input: np.ndarray) -> float:
    return np.mean(input)
