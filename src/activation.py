import numpy as np


class ReLU:

    def forward(self, input: np.ndarray) -> np.ndarray:
        output: np.ndarray = np.maximum(0, input)
        return output
    

class Softmax:

    def forward(self, input: np.ndarray) -> np.ndarray:
        exp_values: np.ndarray = np.exp(input.data)
        output: np.ndarray = exp_values / np.sum(exp_values, axis = 1)
        return output