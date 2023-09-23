import numpy as np


class ReLU:

    def forward(self, input: np.ndarray):
        output: np.ndarray = np.maximum(0, input)
        return output