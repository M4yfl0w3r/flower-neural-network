import numpy as np

from layer import Layer
from layer import DenseLayer 
from utils import model_loss
from utils import model_accuracy 

class Params:
    input: np.ndarray = None
    labels: np.ndarray = None

    optimizer: str = 'SGD'
    
    num_epochs: int = 1000
    learning_rate: float = 1.0


class NeuralNetwork:

    def __init__(self, layers: list[Layer], params: Params):
        self.layers = layers 
        self.params = params
        self.activation_output = None

        self.loss: float = 0.0

    def forward(self):
        output: np.ndarray = self.layers[0].forward(self.params.input)

        for i in range(1, len(self.layers) - 1):
            output = self.layers[i].forward(output)
        
        self.activation_output = output 
        self.loss = self.layers[len(self.layers)].forward(output, self.params.labels)

    def evaluate(self, epoch: int):
        loss = model_loss(self.loss)
        accuracy = model_accuracy(self.activation_output, self.params.labels)
        print(f'Epoch: {epoch} | Loss: {loss:.3f} | Accuracy = {accuracy:.3f}')

    def backward(self):

        output = loss.backward(output, labels)
        output = softmax.backward(output)
        output = layer_3.backward(output)
        output = relu_2.backward(output)
        output = layer_2.backward(output)
        output = relu_1.backward(output)
        output = layer_1.backward(output)

    def update_params(self):
        pass

    def train(self):
        pass