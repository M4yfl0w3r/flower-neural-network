import numpy as np

from layer import Layer
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
        
        self.forward_output: np.ndarray = None
        self.loss: float = 0.0

    def forward(self, epoch: int):
        # TODO: Change to more generic
        output = self.layers[0].forward(self.params.input)
        output = self.layers[1].forward(output)
        output = self.layers[2].forward(output)
        output = self.layers[3].forward(output)
        output = self.layers[4].forward(output)
        output = self.layers[5].forward(output)
        self.loss = self.layers[6].forward(output, self.params.labels)

        self.forward_output = output
        loss: float = model_loss(self.loss)
        accuracy: float = model_accuracy(output, self.params.labels)
        print(f'Epoch: {epoch} | Loss: {loss:.3f} | Accuracy = {accuracy:.3f}')

    def backward(self):
        output = self.layers[6].backward(self.forward_output, self.params.labels)
        output = self.layers[5].backward(output)
        output = self.layers[4].backward(output)
        output = self.layers[3].backward(output)
        output = self.layers[2].backward(output)
        output = self.layers[1].backward(output)
        output = self.layers[0].backward(output)

    def update_params(self):
        if self.params.optimizer == 'SGD':
            self.layers[0].update_params(self.params.learning_rate)
            self.layers[2].update_params(self.params.learning_rate)
            self.layers[4].update_params(self.params.learning_rate)

    def train(self):
        for epoch in range(self.params.num_epochs):
            self.forward(epoch)
            self.backward()
            self.update_params()
