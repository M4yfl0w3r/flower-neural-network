import numpy as np

from layer import DenseLayer
from activation import ReLU
from activation import Softmax
from loss import CategoricalCrossEntropy
from utils import model_accuracy, model_loss
from dataset import iris, spiral_data
from network import NeuralNetwork
from network import Params


LEARNING_RATE = 0.3
NUM_EPOCHS = 1000 

# input, labels = iris()
X, y = spiral_data(150, 3)
input = X
labels = y

layer_1 = DenseLayer(num_inputs = 2, num_neurons = 64)
layer_2 = DenseLayer(num_inputs = 64, num_neurons = 32)
layer_3 = DenseLayer(num_inputs = 32, num_neurons = 3)
relu_1 = ReLU()
relu_2 = ReLU()
softmax = Softmax()
loss = CategoricalCrossEntropy()

for epoch in range(NUM_EPOCHS):
    output = layer_1.forward(input)
    output = relu_1.forward(output)
    output = layer_2.forward(output)
    output = relu_2.forward(output)
    output = layer_3.forward(output)
    output = softmax.forward(output)
    loss_  = loss.forward(output, labels)

    loss_ = model_loss(loss_)
    accuracy = model_accuracy(output, labels)
    print(f'Epoch: {epoch} | Loss: {loss_:.3f} | Accuracy = {accuracy:.3f}')

    output = loss.backward(output, labels)
    output = softmax.backward(output)
    output = layer_3.backward(output)
    output = relu_2.backward(output)
    output = layer_2.backward(output)
    output = relu_1.backward(output)
    output = layer_1.backward(output)

    layer_1.update_params(LEARNING_RATE)
    layer_2.update_params(LEARNING_RATE)
    layer_3.update_params(LEARNING_RATE)


layers = [
    DenseLayer(num_inputs = 2, num_neurons = 64),
    ReLU(),
    DenseLayer(num_inputs = 64, num_neurons = 32),
    ReLU(),
    DenseLayer(num_inputs = 32, num_neurons = 3),
    Softmax(),
    CategoricalCrossEntropy()
]

params = Params()
params.input = input
params.labels = labels

net = NeuralNetwork(layers, params)
net.train()