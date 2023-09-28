from layer import DenseLayer
from activation import ReLU
from activation import Softmax
from loss import CategoricalCrossEntropy
from dataset import iris, spiral_data
from network import NeuralNetwork
from network import Params


X, y = spiral_data(150, 3)
input = X
labels = y

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
