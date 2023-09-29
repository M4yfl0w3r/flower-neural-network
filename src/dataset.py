import numpy as np

from sklearn import preprocessing
from sklearn.datasets import load_iris


def iris():
    iris = load_iris()
    data: np.ndarray = np.array(iris.data)
    labels: np.ndarray = np.array(iris.target)
    data = preprocessing.normalize(data)
    data = data / 10

    random_indices: np.ndarray = np.arange(len(data))
    np.random.shuffle(random_indices)
    data = data[random_indices]
    labels = labels[random_indices]

    return data, labels

def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype = 'uint8')

    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + \
            np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
        
    return X, y
