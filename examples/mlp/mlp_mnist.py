import numpy as np
import gzip
import cPickle as pickle

import config
from mlp.mlp import MultiLayerPerceptron

def create_data():
    with gzip.open(config.MNIST_DATA) as mnist_data:
        training_data, validation_data, test_data = pickle.load(mnist_data)
    
    return (training_data, validation_data, test_data)


def create_inputs(data, shape):
    return [np.reshape(x, shape) for x in data]


def one_hot(data, size):
    e = np.zeros((size, 1))
    e[data] = 1.0
    return e


def create_results(data, size):
    return [one_hot(x, size) for x in data]


def process_data(input_data, input_shape, result_data, result_size):
    processed_input = create_inputs(input_data, input_shape)
    processed_result = create_results(result_data, result_size)
    return zip(processed_input, processed_result)


def create_mnist_data():
    training_data, validation_data, test_data = create_data()

    training_data = process_data(
        training_data[0], (784, 1), training_data[1], 10)
    validation_data = process_data(
        validation_data[0], (784, 1), validation_data[1], 10)
    test_inputs = create_inputs(test_data[0], (784, 1))
    test_data = zip(test_inputs, test_data[1])

    return (training_data, validation_data, test_data)


def main():
    training_data, validation_data, test_data = create_mnist_data()
    layers = [784, 100, 10]

    mlp = MultiLayerPerceptron(layers)
    mlp.sgd(training_data, 10, 5, 3.0, test_data=test_data)


if __name__ == '__main__':
    main()
