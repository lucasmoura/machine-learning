import numpy as np
import random

from model.utils import (sigmoid, sigmoid_derivative, mse, mse_derivative,
                         create_empty_copy_array)


class MultiLayerPerceptron:

    """
    Create a  MultiLayer perceptron network.

    :param layer:       An array containing the amount of neurons each layer in
                        the network will have, including both input and output
                        layer.
    """
    def __init__(self, layers, activation_function=sigmoid, cost_function=mse):
        self.num_layers = len(layers)

        """
        The input layer should not have any bias values associated with it.
        Also, the biases will be column vectors.
        """
        self.biases = [np.random.randn(layer, 1) for layer in layers[1:]]

        """
        Create the weight for each hidden layer in the network
        """
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layers[:-1], layers[1:])]

        self.activation_function = activation_function
        self.cost_function = cost_function

    def feedForward(self, input_data, intermediate_values=False):
        z_values, a_values = [], []
        a = input_data
        a_values.append(a)
        activation_function = self.activation_function

        for weight, bias in zip(self.weights, self.biases):
            z = weight.T.dot(a) + bias
            z_values.append(z)
            a = activation_function(z)
            a_values.append(a)

        if intermediate_values:
            return (a, a_values, z_values)

        return a

    def backpropagation(self, input_data, prediction):
        activation_derivative = eval(
            self.activation_function.__name__ + '_derivative')
        cost_derivative = eval(
            self.cost_function.__name__ + '_derivative')

        update_biases = create_empty_copy_array(self.biases)
        update_weights = create_empty_copy_array(self.weights)

        a, a_values, z_values = self.feedForward(
            input_data, intermediate_values=True)

        error = (cost_derivative(a_values[-1], prediction) *
                 activation_derivative(z_values[-1]))

        update_biases[-1] = error
        update_weights[-1] = a_values[-2].dot(error.T)

        for layer in range(2, self.num_layers):
            error = (self.weights[-layer + 1].dot(error) *
                     activation_derivative(z_values[-layer]))

            update_biases[-layer] = error
            update_weights[-layer] = a_values[-layer-1].dot(error.T)

        return (update_biases, update_weights)

    def sgd(self, training_data, batch_size, epochs,
            learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)

        num_data = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+batch_size]
                for k in range(0, num_data, batch_size)
            ]

            for batch in mini_batches:
                self.update_batch(batch, learning_rate)

            if test_data:
                print("Epoch {}: {} / {}".format(
                    epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(epoch))
        
    def update_batch(self, batch, learning_rate):
        update_biases = create_empty_copy_array(self.biases)
        update_weights = create_empty_copy_array(self.weights)
        size = len(batch)

        for data, prediction in batch:
            update_biases_n, update_weights_n = self.backpropagation(
                data, prediction)

            update_biases = [n + o for n, o in zip(update_biases_n,
                                                   update_biases)]

            update_weights = [n + o for n, o in zip(update_weights_n,
                                                    update_weights)]

        self.weights = [w - (learning_rate / size) * grad_w
                        for w, grad_w in zip(self.weights, update_weights)]
        self.biases = [b - (learning_rate / size) * grad_b
                       for b, grad_b in zip(self.biases, update_biases)]

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedForward(x)), y)
                       for x, y in test_data]
        return sum(int(x == y) for x, y in test_result)
