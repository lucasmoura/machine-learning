import unittest

from mlp.mlp import MultiLayerPerceptron
from numpy import array


class MultiLayerPerceptronTest(unittest.TestCase):

    def setUp(self):
        self.mlp = MultiLayerPerceptron([1, 2, 3])

    def test_bias_initial_values(self):
        biases = self.mlp.biases
        expectedLen = 2

        self.assertEqual(len(biases), expectedLen)
        expectedShape = [(2, 1), (3, 1)]

        for index, bias in enumerate(biases):
            self.assertEqual(bias.shape, expectedShape[index])

    def test_weight_initial_values(self):
        weights = self.mlp.weights
        expectedLen = 2

        self.assertEqual(len(weights), expectedLen)
        expectedShape = [(1, 2), (2, 3)]

        for index, weight in enumerate(weights):
            self.assertEqual(weight.shape, expectedShape[index])

    def test_feed_forward(self):
        self.mlp = MultiLayerPerceptron([1, 2, 3],
                                        activation_function=lambda x: x)
        input_data = array([[1]])

        self.mlp.weights = [array([[1, 2]]), array([[1, 2, 3], [4, 5, 6]])]
        self.mlp.biases = [array([[1], [1]]), array([[1], [2], [1]])]

        expectedValues = array([[15], [21], [25]])
        actualValues = self.mlp.feedForward(input_data)

        self.assertEqual(expectedValues.shape, actualValues.shape)

        for i in range(actualValues.shape[0]):
            self.assertEqual(expectedValues[i, 0], actualValues[i, 0])
