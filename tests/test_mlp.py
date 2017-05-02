import unittest

from mlp.mlp import MultiLayerPerceptron


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
