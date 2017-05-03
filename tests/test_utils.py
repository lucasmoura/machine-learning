import unittest
from model.utils import (featureNormalization, sigmoid, sigmoid_derivative,
                         mse, create_empty_copy_array)
from math import floor, sqrt
from numpy import array, nditer


class TestRegressionUtils(unittest.TestCase):

    def testFeatureNormalization(self):

        test_array = array([[1], [3], [5], [7], [9]])
        (X_normalized, mean_features,
            standard_deviation) = featureNormalization(test_array)

        expectedValue = 5
        actualValue = mean_features[0]

        self.assertEqual(expectedValue, actualValue)

        sigma_value = standard_deviation[0]

        expectedValue = sqrt(8)
        actualValue = sigma_value

        self.assertEqual(floor(expectedValue),
                         floor(actualValue))

        self.assertEqual(expectedValue, actualValue)

        expectedValue = [-4/sigma_value, -2/sigma_value, 0,
                         2/sigma_value, 4/sigma_value]
        size = test_array.shape[0]

        for i in range(size):
            self.assertEqual(X_normalized[i, 0], expectedValue[i])

    def testSigmoid(self):
        value = 0
        expectedValue = 0.5

        self.assertEqual(sigmoid(value), expectedValue)

        values = array([[1, 2], [-1, -2]])
        actualValues = sigmoid(values)

        for value in nditer(actualValues):
            self.assertTrue(0 <= value <= 1)

    def testSigmoidDerivative(self):
        value = 0
        expectedValue = 0.25

        self.assertEqual(sigmoid_derivative(value), expectedValue)

    def testMse(self):
        y = array([[1], [2], [3]])
        y_hat = array([[2], [3], [4]])

        expectedValue = 1
        actualValue = mse(y, y_hat)

        self.assertEqual(expectedValue, actualValue)

        y = array([[1], [2], [3]])
        y_hat = array([[1], [2], [3]])

        expectedValue = 0
        actualValue = mse(y, y_hat)

        self.assertEqual(expectedValue, actualValue)

        y = array([[1], [2], [3]])
        y_hat = array([[3], [1], [5]])

        expectedValue = 3
        actualValue = mse(y, y_hat)

        self.assertEqual(expectedValue, actualValue)

    def testCreateEmptyCopyArray(self):
        item_array = [array([[1], [2], [3]]), array([[1, 2, 3]])]
        empty_array = create_empty_copy_array(item_array)

        for item, empty in zip(item_array, empty_array):
            self.assertEquals(item.shape, empty.shape)

            for value in nditer(empty):
                self.assertEqual(value, 0)
