import unittest
from model.utils import *
from math import floor, sqrt
from numpy import array

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

		self.assertTrue(actualValue==expectedValue)

		expectedValue = [-4/sigma_value, -2/sigma_value, 0,
						 2/sigma_value, 4/sigma_value]

		size = test_array.shape[0]

		for i in range(size):
			self.assertEqual(X_normalized[i, 0], expectedValue[i]) 


	def testSigmoid(self):

		expectedValue = 0.5
		actualValue = sigmoid(0)
		self.assertEqual(expectedValue, actualValue)

		expectedValue = [0.5, 0.9525, 0.002472]
		input_array = array([[0, 3, -6]])
		actualValue = sigmoid(input_array)

		for i in range(len(expectedValue)):
			self.assertTrue(actualValue[0, i] >= expectedValue[i])


		input_matrix = array([[0, 1], [0, -5]])
		expectedValue = array([[0.5, 0.2689], [0.5, 0.0066]])
		actualValue = sigmoid(input_matrix)

		for i in range(2):
			for j in range(2):
				self.assertTrue(actualValue[i,j] >= expectedValue[i,j])
				


