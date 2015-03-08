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


