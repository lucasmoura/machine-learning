import unittest
from model.training_data import TrainingData
from regression.utils import *
from math import floor, sqrt

class TestRegressionUtils(unittest.TestCase):

	def setUp(self):
		
		self.trainingData = TrainingData()	

		delimiter = ','
		training_file = "./tests/test_files/training_data.txt"
		self.trainingData.load_training_data(training_file,
											 delimiter)

	
	def testFeatureNormalization(self):

		(X_normalized, mean_features,
		 standard_deviation) = featureNormalization(self.trainingData.x)

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

		size = self.trainingData.x.shape[0]

		for i in range(size):
			self.assertEqual(X_normalized[i, 0], expectedValue[i]) 


