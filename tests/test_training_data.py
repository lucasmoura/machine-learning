import unittest

from model.training_data import TrainingData
from os import getcwd

class TestTrainingData(unittest.TestCase):

	def setUp(self):
		self.trainingData = TrainingData()

	def testLoadTrainingData(self):
		
		delimiter = ','
		training_file = './tests/test_files/training_data.txt'
		expected_value = 0;

		return_value = self.trainingData.load_training_data(training_file,
															delimiter)

		self.assertEqual(0, return_value)

		x_size = self.trainingData.x.shape;
		num_rows_x = x_size[0];
		expected_value = 5; 

		self.assertEqual(num_rows_x, expected_value)
		
		num_columns_x = x_size[1];
		expected_value = 1;
		
		self.assertEqual(num_columns_x, expected_value)

		y_size = self.trainingData.y.shape;
		num_rows_y = y_size[0]
		expected_value = 5; 

		self.assertEqual(num_rows_y, expected_value)
		
		num_columns_y = y_size[1];
		expected_value = 1;
		
		self.assertEqual(num_columns_y, expected_value)

	def testTrainingDataWithMoreInputs(self):

		delimiter = ','
		training_file = './tests/test_files/training_data_more_inputs.txt'
		expected_value = 0;

		return_value = self.trainingData.load_training_data(training_file,
															delimiter)

		self.assertEqual(0, return_value)

		x_size = self.trainingData.x.shape;
		num_rows_x = x_size[0];
		expected_value = 6; 

		self.assertEqual(num_rows_x, expected_value)
		
		num_columns_x = x_size[1];
		expected_value = 4;
		
		self.assertEqual(num_columns_x, expected_value)

		y_size = self.trainingData.y.shape;
		num_rows_y = y_size[0]
		expected_value = 6; 

		self.assertEqual(num_rows_y, expected_value)
		
		num_columns_y = y_size[1];
		expected_value = 1;
		
		self.assertEqual(num_columns_y, expected_value)

	def testLoadInvalidFile(self):

		delimiter = ','
		training_file = "./tests/test_files/invalid_file.txt"

		with self.assertRaises(IOError):
			self.trainingData.load_training_data(training_file,
											 delimiter)


	def testLoadInvalidFormatFile(self):
		
		delimiter = ','
		training_file = "./tests/test_files/wrong_format_1.txt"

		with self.assertRaises(ValueError):
			self.trainingData.load_training_data(training_file,
										 		 delimiter)

		delimiter = ','
		training_file = "./tests/test_files/wrong_format_2.txt"

		with self.assertRaises(ValueError):
			self.trainingData.load_training_data(training_file,
										 		 delimiter)

	def testAddColumnsofOnes(self):

		delimiter = ','
		training_file = "./tests/test_files/training_data_more_inputs.txt"

		return_value = self.trainingData.load_training_data(training_file,
															delimiter)

		expected_value = 0;
		self.assertEqual(expected_value, return_value)

		num_rows_x = self.trainingData.x.shape[0]

		num_columns_x = self.trainingData.x.shape[1];
		expected_value = 4;
		
		self.assertEqual(num_columns_x, expected_value)

		self.trainingData.add_column_of_ones()

		num_columns_x = self.trainingData.x.shape[1];
		expected_value = 5;

		self.assertEqual(expected_value, num_columns_x)

		expected_value = 1.0
		for i in range(num_rows_x):
			self.assertEqual(expected_value, self.trainingData.x[i,0])



