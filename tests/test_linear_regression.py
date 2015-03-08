import unittest

from mock import MagicMock, PropertyMock
from regression.linear_regression import LinearRegression
from numpy import array
from math import floor


class TestLinearRegression(unittest.TestCase):

	def setUp(self):
		self.linearRegression = LinearRegression()
		

	def createTrainingDataMock(self, x_array, y_array):

		trainingDataMock = MagicMock()

		trainingDataMock.getXMatrix.return_value = x_array

		type(trainingDataMock).y = PropertyMock(return_value=
												y_array)

		return trainingDataMock


	def testCalculateCost(self):

		theta = array( [ [0], [0] ])
		x_array = array( [ [1,2], [1,3], [1, 4] ])
		y_array = array( [ [3], [4], [5] ])

		trainingDataMock = self.createTrainingDataMock(x_array, y_array)

		actual_cost = self.linearRegression.calculateCost(trainingDataMock, theta)
		expected_cost = 8.333	

		self.assertEqual(floor(actual_cost), floor(expected_cost))
		self.assertTrue(actual_cost>expected_cost)

		x_array = array( [ [1,2,3], [1,4,5]])
		y_array = array([ [1], [2] ])
		theta = array( [ [1], [1], [1] ])

		trainingDataMock = self.createTrainingDataMock(x_array, y_array)

		actual_cost = self.linearRegression.calculateCost(trainingDataMock, theta)
		expected_cost = 22.25	

		self.assertEqual(floor(actual_cost), floor(expected_cost))
		self.assertTrue(actual_cost>=expected_cost)

	def testCalculateGradient(self):

		theta = array( [ [0], [0] ])
		x_array = array( [ [1,2], [1,3], [1, 4] ])
		y_array = array( [ [3], [4], [5] ])

		trainingDataMock = self.createTrainingDataMock(x_array, y_array)

		gradient = self.linearRegression.calculateGradient(trainingDataMock, theta)

		expected_value = -4;
		self.assertEqual(gradient[0,0], expected_value)

		expected_value = -12.66
		self.assertEqual(floor(gradient[1,0]), floor(expected_value))
		self.assertTrue(gradient[1,0]<=expected_value)
		
		x_array = array( [ [1,2,3], [1,4,5]])
		y_array = array([ [1], [2] ])
		theta = array( [ [1], [1], [1] ])

		trainingDataMock = self.createTrainingDataMock(x_array, y_array)

		gradient = self.linearRegression.calculateGradient(trainingDataMock, theta)

		expected_value = 6.5;
		self.assertEqual(gradient[0,0], expected_value)

		expected_value = 21
		self.assertEqual(gradient[1,0], expected_value)

		expected_value = 27.5
		self.assertEqual(gradient[2,0], expected_value)
