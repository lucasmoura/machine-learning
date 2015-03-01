from regression import Regression
from numpy import array, dot

class LinearRegression(Regression):

	"""
	This class will be used for implementing the linear regression algorithm. It will
	override the methods calculateCost and calculateGradient defined on the Regression
	base class.
	"""

	def __init__(self):
		pass


	def calculateCost(self, trainingData, theta):

		num_training = trainingData.x.shape[0]

		h_function = dot(trainingData.x,theta)

		error_calc = h_function - trainingData.y;

		cost = dot(error_calc.T, error_calc)

		cost = cost/float(2*num_training);

		return cost.sum();

	
	def calculateGradient(self, trainingData, theta):
		pass
