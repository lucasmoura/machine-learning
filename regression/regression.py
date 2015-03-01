from abc import ABCMeta, abstractmethod
from model.training_data import TrainingData

class Regression(object):
	"""
	Base class used for regression algorithms. It will define a interface for other
	regression algorithms to follow.
	"""

	__metaclass__ = ABCMeta


	@abstractmethod
	def calculateCost(trainingData, theta):

		"""
		Method used to calculate the cost for a given theta value. In machine learning, this function is normally
		used to verify if the algorithm is converging to an expected answer.

		:param trainingData: A object that contain the inputs X, which is a MXN matrix containing the inputs used in the problem,
							 where M is the number of trainingexamples and N is the number of features. It is expected that this
							 matrix possess an extra column of ones in order to make vectorization approaches easier to compute.
							 This object also contains the results Y, which is a matrix MX1 containing the results the given inputs value.
		:param theta: The varaibles that will be calculated by the machine learning algorithm.

		:returns: The total cost for all inputs passed given the theta values passed as parameter.
		"""

	@abstractmethod
	def calculateGradient(trainingData, theta):

		"""
		Method used to calculate the gradient for a given regression problem. The gradient is normally used on gradient descent
		algorithms.

		:param trainingData: A object that contain the inputs X, which is a MXN matrix containing the inputs used in the problem,
							 where M is the number of trainingexamples and N is the number of features. It is expected that this
							 matrix possess an extra column of ones in order to make vectorization approaches easier to compute.
							 This object also contains the results Y, which is a matrix MX1 containing the results the given inputs value.
		:param theta: The varaibles that will be calculated by the machine learning algorithm.

		:returns: A matrix containing the gradient for each theta value. Therefore, it will return a matrix with the same
				  dimension as the theta matrix.

		"""	

