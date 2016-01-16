from regression import Regression
from numpy import dot


class LinearRegression(Regression):

    """
    This class will be used for implementing the linear regression algorithm.
    It will override the methods calculateCost and calculateGradient
    defined on the Regression base class.
    """

    def __calculateError(self, trainingData, theta):

        """
        This method is responsible for calculating the error
        between the h_function and the actual results.

        :param trainingData: An object containing both input matrix, which
                             is a MX(N+1) matrix, where M is the number of
                             training examples and N is the number of features.
                             This object also holds the result for the input
                             matrix, which is a MX1 matrix.
        :param results:      The real results for the given inputs
                             the problem. This is also a MX1 matrix.

        :returns: a MX1 matrix containing the error between the
                            the h_function and results
        """

        h_function = dot(trainingData.getXMatrix(), theta)

        error_calc = h_function - trainingData.y

        return error_calc

    def calculateCost(self, trainingData, theta):

        num_training = trainingData.getXMatrix().shape[0]

        error_calc = self.__calculateError(trainingData, theta)

        cost = dot(error_calc.T, error_calc)
        cost = cost/float(2*num_training)

        return cost.sum()

    def calculateGradient(self, trainingData, theta):

        num_training = trainingData.getXMatrix().shape[0]

        error_calc = self.__calculateError(trainingData, theta)

        return (dot(trainingData.getXMatrix().T,
                error_calc))/float(num_training)
