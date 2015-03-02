from numpy import array, zeros
from linear_regression import LinearRegression

def gradient_descent(regression_type, trainingData, theta, 
					 maxIter, learningRate):

	"""
	This method is responsible for performing gradient descent
	in order to predict the appropriat theta values for a given
	training data

	:param trainingData: an object that hold the inputs, X, and 
						 results, Y,matrix for a given training
						 set. The input matrix is MX(N+1) matrix,
						 where M is the number of training examples
						 and N is thenumber of features. The results
						 matrix has a dimension MX1 and holds the
						 results for the input matrix X
	
	:param theta: The initial theta values that will be used in
				  the algorithm
	
	:param maxIter: The number of iterations the algorithm will
					perform
	
	:param learningRate: The learning rate used in to perform the
						 gradient descent
	
	:returns: A tuple that contains a array containing the cost of
			  the theta value on each iteration and an array
			  containing the final theta value found
	"""
	regression = None

	if(regression_type == 1):
		regression = LinearRegression()

	all_costs = zeros(maxIter)
	
	for num_iter in range(maxIter):

		gradient = regression.calculateGradient(trainingData, theta)
		gradient = gradient*learningRate;

		theta = theta - gradient;

		all_costs[num_iter] = regression.calculateCost(trainingData, theta)

	return (all_costs, theta)	



