import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from model.training_data import TrainingData
from regression.linear_regression import LinearRegression
from regression.gradient_descent import gradient_descent
from numpy import array, dot
import pylab as plb

def main():
	
	print ("Initializing data...")
	trainingData = TrainingData()
	trainingData.load_training_data('../data/ex1data1.txt', ',')

	print("Plotting data...")
	plb.figure()
	plb.plot(trainingData.x, trainingData.y, 'rx')
	plb.show()
	
	trainingData.add_column_of_ones()

	theta = array([[0], [0]])
	maxIter = 1500
	learningRate = 0.01
	regression_type = 1

	print "Perfoming linear regression"
	(all_costs, theta) = gradient_descent(regression_type, trainingData,
										  theta, maxIter, learningRate)

	print("Theta found: %f %f" %(theta[0], theta[1]))
	plb.figure()
	plb.plot(trainingData.x[:,1], trainingData.y, 'rx',
			 label = "Training Data")

	x = trainingData.x[:, 1]
	result = dot(trainingData.x, theta)	

	plb.plot(x, result, label = "Linear Regression")
	plb.legend(loc='upper left')
	plb.show()

	print ("Displaying cost during the gradient descent iteration")
	plb.figure()
	plb.plot(all_costs)
	plb.show()

	predict1 = array([[1, 3.5]])
	predict1 = dot(predict1,theta)
	predict1 = predict1.sum()

	print ("For a population = 35,000, we predict a profit of %f\n" %(predict1*10000))

	predict2 = array([[1, 7]])
	predict2 = dot(predict2,theta)
	predict2 = predict2.sum()


	print("For a population = 70,000, we predict a profit of %f\n" %(predict2*10000))


if __name__ == '__main__':
	main()
