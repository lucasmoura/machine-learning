import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from model.training_data import TrainingData
from model.utils import featureNormalization
from regression.linear_regression import LinearRegression
from regression.gradient_descent import gradient_descent
from numpy import array, dot, c_, size, ones
import pylab as plb


print("Loading data from ex1data2.txt")

trainingData = TrainingData()
trainingData.load_training_data("../data/ex1data2.txt", ',')


print("\n\nPrinting 10 examples from the dataset:\n")

for i in range(10):

	print("x = [%.0f %.0f], y = %.0f" %(trainingData.x[i, 0], 
								  		trainingData.x[i, 1],
								  		trainingData.y[i, 0]))




print("\nNormalizing features ...\n")

trainingData.normalizeFeatures()
trainingData.add_column_of_ones()

theta_init = array([[0], [0], [0]])
maxIter = 400
regression_type = 1
learningRate = 0.3
regressionType = 0

print("\nPerforming gradient descent..\n")

(all_costs1, theta) = gradient_descent(regression_type, trainingData,
									  theta_init, maxIter, learningRate)

learningRate = 0.01

(all_costs2, theta) = gradient_descent(regression_type, trainingData,
									  theta_init, maxIter, learningRate)

learningRate = 0.1

(all_costs3, theta) = gradient_descent(regression_type, trainingData,
									  theta_init, maxIter, learningRate)




plb.figure()
plb.plot(range(1, size(all_costs1)+1), all_costs1, 'b')
plb.plot(range(1, size(all_costs2)+1), all_costs2, 'r')
plb.plot(range(1, size(all_costs3)+1), all_costs3, 'k')
plb.xlabel("Number of iterations")
plb.ylabel("Cost J")
plb.show()

print("\nTheta computed from gradient descent: \n")
print("%f %f %f\n\n" %(theta[0,0], theta[1,0], theta[2,0]))


test = array([[1650, 3]])
test = test - trainingData.mean_values
test = test/trainingData.standard_deviation_values
test = c_[ones(test.shape[0]), test]
test = dot(test, theta)

print("\nPredict price of a 1650 sq-ft, 3 br house: \n%f\n" 
	  %(test.sum()))


