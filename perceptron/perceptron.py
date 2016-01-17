from numpy import random, dot, where


class Perceptron():

    """
    This class represents a percentron algorithm implementation.
    Therefore, it will possess only two layer of nodes, an input
    layer, containing the learning problem inputs and an output
    layer, which contains the neurons responsible for generating
    the learning output.
    """

    """
    param inputMatrix:   A matrix N x m containing the input vector
                         that will be used to train the perceptron,
                         where N is the number of training examples
                         and m is the number of features that each
                         training example has
    param answerMatrix:  A matrix N X 1 containing the right labels
                         for each input vector of the inputMatrix
                         variable
    param numNeurons:    Number of output neurons that the network will
                         possess. This number depends on the number of
                         possible outputs that the network can have
    """

    def __init__(self, inputMatrix, answerMatrix, numNeurons):
        self.inputMatrix = inputMatrix
        self.answerMatrix = answerMatrix
        self.numNeurons = numNeurons

    """
    Method used to calculate the single output value
    of the perceptron algorithm. This method also holds
    the threshold implementation, which verifies if a
    neuron should fire or not.
    """
    def recall(self):
        activation = dot(self.inputMatrix, self.weights)
        return where(activation > 0, 1, 0)

    def trainPerceptron(self):
        pass

    """
    This method will be used to initialize the weights
    matrix for the perceptron algorithm. This matrix will
    be m X n, where m is the number of features an input
    vector has and n is the number of output neurons
    the algorithm has.
    """
    def generateWeights(self):
        numInputs = self.inputMatrix.shape[1]
        self.weights = random.rand(numInputs,
                                   self.numNeurons) * 0.1 - 0.05
