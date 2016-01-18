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
    :param inputMatrix:   A matrix N x m containing the input vector
                         that will be used to train the perceptron,
                         where N is the number of training examples
                         and m is the number of features that each
                         training example has
    :param answerMatrix:  A matrix N X 1 containing the right labels
                         for each input vector of the inputMatrix
                         variable
    :param numNeurons:    Number of output neurons that the network will
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
    def feedForward(self):
        activation = dot(self.inputMatrix, self.weights)
        return where(activation > 0, 1, 0)

    """
    Method used to update the weights of each output neuron based
    on the given results of a training step. The error can be calculated by
    the function:

    wij -= n(yj - tj)*xi

    where:

        n  = learning rate
        yi = correct label for the given input vector
        tj = generated label for the given input vectoro
        xi = feature associated with the given weight being updated

    :param resultMatrix: A Nxn matrix that hold the obtained labels t for
                         a certain training step.
    :param learningRate: The learning rate that will be used on the weight
                         update.
    """
    def updateWeights(self, resultMatrix, learningRate):
        self.weights -= learningRate * dot(self.inputMatrix.T,
                                           resultMatrix - self.answerMatrix)

    """
    Method used to train the perceptron algorithm. For each step, it first
    calculate the feedForward answer for the given weights and the update the
    weights given the received answer.

    :param numSteps:     The number of training steps the algorithm will need
                         to reach an optimal result.
    :param learningRate: The learning rate that will be used on the weight
                         update.
    """
    def trainPerceptron(self, numSteps, learningRate):
        for step in range(numSteps):
            resultMatrix = self.feedForward()
            self.updateWeights(resultMatrix, learningRate)

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
