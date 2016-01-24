import unittest

from numpy import array

from perceptron.perceptron import Perceptron
from model.training_data import TrainingData


class PerceptronTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        trainingData = TrainingData()
        trainingData.loadTrainingData('data/perceptron/perceptron_or.txt', ' ')
        trainingData.addColumnOfOnes()

        cls.x = trainingData.getXMatrix()
        cls.y = trainingData.y

    def setUp(self):
        self.perceptron = Perceptron(PerceptronTest.x, PerceptronTest.y, 1)

    def compareColumnMatrix(self, expectedValues, actualValues):
        for i in range(actualValues.shape[0]):
            self.assertEqual(expectedValues[i][0], actualValues[i][0])

    def testGenerateWeights(self):
        expectedRows = 3
        expectedColumns = 1

        self.perceptron.generateWeights()

        self.assertEqual(expectedRows,
                         self.perceptron.weights.shape[0])
        self.assertEqual(expectedColumns,
                         self.perceptron.weights.shape[1])

    def testFeedForward(self):
        expectedValues = array([[1], [1], [1], [1]])

        self.perceptron.weights = array([[0.5], [0.2], [-0.1]])

        actualValues = self.perceptron.feedForward()

        for i in range(actualValues.shape[0]):
            self.assertEqual(expectedValues[i][0], actualValues[i][0])

        expectedValues = array([[1], [0], [1], [1]])

        self.perceptron.weights = array([[0.3], [0.2], [-0.3]])

        actualValues = self.perceptron.feedForward()

        self.compareColumnMatrix(expectedValues, actualValues)

    def testUpdateWeights(self):
        expectedValues = array([[-1.25], [0.5], [0.2]])

        self.perceptron.weights = array([[-1], [0.5], [0.2]])

        answerMatrix = array([[1], [1], [1], [1]])
        learningRate = 0.25

        self.perceptron.updateWeights(answerMatrix, learningRate)

        self.compareColumnMatrix(expectedValues, self.perceptron.weights)
