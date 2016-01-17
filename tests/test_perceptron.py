import unittest

from perceptron.perceptron import Perceptron
from model.training_data import TrainingData


class PerceptronTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        trainingData = TrainingData()
        trainingData.loadTrainingData('data/perceptron_or.txt', ' ')
        trainingData.addColumnOfOnes()

        cls.x = trainingData.getXMatrix()
        cls.y = trainingData.y

    def setUp(self):
        self.perceptron = Perceptron(PerceptronTest.x, PerceptronTest.y, 1)

    def testGenerateWeights(self):
        expectedRows = 3
        expectedColumns = 1

        self.perceptron.generateWeights()

        self.assertEqual(expectedRows,
                         self.perceptron.weights.shape[0])
        self.assertEqual(expectedColumns,
                         self.perceptron.weights.shape[1])
