import unittest
from numpy import array

from model.evaluation import ConfusionMatrix


class ConfusionMatrixTest(unittest.TestCase):

    def setUp(self):
        self.confusionMatrix = ConfusionMatrix(None, None)

    def setConfusionMatrix(self, inputLabels, outputLabels, validLabels):
        self.confusionMatrix.inputLabels = inputLabels
        self.confusionMatrix.outputLabels = outputLabels
        self.confusionMatrix.labelsIndex = validLabels
        self.confusionMatrix.numLabels = validLabels

    def compareArrays(self, expectedValues, actualValues):
        for i in range(expectedValues.shape[0]):
            for j in range(expectedValues.shape[1]):
                    self.assertEqual(expectedValues[i][j], actualValues[i][j])

    def testConfusionMatrixValues(self):
        inputLabels = array([[1], [0], [1], [0]])
        outputLabels = array([[1], [1], [1], [1]])
        validLabels = [0, 1]

        expectedValues = array([[0, 2], [0, 2]])

        self.setConfusionMatrix(inputLabels, outputLabels, validLabels)

        actualValues = self.confusionMatrix.confusionMatrixValues()

        self.compareArrays(expectedValues, actualValues)

        inputLabels = array([['X'], ['Y'], ['Z'], ['Y'], ['Z']])
        outputLabels = array([['X'], ['Z'], ['X'], ['Y'], ['Z']])
        validLabels = ['X', 'Y', 'Z']

        expectedValues = array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])

        self.setConfusionMatrix(inputLabels, outputLabels, validLabels)

        actualValues = self.confusionMatrix.confusionMatrixValues()

        self.compareArrays(expectedValues, actualValues)
