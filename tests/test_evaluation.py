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

    def testCreateMatrixDisplayTopLabels(self):
        expectedValue = "  | 0 | 1 |\n"
        self.assertEqual(expectedValue,
                         self.confusionMatrix.createMatrixDisplayTopLabels())

        self.confusionMatrix.validLabels = ['X', 'Y', 'Z']
        expectedValue = "  | X | Y | Z |\n"
        self.assertEqual(expectedValue,
                         self.confusionMatrix.createMatrixDisplayTopLabels())

    def testCreateMatrixDisplayBody(self):
        matrixValues = array([[0, 2], [0, 2]])
        expectedValue = "0 | 0 | 2 |\n1 | 0 | 2 |\n"
        actualValue = self.confusionMatrix.createMatrixDisplayBody(
                matrixValues)

        self.assertEqual(expectedValue, actualValue)

        matrixValues = array([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        expectedValue = "X | 1 | 0 | 0 |\nY | 0 | 1 | 1 |\nZ | 1 | 0 | 1 |\n"
        validLabels = ['X', 'Y', 'Z']
        self.confusionMatrix.validLabels = validLabels
        self.confusionMatrix.labelsIndex = validLabels
        actualValue = self.confusionMatrix.createMatrixDisplayBody(
                matrixValues)

        self.assertEqual(expectedValue, actualValue)

    def testGetAccuracy(self):
        matrixValues = array([[2, 0], [0, 2]])
        expectedValue = 1

        self.assertEqual(expectedValue,
                         self.confusionMatrix.getAccuracy(matrixValues))

        matrixValues = array([[2, 2], [2, 2]])
        expectedValue = 0.5

        self.assertEqual(expectedValue,
                         self.confusionMatrix.getAccuracy(matrixValues))
