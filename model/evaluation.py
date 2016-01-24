import numpy as np


class ConfusionMatrix(object):

    """
    This class will be used to reprent a confusiom matrix, which is one
    way to evaluate a machine learning algorithm. This matrix will
    possess the following format:

       | C1 | C2 | C3 |
    C1 | 5  | 1  | 0  |
    C2 | 1  | 4  | 1  |
    C3 | 2  | 0  | 4  |

    In the example above, each row of the matrix represents
    how the label Cx was classified in a certain output of
    a machine learning algorithm, In the first row, the label
    C1 was correctly classified as C1 5 times, and incorrectly
    classified as C2 one time. The number of datapoints labeled
    as C1 can be obtained by the sum of the row's column and the
    overall accuracy of the machine learning algorithm, can be
    obtained by the sum of the matrix diagonal divided by
    the sum of the whole matrix.
    """

    """
    :param inputLabels:  The labels that come with the data
                         used to train the machine learning algorithm.
                         This variable must be a column vector.
    :param outputLabels: The labels that were generated by the machine
                         learning algorithm. This variable must also
                         be a column matrix.
    :param validLabels   The possible and ordered label values that
                         a datapoint can be classified on. This parameter
                         is required in order to better enumerate
                         the labels for creating the matrix values.
                         By default, this parameter receives the array
                         containing [0, 1].
    """
    def __init__(self, inputLabels, outputLabels, validLabels=[0, 1]):
        self.inputLabels = inputLabels
        self.outputLabels = outputLabels
        self.validLabels = validLabels
        self._numLabels = len(validLabels)
        self._labelsIndex = self.createLabelsIndex(validLabels)

    """
    Get method for num labels.
    """
    @property
    def numLabels(self):
        return self._numLabels

    """
    Set the value of numLabels as the number of elements inside
    the validLabels array.
    """
    @numLabels.setter
    def numLabels(self, validLabels):
        self._numLabels = len(validLabels)

    """
    Get method for labels index.
    """
    @property
    def labelsIndex(self):
        return self._labelsIndex

    """
    Method used to create a dictionary from the valid labels
    to discrete values. For example, given the valid labels
    [x, y, z], the following dictionary will be created:

    {
        x: 0
        y: 1
        z: 2
    }

    :param validLabels: Ordered and possible labels that a datapoint
                        can assume.

    :return: The labels dictionary already explained.
    """
    def createLabelsIndex(self, validLabels):
        labelDict = {}
        indexValue = 0

        for label in validLabels:
            labelDict[label] = indexValue
            indexValue += 1

        return labelDict

    """
    This method set labelsIndex as the return of createLabelsIndex.
    """
    @labelsIndex.setter
    def labelsIndex(self, validLabels):
        self._labelsIndex = self.createLabelsIndex(validLabels)

    """
    This method is used to create the confusion matrix values. However,
    a separate one will be used to construct the matrix visualization.
    """
    def confusionMatrixValues(self):
        matrixValues = np.zeros(shape=(self.numLabels, self.numLabels))
        size = self.inputLabels.shape[0]

        for i in range(size):
            row = self.labelsIndex[self.inputLabels[i][0]]
            column = self.labelsIndex[self.outputLabels[i][0]]

            matrixValues[row][column] += 1

        return matrixValues

    """
    This method will create the top part of the Confusion Matrix. For example,
    given the following Confusion Matrix:

       | C1 | C2 |
    C1 | 1  | 0  |
    C2 | 0  | 1  |

    The method will return:

       | C1 | C2 |

    :return: The Confusion Matrix header, containing the labels.

    """
    def createMatrixDisplayTopLabels(self):
        confusionMatrixLabels = " "

        for label in self.validLabels:
            confusionMatrixLabels += " | "+str(label)

        return confusionMatrixLabels+" |\n"

    """
    This method will create the body part of the Confusion Matrix. For example,
    given the following Confusion Matrix:

       | C1 | C2 |
    C1 | 1  | 0  |
    C2 | 0  | 1  |

    The method will return:

    C1 | 1  | 0  |
    C2 | 0  | 1  |

    :param matrixValues: The values that will populate the Confusion Matrix.
                         This values are generated by the method
                         confusionMatrixValues.

    :return: The Confusion Matrix body populated with the correct values.

    """
    def createMatrixDisplayBody(self, matrixValues):
        confusionMatrixBody = ""

        for label in self.validLabels:
            confusionMatrixBody += str(label)
            row = self.labelsIndex[label]

            for column in range(matrixValues.shape[1]):
                value = int(matrixValues[row][column])
                confusionMatrixBody += " | "+str(value)

            confusionMatrixBody += " |\n"

        return confusionMatrixBody

    """
    This method is used to put together the values and the actual matrix.
    Therefore, it serves as an interface to other method, in order to create
    and display the Confusion Matrix.
    """
    def displayMatrix(self):
        matrixValues = self.confusionMatrixValues()

        confusionMatrixStr = self.createMatrixDisplayTopLabels()
        confusionMatrixStr += self.createMatrixDisplayBody(matrixValues)

        print(confusionMatrixStr)
        print("Accuracy: {0}".format(self.getAccuracy(matrixValues)))

    """
    This method will be used to get the accuracy of a given
    machine learning algorithm. It basically get the sum of
    the diagonal of the Confusion Matrix divided by the sum of
    the whole matrix.

    :param matrixValues: The values that will populate the Confusion Matrix.
                         This values are generated by the method
                         confusionMatrixValues.

    :return: An accuracy value for Confusion Matrix values.
    """
    def getAccuracy(self, matrixValues):
            return float(np.trace(matrixValues))/np.sum(matrixValues)
