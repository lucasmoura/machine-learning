import config
import numpy as np

from model.training_data import TrainingData
from perceptron.perceptron import Perceptron
from model.evaluation import ConfusionMatrix


def dataPreprocessing(trainingData):
    data = trainingData.getXMatrix()
    data[np.where(data[:, 0] > 8), 0] = 8
    data[np.where(data[:, 7] <= 30), 7] = 1
    data[np.where((data[:, 7] > 30) & (data[:, 7] <= 40)), 7] = 2
    data[np.where((data[:, 7] > 40) & (data[:, 7] <= 50)), 7] = 3
    data[np.where((data[:, 7] > 40) & (data[:, 7] <= 50)), 7] = 3
    data[np.where((data[:, 7] > 50) & (data[:, 7] <= 60)), 7] = 4
    data[np.where((data[:, 7] > 60) & (data[:, 7] <= 70)), 7] = 5
    data[np.where((data[:, 7] > 70) & (data[:, 7] <= 80)), 7] = 6
    data[np.where((data[:, 7] > 80) & (data[:, 7] <= 90)), 7] = 7
    data[np.where((data[:, 7] > 90) & (data[:, 7] <= 100)), 7] = 8

    print trainingData.getXMatrix()
    trainingData.normalizeFeatures()

    trainingData.addColumnOfOnes()


def main():

    print("This example will train a perceptron to learn the PIMA dataset\n\n")

    print ("Initializing data...\n")
    trainingData = TrainingData()
    trainingData.loadTrainingData(config.PERCEPTRON_DATA_PIMA, ',')

    dataPreprocessing(trainingData)

    trainingData.addColumnOfOnes()

    print ("Initializing perceptron...\n")
    numberOfNeurons = 1
    perceptron = Perceptron(trainingData.getXMatrix(), trainingData.y,
                            numberOfNeurons)
    perceptron.generateWeights()

    print ("Initial values:\n")
    print ("weights: ")
    print perceptron.weights

    print("\nTraining the network with 10 steps and learning rate of 0.25...")
    numSteps = 50
    learningRate = 0.25
    perceptron.trainPerceptron(numSteps, learningRate)

    print ("Final values: \n")
    print ("weights: ")
    print perceptron.weights
    answer = perceptron.feedForward()

    print("Confusion Matrix: ")
    confusionMatrix = ConfusionMatrix(trainingData.y, answer)
    confusionMatrix.displayMatrix()


if __name__ == "__main__":
    main()
