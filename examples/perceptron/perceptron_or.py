import config

from model.training_data import TrainingData
from perceptron.perceptron import Perceptron


def main():

    print("This example will train a perceptron to learn the OR logic\n\n")

    print ("Initializing data...\n")
    trainingData = TrainingData()
    trainingData.loadTrainingData(config.PERCEPTRON_DATA_OR, ' ')
    trainingData.addColumnOfOnes()

    print ("Initializing perceptron...\n")
    numberOfNeurons = 1
    perceptron = Perceptron(trainingData.getXMatrix(), trainingData.y,
                            numberOfNeurons)
    perceptron.generateWeights()

    print ("Initial values:\n")
    print ("weights: ")
    print perceptron.weights
    print ("Answer: ")
    print perceptron.feedForward()

    print("\nTraining the network with 5 steps and learning rate of 0.25...")
    numSteps = 5
    learningRate = 0.25
    perceptron.trainPerceptron(numSteps, learningRate)

    print ("Final values: \n")
    print ("weights: ")
    print perceptron.weights
    print ("Answer: ")
    print perceptron.feedForward()

if __name__ == "__main__":
    main()
