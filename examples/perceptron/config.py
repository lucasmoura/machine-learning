import os
import sys

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

DATA_FOLDER = "../../data/"
PERCEPTRON_FOLDER = DATA_FOLDER+"perceptron/"

PERCEPTRON_DATA_OR = PERCEPTRON_FOLDER+'perceptron_or.txt'
PERCEPTRON_DATA_AND = PERCEPTRON_FOLDER+'perceptron_and.txt'
PERCEPTRON_DATA_PIMA = PERCEPTRON_FOLDER+'pima-indians-diabetes.data'
