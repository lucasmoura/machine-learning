import os
import sys

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

DATA_FOLDER = "../../data/"
PERCEPTRON_FOLDER = "perceptron/"

PERCEPTRON_DATA_OR = DATA_FOLDER+PERCEPTRON_FOLDER+'perceptron_or.txt'
PERCEPTRON_DATA_AND = DATA_FOLDER+PERCEPTRON_FOLDER+'perceptron_and.txt'
