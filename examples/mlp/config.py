import os
import sys

lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)

DATA_FOLDER = "../../data/"
MLP_FOLDER = DATA_FOLDER + "mlp/"

MNIST_DATA = DATA_FOLDER + 'mnist.pkl.gz'
