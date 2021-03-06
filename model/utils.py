import numpy as np


def featureNormalization(X_inputs):

    """
        This method is responsible for applying feature scaling on
        the features of an X input matrix, which is a MXN matrix, where
        M is the number of training examples and N is the number of features

        The X_inputs matrix should not have an extra column of ones
        added to it. With that said, this method will basically
        calculate the mean value for every feature on X_inputs and subtract
        this value from the X_input dataset. After that, every feature will
        be normalized by the standard deviation of each feature.

        :param X_inputs: A MXN matrix containing the traning examples.

        :returns: A tuple containing three different attributes. The first one
                  is the normalized X_input matrix, which will also be a M X N
                  matrix. The  second one is a mean vector for each feature on
                  X_input and the third and final one is a standard deviation
                  vector containing the standard deviation for each feature of
                  X_input after the mean values were already subtracted from
                  the dataset.

    """

    X_normalized = X_inputs
    mean_features = np.mean(X_inputs, axis=0)
    X_normalized = X_normalized - mean_features

    standard_deviation = np.std(X_normalized, axis=0)
    X_normalized = X_normalized/standard_deviation

    return (X_normalized, mean_features, standard_deviation)


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_derivative(value):
    return sigmoid(value) * (1 - sigmoid(value))


def mse(y, y_hat):
    """
        Calculate the Minimum Square Error cost for a range of predictions.

        :param y:       The predictions created by a machine learning model
        :param y_hat:   The real predictions for the data feed to the machine
                        learning model.

        :returns: The metric value for the given predictions.
    """
    return np.mean(np.square(y_hat - y))


def mse_derivative(y, y_hat):
    return y - y_hat


def cross_entropy(y, y_hat):
    return -np.sum(y_hat * np.log(y) + (1 - y_hat) * np.log(1 - y))


def cross_entropy_derivative(y, y_hat):
    return -((y_hat / y) + ((y_hat - 1) / (1 - y)))


def create_empty_copy_array(item_array):
    return [np.zeros(item.shape) for item in item_array]
