import numpy as np

class MultiLayerPerceptron:

    """
    Create a  MultiLayer perceptron network.

    :param layer:       An array containing the amount of neurons each layer in
                        the network will have, including both input and output
                        layer.
    """
    def __init__(self, layers):
        self.num_layers = len(layers)

        """
        The input layer should not have any bias values associated with it.
        Also, the biases will be column vectors.
        """
        self.biases = [np.random.randn(layer, 1) for layer in layers[1:]]


