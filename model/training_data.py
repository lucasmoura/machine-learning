from numpy import loadtxt, c_, ones
from utils import featureNormalization


class TrainingData():

    FILE_LOADED_SUCCESSFULLY = 0

    def __init__(self):
        self.x = None
        self.y = None
        self.x_normalized = None
        self.mean_values = None
        self.standard_deviation_values = None
        self.is_normalized = False

    def load_training_data(self, dataFile, delimiter):
        """
        This method is used to load the training data from a txt file.
        Currently, the testing file need to be a txt file and every line
        on the file must possess the same amount of data

        This method is actually responsible for creating the X matrix, that
        will be used to train the machine learning methods, and Y matrix,
        that represents the answer for the X matrix.

        :param dataFile:  the path used to locate the training data file
        :param delimiter: the delimiter used to separate each value on
                            the file

        :returns: A constant indicating that the file was read successfully
                    or raise an IOError otherwise
        """
        try:
            training_data = loadtxt(dataFile, delimiter=delimiter)
            num_columns = training_data.shape[1]

            self.x = training_data[:, 0:num_columns-1]
            self.y = training_data[:, num_columns-1:num_columns]

            return self.FILE_LOADED_SUCCESSFULLY

        except IOError:
            raise

    def add_column_of_ones(self):
        """
        This method is used for making it easier to create a vectorized
        solution for many machine learning algorithms. It will basically
        add a extra columns of "1" at the beginning of the X matrix or
        the Normalized X matrix if the process of normalization had
        already been done
        """

        if self.is_normalized:
            self.x_normalized = c_[ones(self.x_normalized.shape[0]),
                                   self.x_normalized]
        else:
            self.x = c_[ones(self.x.shape[0]), self.x]

    def normalizeFeatures(self):

        """
            This method is used to apply feature scaling on the features laoded
            on the x matrix. This method also populates the attributes
            x_normalized, mean_values and standard_deviation_values
        """

        (X_norm, mean_values, sigma_values) = featureNormalization(self.x)

        self.x_normalized = X_norm
        self.mean_values = mean_values
        self.standard_deviation_values = sigma_values
        self.is_normalized = True

    def getXMatrix(self):

        """
            Method used to get the correct x matrix for the training data.
            If the features were regularized, the method will return
            the x_normalized matrix, else the x one
        """

        if self.is_normalized:
            return self.x_normalized
        else:
            return self.x
