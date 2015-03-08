from numpy import mean, std

def featureNormalization(X_inputs):

	"""This method is responsible for applying feature scaling on
	   the features of an X input matrix, which is a MXN matrix, where
	   M is the number of training examples and N is the number of features

	   The X_inputs matrix should not have an extra column of ones added to it.
	   With that said, this method will basically calculate the mean value for every
	   feature on X_inputs and subtract this value from the X_input dataset. After that,
	   every feature will be normalized by the standard deviation of each feature.

	   :param X_inputs: A MXN matrix containing the traning examples.

	   :returns: A tuple containing three different attributes. The first one is the
	   			 normalized X_input matrix, which will also be a MXN matrix. The 
				 second one is a mean vector for each feature on X_input and
				 the third and final one is a standard deviation vector containing the
				 standard deviation for each feature of X_input after the mean values were
				 already subtracted from the dataset.

	"""

	X_normalized = X_inputs;
	mean_features = mean(X_inputs, axis = 0)
	X_normalized = X_normalized - mean_features

	standard_deviation = std(X_normalized, axis = 0)
	X_normalized = X_normalized/standard_deviation

	return (X_normalized, mean_features, standard_deviation)
