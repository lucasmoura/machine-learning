from numpy import loadtxt, c_, ones

class TrainingData():

	FILE_LOADED_SUCCESSFULLY = 0

	def __init__(self):
		pass


	def load_training_data(self, dataFile, delimiter):
		"""
		This method is used to load the training data from a txt file. Currently, the testing
		file need to be a txt file and every line on the file must possess the same amount
		of data

		This method is actually responsible for creating the X matrix, that will be used to train the machine learning methods, and Y matrix,
		that represents the answer for the X matrix.

		:param dataFile: the path used to locate the training data file
		:param delimiter: the delimiter used to separate each value on the file

		:returns: A constant indicating that the file was read successfully or raise an IOError otherwise
		"""
		try:		
			training_data = loadtxt(dataFile, delimiter = delimiter)
			num_columns = training_data.shape[1]

			self.x = training_data[:, 0:num_columns-1]
			self.y = training_data[:, num_columns-1:num_columns]

			return self.FILE_LOADED_SUCCESSFULLY;

		except IOError, ValueError:
			raise
			
	def add_one_column(self):
		"""
		This method is used for making it easier to create a vectorized solution for many machine learning algorithms.
		It will basically add a extra columns of "1" at the beginning of the X matrix
		"""

		self.x = c_[ones(self.x.shape[0]), self.x]
