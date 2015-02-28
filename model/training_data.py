from numpy import loadtxt, array

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
		"""
		try:		
			training_data = loadtxt(dataFile, delimiter = delimiter)
			num_columns = training_data.shape[1]

			self.x = training_data[:, 0:num_columns-1]
			self.y = training_data[:, num_columns-1:num_columns]

			return self.FILE_LOADED_SUCCESSFULLY;

		except IOError:
			raise
			
	def print_x(self):
	   
	   	print(self.x.shape)
		print(self.x)

	def print_y(self):
		print("\n\n\nY\n\n\n")
		print(self.y)
