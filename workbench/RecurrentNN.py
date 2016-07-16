# Example class for a randomforsest-predictor

#from .. 
import utils

logger = utils.getLogger("RecurrentNN.py")		

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization


from .BaseModel import BaseModel

def slidingWindow(X,windowSize = 10, numWindows=-1):
	if numWindows == -1:
		numWindows = len(X)-windowSize
	logger.debug("Generating %d windows" % numWindows)
	i = 0
	while i < numWindows:
		yield list(np.array(X[i:i+windowSize]))
		i += 1

class RecurrentNN(BaseModel):

	def __init__(self, args):
		# ignore the args for now
		pass

		
	# method that fits a version with the whole time series of the data
	# does not work at all (is not converging)
	# probably: window too long
	'''
	def fit(self, X, Y):
		# Here comes the machine learning
		logger.debug("Starting RecurrentNN.fit()")
		logger.debug("Building RecurrentNN")
		
		#min_samples_leaf = 5
		#n_estimators = 100
		#logger.debug("Using parameters: min_samples_leaf = %d, n_estimators = %d" % (min_samples_leaf, n_estimators))
		
		#self.model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, min_samples_leaf=min_samples_leaf)
		#self.model.fit(X, Y)		
		
		print("Shape of input X: " + str(X.shape))
		if Y is not None:
			print("Shape of expected output Y: " + str(Y.shape))
		
		inputdim = X.shape[-1]  # last dimension of X is the number of features
		print("Setting input dim to %d" % inputdim)
		L1_outputdim = 1
		#L2_outputdim
		
		Final_outputdim = Y.shape[-1] # last dimension of X is the number of features
		print("Setting input Final_outputdim to %d" % Final_outputdim)
		
		self.model = Sequential()
		self.model.add(LSTM(Final_outputdim, input_dim=inputdim, return_sequences = True ))    
		#self.model.add(LSTM(Final_outputdim, input_dim=inputdim, return_sequences = True ))    
		logger.debug("Compiling RecurrentNN")
		self.model.compile(loss='mse', optimizer='rmsprop')
		
		# preparing input data
		print("preparing input data for model")
		
		X1 = np.reshape(X.as_matrix(), (1, X.shape[0], X.shape[1]))
		Y1 = np.reshape(Y.as_matrix(), (1, Y.shape[0], Y.shape[1]))
		print("Shape of input X1: " + str(X1.shape))
		print("Shape of input Y1: " + str(Y1.shape))
		
		print(X1[0,0:10])
		print(Y1[0,0:10])
		
		if Y is not None:		
			epochs = 2
			batch_size = 1
			logger.debug("Running Fit")
			for i in range(epochs):
				print('Epoch', i, '/', epochs)
				self.model.fit(X1,
						  Y1,
						  batch_size=batch_size,
						  verbose=1,
						  nb_epoch=1,
						  shuffle=False)
				self.model.reset_states()
		
		logger.debug("Done RandomForestRegressor.fit()")
		pass
	'''
	def fit(self, X, Y):
		# Here comes the machine learning
		logger.debug("Starting RecurrentNN.fit()")
		logger.debug("Building RecurrentNN")
		
		#min_samples_leaf = 5
		#n_estimators = 100
		#logger.debug("Using parameters: min_samples_leaf = %d, n_estimators = %d" % (min_samples_leaf, n_estimators))
		
		#self.model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, min_samples_leaf=min_samples_leaf)
		#self.model.fit(X, Y)		
		
		print("Shape of input X: " + str(X.shape))
		if Y is not None:
			print("Shape of expected output Y: " + str(Y.shape))
		
		inputdim = X.shape[-1]  # last dimension of X is the number of features
		print("Setting input dim to %d" % inputdim)
		L1_outputdim = 1
		#L2_outputdim
		
		Final_outputdim = Y.shape[-1] # last dimension of X is the number of features
		print("Setting input Final_outputdim to %d" % Final_outputdim)
		
		
		#normalizer = Sequential()
		#normalizer.add(BatchNormalization(input_shape = (X.shape[0], X.shape[1]), axis=1))    
		#X_normalized = normalizer.predict(np.reshape(X.as_matrix(), (1, X.shape[0], X.shape[1])))
		from sklearn.preprocessing import normalize
		X_normalized = normalize (X.as_matrix(), axis=0)  # normalize each feature to be mean 0 and sd =1
		
		#print(X_normalized.shape)
		#print(X[0:10])
		#print(X_normalized[0:10])
			
		# preparing input data
		print("preparing input data for model")
				
		timesteps = 10
		batch_size = 50
				
		X_generator = slidingWindow(X_normalized,timesteps)
		Y_generator = slidingWindow(Y,timesteps)
		X1 = np.zeros((batch_size,timesteps,inputdim))
		Y1 = np.zeros((batch_size,timesteps,Final_outputdim))
					
		print("Shape of input X1: " + str(X1.shape))
		print("Shape of input Y1: " + str(Y1.shape))
			
		#print(X1[0,0:10])
		#print(Y1[0,0:10])
		
		# model without input normalization - does not converge
		if 0:
			self.model = Sequential()
			self.model.add(LSTM(Final_outputdim, input_dim=inputdim, return_sequences = True ))    
			#self.model.add(LSTM(Final_outputdim, input_dim=inputdim, return_sequences = True ))    
			
		if 1:
			self.model = Sequential()
			#self.model.add(BatchNormalization(input_shape = (timesteps, inputdim)))    
			#self.model.add(LSTM(Final_outputdim, return_sequences = True ))    
			self.model.add(LSTM(Final_outputdim, input_dim=inputdim, return_sequences = True ))    
			
		logger.debug("Compiling RecurrentNN")
		self.model.compile(loss='mse', optimizer='rmsprop')
		
		if Y is not None:		
			number_of_batches = 20
			
			logger.debug("Running Fit")
			for i in range(number_of_batches):
				print('Batch #', i, '/', number_of_batches)
				
				print("Generating next batch of samples")
				for i in range(batch_size):
					X1[i] = next(X_generator)
					Y1[i] = next(Y_generator)
					#Y1 = Y1.append(next(Y_generator))#.append(Y_generator)
				
				print("Fitting")
				epochs_on_batch = 20
				self.model.fit(X1,
						  Y1,
						  batch_size=batch_size,
						  verbose=1,
						  nb_epoch=epochs_on_batch,
						  shuffle=False)
				self.model.reset_states()
		
		logger.debug("Done RandomForestRegressor.fit()")
		pass
		
	def predict(self, X):
		logger.debug("Starting RecurrentNN.predict()")
		
		from sklearn.preprocessing import normalize
		X_normalized = normalize (X.as_matrix(), axis=0)  # normalize each feature to be mean 0 and sd =1
		
		X1 = np.reshape(X_normalized, (1, X.shape[0], X.shape[1]))
		Y_hat = self.model.predict(X1)
		
		Y_hat1 = np.reshape(Y_hat, (Y_hat.shape[1], Y_hat.shape[2]))
		logger.debug("Done RecurrentNN.predict()")
		return Y_hat1
		
	def evaluate(self, train_X, train_Y, validation_X, validation_Y):	
		
		res = super(RecurrentNN, self).evaluate(train_X, train_Y, validation_X, validation_Y)
			
		'''	
		logger.info("Feature importances:")
		for importance,feature in sorted(zip(self.model.feature_importances_, train_X.columns), key=lambda x: x[0], reverse=True):
			logger.info("%f %s"%(importance,feature))
		'''
		
		return res

	
logger.info("Class RecurrentNN loaded")