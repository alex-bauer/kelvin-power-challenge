# Example class for a randomforsest-predictor

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Activation, GRU, TimeDistributed
from keras.optimizers import SGD


def slidingWindow(X, windowSize=15, numWindows=-1):
    if numWindows == -1:
        numWindows = len(X) - windowSize
    i = 0
    while i <= numWindows:
        yield list(np.array(X[i:i + windowSize]))
        i += windowSize


X = np.zeros((100, 2))
Y = np.zeros((100, 1))

X[5, 0] = 1
X[14, 1] = 1
Y[5:15,0] = 1

batch_size = 64
timesteps = 25
X=np.vstack([X]*batch_size*timesteps)
Y=np.vstack([Y]*batch_size*timesteps)
import matplotlib.pyplot as plt

#plt.plot(X)
#plt.plot(Y)
#plt.show()

print("Shape of input X: " + str(X.shape))
if Y is not None:
    print("Shape of expected output Y: " + str(Y.shape))

inputdim = X.shape[-1]  # last dimension of X is the number of features
print("Setting input dim to %d" % inputdim)
L1_outputdim = 1
# L2_outputdim

Final_outputdim = Y.shape[-1]  # last dimension of X is the number of features
print("Setting input Final_outputdim to %d" % Final_outputdim)

from sklearn.preprocessing import StandardScaler

X_normalized =  StandardScaler().fit_transform(X) # normalize each feature to be mean 0 and sd =1

print("preparing input data for model")

X_generator = slidingWindow(X_normalized, timesteps)
Y_generator = slidingWindow(Y, timesteps)

# model without input normalization - does not converge
model = Sequential()
model.add(GRU(32, batch_input_shape=(batch_size,timesteps,2), return_sequences=True, stateful=True))
model.add(TimeDistributed(Dense(32, activation='sigmoid')))
model.add(TimeDistributed(Dense(1, activation='linear')))
#model.add(Activation('sigmoid'))

# self.model.add(LSTM(Final_outputdim, input_dim=inputdim, return_sequences = True ))

model.compile(loss='mse', optimizer='rmsprop')

X_seq=list(X_generator)
Y_seq=list(Y_generator)
model.fit(X_seq,Y_seq,
               batch_size=batch_size,
               verbose=1,
               nb_epoch=1,
               shuffle=False)

print np.array(X_seq).shape
Y_hat = np.vstack(model.predict(X_seq, batch_size=batch_size))
plt.plot(Y)
plt.plot(Y_hat)
plt.show()
