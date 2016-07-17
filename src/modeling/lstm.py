# LSTM
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM as kerasLSTM
import math
#, BatchNormalization

import pandas as pd  
from random import random   

class LSTM:
    def __init__(self,params):
        self.params = params
        pass
    """    
    def build_batch(self,data,batch_size,sequence_length,index):
        if index>=sequence_length:
            return data.iloc[index-sequence_length:index]
    def generator(self,batch_size,sequence_length,data,shuffle=True):
        num_batches = math.floor(data.shape[0] / batch_size)
        
        
        while True:
            order =  np.argsort(np.random.rand(10))
            for b in range(num_batches):
                yield self.build_batch(data,batch_size,sequence_length,[order[b]])
    """
    
    """
    def build_batches(self,data,batch_size,sequence_length):
        num_batches = int(math.floor(data.shape[0] / batch_size))
        #print "num batches:",num_batches
        batches = [] 
        batch = np.zeros((batch_size,sequence_length,data.shape[1]))
        for nb in range(num_batches):
            for ex in range(batch_size):
                index = nb * batch_size + ex
                tmp = data.iloc[index:index+sequence_length,:].as_matrix()
                batch[ex,:,:] = tmp
                #for s in range(sequence_length):
                #    batches[ex,:,:] = data[]
            batches.append(batch)
        return batches
    """            
    def prepare_data(self,data,sequence_length):
        
        #prepared_data = np.zeros((data.shape[0],sequence_length,data.shape[1]))
        prepared_data = []
        #print "data.shape[0] -> %d"%data.shape[0]
        for ex in range(data.shape[0]-sequence_length+1):
            prepared_data.append(data.iloc[ex:ex+sequence_length].as_matrix())
        print "last ex %d"%range(data.shape[0]-sequence_length)[-1]
        i=1
        for ex in range(data.shape[0]-sequence_length+1,data.shape[0]):
            #print "ex: %d"%ex
            tmp = np.zeros((sequence_length,data.shape[1]))
            
            #print "data.iloc[%d:,:]"%(ex)
            #print "tmp[%d:,:]"%(i)
            dt = data.iloc[ex:,:].as_matrix()
            #print "dt.shape",dt.shape
            tmp[i:,:] = dt
            prepared_data.append(tmp)
            i += 1

        return np.array(prepared_data)
        
    
    def fit(self, X_train, Y_train):
        num_units = self.params["num_units"]
        sequence_length = self.params['sequence_length']
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]
        batch_size = self.params['batch_size']
        
        print "SPECS:"
        print " num_units (LSTM)",num_units
        print " sequence_length",sequence_length
        print " input_dim (X)",input_dim
        print " output_dim (Y)",output_dim
        print " batch_size",batch_size
        
        dataX = self.prepare_data(X_train,sequence_length)
        #batchesY = self.prepare_data(Y_train,sequence_length)
        #dataY = Y_train.iloc[:dataX.shape[0]].as_matrix()
        dataY = Y_train.as_matrix()
        
        #print " x shape ",dataX.shape
        #print " y shape ",dataY.shape
        
        print('Build model...')
        self.model = Sequential()
        self.model.add(kerasLSTM(num_units, return_sequences=True, input_shape=(sequence_length, input_dim)))
        self.model.add(kerasLSTM(num_units, return_sequences=False))
        self.model.add(Dense(output_dim,activation="linear"))  
        #self.model.add(Activation())  
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop")  
        self.model.fit(dataX, dataY, batch_size=batch_size, nb_epoch=self.params['n_epochs'], validation_split=0)

    def predict(self,X_valid):
        dataX = self.prepare_data(X_valid,self.params['sequence_length'])
        print " x shape ",dataX.shape
        return self.model.predict(dataX,batch_size = self.params['batch_size'],verbose=1)
