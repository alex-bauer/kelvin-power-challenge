import operator
import sys
sys.path.append("../")

import numpy as np

import xgboost as xgb

from keras import callbacks
from keras.layers import TimeDistributed
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from utils.utils import *

def slidingWindow(X, windowSize=10, numWindows=-1):
    if numWindows == -1:
        numWindows = len(X)
    print("Generating %d windows" % (numWindows/windowSize))
    i = 0
    while i < numWindows:
        yield list(X.iloc[i:i + windowSize].values)
        i += windowSize


def train_predict_reg(X_train, y_train, X_val, params, weights=None):
    if weights is None:
        weights = np.ones(len(y_train))

    features = list(X_train.columns)
    X_val = X_val.copy()
    X_train = X_train.copy()

    if 'reg_skl_etr' in params["model"]:
        X_train = X_train.fillna(-999.0)
        X_val = X_val.fillna(-999.0)

        X_train = X_train.replace(-np.inf, -10000)
        X_train = X_train.replace(np.inf, 10000)
        X_val = X_val.replace(-np.inf, -10000)
        X_val = X_val.replace(np.inf, 10000)

        clf = ExtraTreesRegressor(n_estimators=int(params['n_estimators']),
                                  min_samples_leaf=max(1, int(params['min_samples_leaf'])),

                                  max_features=params['max_features'],
                                  max_depth=None if not params.has_key('max_depth') else int(params['max_depth']),
                                  random_state=params['random_state'], n_jobs=params['n_jobs'])

        clf.fit(X_train, y_train)
        features = list(X_train.columns)
        print sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True)
        y_val_prob = clf.predict(X_val)
        return clf, y_val_prob, None

    if 'reg_skl_rfr' in params["model"]:
        X_train = X_train.fillna(-999.0)
        X_val = X_val.fillna(-999.0)

        X_train = X_train.replace(-np.inf, -10000)
        X_train = X_train.replace(np.inf, 10000)
        X_val = X_val.replace(-np.inf, -10000)
        X_val = X_val.replace(np.inf, 10000)

        clf = RandomForestRegressor(n_estimators=int(params['n_estimators']),
                                    min_samples_leaf=max(1, int(params['min_samples_leaf'])),
                                    max_features=params['max_features'],
                                    max_depth=None if not params.has_key('max_depth') else int(params['max_depth']),
                                    random_state=params['random_state'], n_jobs=params['n_jobs'])

        clf.fit(X_train, y_train)
        features = list(X_train.columns)

        print sorted(zip(features, clf.feature_importances_), key=lambda x: x[1], reverse=True)
        y_val_prob = clf.predict(X_val)
        return clf, y_val_prob, None

    if params["model"] == 'reg_keras_dnn':
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.mean())
        X_val = X_val.fillna(X_val.mean())
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        y_scaler = StandardScaler(with_std=False)
        y_train = y_scaler.fit_transform(y_train)

        model = Sequential()
        # ## input layer
        model.add(Dropout(params["input_dropout"], input_shape=[X_train.shape[1]]))

        hidden_layers = params['hidden_layers']
        units = params["hidden_units"]

        while hidden_layers > 0:
            model.add(Dense(units, init='glorot_uniform'))
            if params["batch_norm"]:
                model.add(BatchNormalization())
            if params["hidden_activation"] == "prelu":
                model.add(PReLU())
            else:
                model.add(Activation(params['hidden_activation']))
            model.add(Dropout(params["hidden_dropout"]))
            hidden_layers -= 1

        model.add(Dense(33, init='glorot_uniform', activation='sigmoid'))
        # ## output layer
        model.add(Dense(33, init='glorot_uniform', activation='linear'))

        model.compile(loss='mean_squared_error', optimizer='adam')

        ## to array
        X_train_ndarray = X_train
        y_train_ndarray = y_train
        X_val_ndarray = X_val

        X_es_train, X_es_eval, y_es_train, y_es_eval = train_test_split(X_train, y_train,
                                                                        test_size=0.1,
                                                                        random_state=0)

        if params['early_stopping']:
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')
            ## train
            model.fit(X_es_train, y_es_train,
                      nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], callbacks=[earlyStopping],
                      validation_data=[X_es_eval, y_es_eval], verbose=2)
        else:
            model.fit(X_train_ndarray, y_train_ndarray,
                      nb_epoch=params['nb_epoch'], batch_size=params['batch_size'],
                      validation_split=0.1, verbose=2)
            ##prediction
        pred = model.predict(X_val_ndarray, verbose=0)
        pred = y_scaler.inverse_transform(pred)

        return model, pred, None

    if params["model"] == 'reg_keras_lstm':
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform((X_train.fillna(0).values)), columns=X_train.columns,
                               index=X_train.index)

        X_val = pd.DataFrame(scaler.transform(X_val.fillna(0).values), columns=X_val.columns, index=X_val.index)

        num_units = params["hidden_units"]
        sequence_length = params['sequence_length']
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        batch_size = params['batch_size']
        backwards = params['backwards'] if 'backwards' in params else False

        print "SPECS:"
        print " num_units (LSTM)", num_units
        print " sequence_length", sequence_length
        print " input_dim (X)", input_dim
        print " output_dim (Y)", output_dim
        print " batch_size", batch_size

        print "X_train len", len(X_train)

        start = len(X_train.index) % (batch_size * sequence_length)

        X_train_Window_Generator = slidingWindow(X_train.iloc[start:], sequence_length)  # , 10, 1)
        Y_train_Window_Generator = slidingWindow(y_train.iloc[start:], sequence_length)  # , 10, 1)

        print('Build model...')

        model = Sequential()
        model.add(LSTM(num_units, batch_input_shape=(batch_size, sequence_length, input_dim), return_sequences=True,
                       stateful=True, go_backwards=backwards))

        if "2-lstm" in params:
            model.add(TimeDistributed(Dense(num_units, activation='relu')))
            model.add(LSTM(num_units, batch_input_shape=(batch_size, sequence_length, input_dim), return_sequences=True,
                       stateful=True, go_backwards=backwards))
        model.add(TimeDistributed(Dense(num_units, activation='relu')))
        model.add(Dropout(params['hidden_dropout']))
        model.add(TimeDistributed(Dense(32, activation='sigmoid')))
        model.add(TimeDistributed(Dense(output_dim, activation='linear')))

        model.compile(loss='mse', optimizer='rmsprop')

        X_seq = list(X_train_Window_Generator)
        Y_seq = list(Y_train_Window_Generator)

        if backwards:
            X_seq.reverse()
            Y_seq.reverse()

        model.fit(X_seq, Y_seq,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=params['nb_epoch'],
                  shuffle=False)

        model = model

        batch_size = params['batch_size']
        sequence_length = params['sequence_length']
        # merge the train and the test

        X_merged = pd.concat([X_train, X_val])
        print len(X_merged.index)

        start = len(X_merged.index) % (batch_size * sequence_length)

        X_train_Window_Generator = slidingWindow(X_merged.iloc[start:], sequence_length)  # , 10, 1)
        dataX = list(X_train_Window_Generator)
        if backwards:
            dataX.reverse()
        Y_hat = model.predict(dataX, batch_size=batch_size, verbose=1)

        # now get the tail of Y_hat
        Y_hat1 = np.vstack(Y_hat)
        if backwards:
            Y_hat1=Y_hat1[::-1,:]

        res = Y_hat1[-len(X_val.index):, :]

        return None, res, None

    if params["model"] in ['reg_xgb_tree']:

        X_trainsets = []
        y_train_sets = []
        X_testssets = []

        for ix, col in enumerate(config.target_cols):
            X_train_col = X_train.copy()
            X_test_col = X_val.copy()
            X_train_col['out'] = ix
            X_test_col['out'] = ix
            X_testssets.append(X_test_col)
            X_trainsets.append(X_train_col)
            y_train_sets.append(y_train[col].values)

        X_train = pd.concat(X_trainsets)
        X_val = pd.concat(X_testssets)
        y_train = np.concatenate(y_train_sets)

        X_train_xgb = X_train.fillna(-999.0)
        X_val_xgb = X_val.fillna(-999.0)

        params['num_round'] = max(params['num_round'], 10)

        params['nthread'] = params['n_jobs']
        params['seed'] = params['random_state']

        X_es_train, X_es_eval, y_es_train, y_es_eval = train_test_split(X_train_xgb, y_train,
                                                                        test_size=0.2,
                                                                        random_state=0)

        dvalid_base = xgb.DMatrix(X_es_eval, label=y_es_eval, feature_names=list(X_es_eval.columns))
        dtrain_base = xgb.DMatrix(X_es_train, label=y_es_train,
                                  feature_names=list(X_es_eval.columns))
        dtest_base = xgb.DMatrix(X_val_xgb, feature_names=list(X_es_eval.columns))

        watchlist = [(dtrain_base, 'train'), (dvalid_base, 'valid')]

        if params['early_stopping'] == True:
            model = xgb.train(params, dtrain_base, int(params['num_round']), watchlist, early_stopping_rounds=20)
        else:
            model = xgb.train(params, dtrain_base, int(params['num_round']), watchlist)

        importance = model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))

        print importance

        y_val_prob = model.predict(dtest_base)

        y_val_prob = y_val_prob.reshape((len(config.target_cols), -1)).T

        y_train_prob = None

        return model, y_val_prob, y_train_prob
