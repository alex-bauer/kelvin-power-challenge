"""
Generates library of models by leaving out 20% of the features
"""


import random

import sys
from datetime import timedelta

sys.path.append("../")
from config import *
from utils.utils import RMSE

import json
import os
import pandas as pd
import numpy as np
from models import train_predict_reg

import logging
import gc

np.random.seed(260580)

def load_data(featureset):
    data = pd.read_pickle(config.featuresets_folder + '/' + featureset + '.pkl')
    train_ix = ~data[config.target_cols[0]].isnull()
    train = data[train_ix]
    test = data[~train_ix]
    return train, test

def get_weights(y):
    return np.ones(len(y))


import multiprocessing as mp

jobs = mp.cpu_count() - 1

def train_model(params):
    folder = config.models_folder + '/random/{}/{}'.format(modelname, iteration)
    if not params.has_key('random_state'):
        params['random_state'] = 0

    model_name = folder.replace('/', '_')

    train, test = load_data(params['featureset'])

    features = list(set(train.columns).difference(config.target_cols))

    features = [f for f in random.sample(features, int(len(features) * 0.8))]

    params['features']=features

    params['n_jobs'] = jobs

    train_logger.info('Training random model: ' + json.dumps(params))

    X = train[features]
    Y = train[config.target_cols]

    losses = []
    oobs = []
    for fold in config.folds:
        tr_index = (train.index < fold[0]) | (train.index > (fold[1] + 2 * timedelta(days=365)))
        val_index = (train.index > fold[0]) & (train.index < fold[1])

        X_tr = X[tr_index]
        Y_tr = Y[tr_index]
        X_val = X[val_index]
        Y_val = Y[val_index]

        model, Y_val_pred, Y_train_prob = train_predict_reg(X_tr, Y_tr, X_val,
                                                            params, weights=get_weights(Y_tr))

        loss = RMSE(Y_val, Y_val_pred)
        train_logger.info('Fold completed, rmse: {}'.format(loss))
        losses.append(loss)

        oob = pd.DataFrame(Y_val_pred, index=X_val.index, columns=config.target_cols)
        oobs.append(oob)

        gc.collect()

    X_test = test[features]
    model, Y_test_prob, Y_train_prob = train_predict_reg(X, Y, X_test,
                                                         params, weights=get_weights(Y))

    if not os.path.exists(folder):
        os.makedirs(folder)

    oobs = pd.concat(oobs)
    oobs.to_pickle(folder + '/oob.pkl')
    params['loss'] = np.mean(losses)
    params['losses'] = losses
    json.dump(params, open(folder + '/params.json', 'w'))

    Y_test_prob = pd.DataFrame(Y_test_prob, index=test.index, columns=config.target_cols)

    Y_test_prob['ut_ms'] = (Y_test_prob.index.astype(np.int64) * 1e-6).astype(int)

    Y_test_prob[['ut_ms'] + config.target_cols].to_csv(folder + '/preds.csv', index=False)
    Y_test_prob[['ut_ms'] + config.target_cols].to_csv(folder + '/' + model_name.replace('.', '') + '_sub.csv',
                                                       index=False)

    rand_logger.info('{},{},{}'.format(str(np.mean(losses)), iteration, json.dumps(params)))


if __name__ == "__main__":

    train_logger = logging.getLogger('training')
    train_logger.setLevel(logging.DEBUG)
    train_logger.addHandler(logging.FileHandler(config.log_folder + '/training.log'))
    train_logger.addHandler(logging.StreamHandler())

    rand_logger = logging.getLogger('rand')
    rand_logger.setLevel(logging.DEBUG)
    rand_logger.addHandler(logging.FileHandler(config.log_folder + '/random_fe.log'))
    rand_logger.addHandler(logging.StreamHandler())


    for i in range(1, len(sys.argv)):
        config_name = sys.argv[i]
        params = json.load(open(config.model_config_folder + '/level1/' + config_name + '.json'))
        modelname = params['modelname']
        for iteration in range(0,100):
            train_model(params)
