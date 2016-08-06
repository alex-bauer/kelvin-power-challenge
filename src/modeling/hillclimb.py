"""
Implements Caruana's greedy ensemble selection methods

Caruana et al., Ensemble selection from a library of models, Proceedings of ICML, 2004
http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
"""

import random

import sys

sys.path.append("../")
from config import *
from utils.utils import RMSE

import json
import os
import pandas as pd
import numpy as np

import glob

cfg_init_models = 5
cfg_n_iter = 20
cfg_n_bags = 20


class Model():
    def __init__(self, name, oob, preds, rmse):
        self.name = name
        self.oob = oob
        self.preds = preds
        self.rmse = rmse


from copy import copy


class HillClimb():
    def __init__(self, model):
        self.models = [model]
        self.preds = model.preds
        self.oob = model.oob
        self.update_rmse()

    def update_rmse(self):
        self.tr_rmse = RMSE(hillclimb_train, self.oob.loc[hillclimb_train.index])
        self.val_rmse = RMSE(hillclimb_val, self.oob.loc[hillclimb_val.index])

    def add_model(self, model):
        new_climb = copy(self)
        new_climb.models = list(self.models)
        new_climb.preds = (len(self.models) * self.preds + model.preds) / (len(self.models) + 1)
        new_climb.oob = (len(self.models) * self.oob + model.oob) / (len(self.models) + 1)
        new_climb.models.append(model)
        new_climb.update_rmse()
        return new_climb

    def add_models(self, models):
        new_climb = self
        for model in models:
            new_climb = self.add_model(model)
        return new_climb

    def mean_rmse(self):
        return (self.tr_rmse + self.val_rmse) / 2.0

    @classmethod
    def from_models(cls, models):
        climb = cls(models[0])
        for model in models[1:]:
            climb = climb.add_model(model)
        return climb


def read_models_from_dir(dir):
    model_array = []

    models = glob.glob(dir + '/*/')

    selected_models = filter(lambda x: 'bag' not in x, models)

    print selected_models

    for model in selected_models:
        try:
            pred_file = model + '/' + 'preds.csv'
            oob_file = model + '/' + 'oob.pkl'

            oob = pd.read_pickle(oob_file)
            cols = [model + str(i) for i in oob.columns]
            print model, oob.shape
            preds = pd.read_csv(pred_file)
            preds['ut_ms'] = pd.to_datetime(preds['ut_ms'], unit='ms')
            preds = preds.set_index('ut_ms')
            model_array.append((Model(model, oob, preds, RMSE(target.loc[oob.index], oob))))
        except:
            print "Error! ", model
            pass
    return model_array


target = pd.read_pickle(config.data_folder + '/target.pkl')
samples = len(target) / 2
models = read_models_from_dir(config.models_folder + '/random/*/')

target = target.loc[models[0].oob.index]
samples = len(target) / 2

hillclimb_train = target.iloc[samples:]
hillclimb_val = target.iloc[0:samples]

## Initialize Hillclimb
best_models = sorted(models, key=lambda model: model.rmse)

climb = HillClimb(best_models[0])
print "Best model: {}, {}, {}, {}".format([model.name for model in climb.models], climb.tr_rmse, climb.val_rmse,
                                          climb.mean_rmse())

for k in range(1, cfg_init_models):
    climb = climb.add_model(best_models[k])

print "Init: {}, {}, {}, {}".format([model.name for model in climb.models], climb.tr_rmse, climb.val_rmse,
                                    climb.mean_rmse())

final_models = []
for n in range(cfg_n_bags):
    print "Bag iteration ", n
    submodels = random.sample(models, int(len(models) * 0.5))
    ## Initialize Hillclimb
    best_models = sorted(submodels, key=lambda model: model.rmse)
    climb = HillClimb.from_models(best_models[0:cfg_init_models])
    for k in range(cfg_n_iter):
        scores = [climb.add_model(model).tr_rmse for model in submodels]
        if min(scores) < climb.tr_rmse:
            new_model = submodels[np.argmin(scores)]
            new_climb = climb.add_model(new_model)
            #print np.min(scores), new_climb.tr_rmse
            # if new_climb.mean_rmse() > climb.mean_rmse(): break
            climb = new_climb
            print "{}, new score: {}, {}, {}".format(new_model.name, climb.tr_rmse, climb.val_rmse, climb.mean_rmse())
        else:
            break
    final_models.extend(climb.models)

print final_models
climb = HillClimb.from_models(final_models)

print "Final score: {}, {}, {}".format(climb.tr_rmse, climb.val_rmse, climb.mean_rmse())
print [model.name for model in climb.models]

folder = config.models_folder + '/hillclimbs/' + datetime.now().strftime("%m-%d_%H")

if not os.path.exists(folder):
    os.makedirs(folder)

params = {}

params['models'] = [model.name for model in climb.models]
params['loss'] = climb.mean_rmse()

json.dump(params, open(folder + '/params.json', 'w'))

Y_test_prob = climb.preds
Y_test_prob['ut_ms'] = (Y_test_prob.index.astype(np.int64) * 1e-6).astype(int)

Y_test_prob[['ut_ms'] + config.target_cols].to_csv(
    folder + '/hillclimb_' + datetime.now().strftime("%m-%d_%H") + '.csv', index=False)

print "Hello"
