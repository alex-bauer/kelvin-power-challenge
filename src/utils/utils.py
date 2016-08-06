""" Some basic utility functions"""

import sys
import glob

import pandas as pd
import numpy as np
from config import config

sys.path.append("../")

import os
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

ensure_folder(config.featuresets_folder)
ensure_folder(config.log_folder)
ensure_folder(config.features_folder)
ensure_folder(config.models_folder)

# The evaluation metric
def RMSE(val, pred):
    diff = (val - pred) ** 2
    rmse = np.mean(diff.values) ** 0.5
    return rmse



def convert_time(df):
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
    return df


def resample(df):
    df = df.set_index('ut_ms')
    df = df.resample('1H').mean()
    return df


def read_models_from_dir(dir):
    models = glob.glob(dir + '/*/')

    selected_models = filter(lambda x: 'bag' not in x, models)

    print selected_models
    bagged_oobs = []
    bagged_preds = []

    for model in selected_models:

        pred_file = model + '/' + 'preds.csv'
        oob_file = model + '/' + 'oob.pkl'

        oob = pd.read_pickle(oob_file)
        preds = pd.read_csv(pred_file)
        preds['ut_ms'] = pd.to_datetime(preds['ut_ms'], unit='ms')
        preds=preds.set_index('ut_ms')
        bagged_oobs.append(oob)
        bagged_preds.append(preds)

    return bagged_oobs, bagged_preds, selected_models
