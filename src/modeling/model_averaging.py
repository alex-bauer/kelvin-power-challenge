from datetime import datetime
import json
import os

import pandas as pd
import glob as glob
import numpy as np
import sys

sys.path.append("../")
from config import *

from config import config

from utils.utils import RMSE, read_models_from_dir


target=pd.read_pickle(config.data_folder+'/target.pkl')


def bag_models(folder):
    bagged_oobs, bagged_preds, selected_models = read_models_from_dir(folder)


    new_oob, new_preds= bagged_oobs[0].copy(), bagged_preds[0].copy()

    for col in config.target_cols:
        new_oob[col]=np.sum(np.array([b[col] for b in bagged_oobs]), axis=0)/len(bagged_oobs)
        new_preds[col]=np.sum([b[col] for b in bagged_preds], axis=0)/len(bagged_oobs)

    new_oob.to_pickle(folder + '/oob.pkl')

    toob = target.loc[new_oob.index]
    loss=RMSE(toob,new_oob)

    new_preds=pd.DataFrame(new_preds.values, index=new_preds.index, columns=config.target_cols)

    new_preds['ut_ms'] = (new_preds.index.astype(np.int64) * 1e-6).astype(int)

    new_preds[['ut_ms']+config.target_cols].to_csv(folder+'/preds.csv', index=False)

    params = {}
    params['loss_cv_mean'] = loss
    params['models'] = selected_models

    print params

    json.dump(params, open(folder + '/params.json', 'w'))


if __name__ == "__main__":
    folder = sys.argv[1]
    bag_models(folder)