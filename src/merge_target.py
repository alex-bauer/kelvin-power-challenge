"""Reads all power files and concatenates them into one pandas file"""

import sys

sys.path.append("../")
from utils.utils import *

def parse_power(filename, dropna=True):
    df = pd.read_csv(config.data_folder + '/' + filename)
    df = convert_time(df)
    df = resample(df)
    if dropna:
        df = df.dropna()
    return df

pow_train1 = parse_power('/train_set/power--2008-08-22_2010-07-10.csv')
pow_train2 = parse_power('/train_set/power--2010-07-10_2012-05-27.csv')
pow_train3 = parse_power('/train_set/power--2012-05-27_2014-04-14.csv')

pow_train = pd.concat([pow_train1, pow_train2, pow_train3])

pow_test = parse_power('power-prediction-sample-2014-04-14_2016-03-01.csv', False)

pow_all=pd.concat([pow_train, pow_test])
pow_all.to_pickle(config.data_folder+'/target.pkl')

