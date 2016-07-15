"""Concatenate and resample LTDATA"""

import sys

sys.path.append("../")

from utils.utils import *
import os

folder=config.features_folder
if not os.path.exists(folder):
    os.makedirs(folder)


# Function to read a csv file and resample to hourly consumption
def parse_ltdata(filename, dropna=True):
    df = pd.read_csv(config.data_folder + '/' + filename)
    df = convert_time(df)
    df = df.set_index('ut_ms')
    return df


ltdata_train1 = parse_ltdata('/train_set/context--2008-08-22_2010-07-10--ltdata.csv')
ltdata_train2 = parse_ltdata('/train_set/context--2010-07-10_2012-05-27--ltdata.csv')
ltdata_train3 = parse_ltdata('/train_set/context--2012-05-27_2014-04-14--ltdata.csv')

ltdata_train = pd.concat([ltdata_train1, ltdata_train2, ltdata_train3])

ltdata_test = parse_ltdata('/test_set/context--2014-04-14_2016-03-01--ltdata.csv')

ltdata_all = pd.concat([ltdata_train, ltdata_test])

ltdata_all.columns=['ltdata_'+col for col in ltdata_all.columns]
target = pd.read_pickle(config.data_folder + '/target.pkl')

target = target.join(ltdata_all.reindex(target.index,method='nearest'))

ltdata_out=target.drop(config.target_cols, axis=1)
ltdata_out.fillna(method='ffill').fillna(method='bfill').to_pickle(config.features_folder+'/ltdata.pkl')

