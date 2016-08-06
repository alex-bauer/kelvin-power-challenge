"""Concatenate and resample SAAF data"""

import sys

import os

sys.path.append("../")

from utils.utils import *

folder = config.features_folder
if not os.path.exists(folder):
    os.makedirs(folder)


def parse_saaf(filename, dropna=True):
    df = pd.read_csv(config.data_folder + '/' + filename)
    df = convert_time(df)
    df = resample(df)
    if dropna:
        df = df.dropna()
    return df


saaf_train1 = parse_saaf('/train_set/context--2008-08-22_2010-07-10--saaf.csv')
saaf_train2 = parse_saaf('/train_set/context--2010-07-10_2012-05-27--saaf.csv')
saaf_train3 = parse_saaf('/train_set/context--2012-05-27_2014-04-14--saaf.csv')

saaf_train = pd.concat([saaf_train1, saaf_train2, saaf_train3])

saaf_test = parse_saaf('/test_set/context--2014-04-14_2016-03-01--saaf.csv')

saaf_all = pd.concat([saaf_train, saaf_test])

#Binary indicator for solar conjunction
saaf_all['conjunction']=0
saaf_all.loc['2008-11-16':'2008-12-20','conjunction']=1
saaf_all.loc['2011-01-17':'2011-02-20','conjunction']=1
saaf_all.loc['2013-03-28':'2013-05-05','conjunction']=1
saaf_all.loc['2015-05-27':'2015-07-01','conjunction']=1

cols=['sa','sx','sy','sz']

#Averages over previous hours
for col in cols:
    saaf_all[col+'_last_1']=saaf_all[col].shift(-1)
    saaf_all[col+'_last_3']=saaf_all[col].rolling(3).mean()
    saaf_all[col+'_last_24']=saaf_all[col].rolling(24).mean()


target = pd.read_pickle(config.data_folder + '/target.pkl')
target = target.join(saaf_all.reindex(target.index, method='nearest'))

saaf_all = target.drop(config.target_cols, axis=1)

saaf_all.fillna(method='ffill').fillna(method='bfill').to_pickle(config.features_folder + '/saaf.pkl')

print "Done."