"""Generate indicator variables for states in FTL log"""

import sys

sys.path.append("../")

from utils.utils import *
import os


def parse_ftl(filename):
    df = pd.read_csv(config.data_folder + '/' + filename)
    df['utb_ms'] = pd.to_datetime(df['utb_ms'], unit='ms')
    df['ute_ms'] = pd.to_datetime(df['ute_ms'], unit='ms')
    return df


folder = config.features_folder
if not os.path.exists(folder):
    os.makedirs(folder)

ftl_train1 = parse_ftl('/train_set/context--2008-08-22_2010-07-10--ftl.csv')
ftl_train2 = parse_ftl('/train_set/context--2010-07-10_2012-05-27--ftl.csv')
ftl_train3 = parse_ftl('/train_set/context--2012-05-27_2014-04-14--ftl.csv')

ftl_train = pd.concat([ftl_train1, ftl_train2, ftl_train3])

ftl_test = parse_ftl('/test_set/context--2014-04-14_2016-03-01--ftl.csv')

ftl_all = pd.concat([ftl_train, ftl_test])

ftl_all = ftl_all.reset_index()
types = ftl_all['type'].unique()

target = pd.read_pickle(config.data_folder + '/target.pkl')

for type in reversed(types):
    u = ftl_all[ftl_all['type'] == type]
    df = pd.DataFrame(index=target.index)
    name = 'ftl_' + type
    df[name] = 0
    rows = []
    for ix, row in list(u.iterrows()):
        df.loc[row['utb_ms']:row['ute_ms'],name] = 1
    target = target.join(df)
    target[name] = target[name].fillna(0)

name='ftl_comms'
u=ftl_all[ftl_all['flagcomms']==True]
df=pd.DataFrame(index=target.index)
df[name]=0
rows=[]
for ix,row in u.iterrows():
    df.loc[row['utb_ms']:row['ute_ms'],name]=+1

ftl_out = target.drop(config.target_cols, axis=1)
ftl_out.fillna(method='ffill').fillna(method='bfill').to_pickle(config.features_folder + '/ftl_states.pkl')

print "Done."