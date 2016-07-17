import sys
from datetime import datetime

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("../")

from utils.utils import *


# Function to read a csv file and resample to hourly consumption
def parse_dmop(filename, dropna=True):
    df = pd.read_csv(config.data_folder + '/' + filename)
    df = convert_time(df)
    df = df.set_index('ut_ms')
    return df


dmop_train1 = parse_dmop('/train_set/context--2008-08-22_2010-07-10--dmop.csv')
dmop_train2 = parse_dmop('/train_set/context--2010-07-10_2012-05-27--dmop.csv')
dmop_train3 = parse_dmop('/train_set/context--2012-05-27_2014-04-14--dmop.csv')

dmop_train = pd.concat([dmop_train1, dmop_train2, dmop_train3])

dmop_test = parse_dmop('/test_set/context--2014-04-14_2016-03-01--dmop.csv')

dmop_all = pd.concat([dmop_train, dmop_test])

dmop_all['subsystem']=dmop_all['subsystem'].map(lambda x:x.split('.')[0])
#dmop_all=dmop_all[dmop_all['subsystem'].str.startswith('A')]
#dmop_all['device']=dmop_all['subsystem'].map(lambda x: x[0:4])
dmop_all['cmd']=dmop_all['subsystem'].map(lambda x: x[4:])

def generate_count(grouper,name):
    print name
    dmop_all['grp'] = dmop_all.index.map(grouper)
    dmop_hour = dmop_all.groupby(['grp','subsystem']).count()
    dmop_hour=dmop_hour.reset_index().pivot(columns='subsystem', index='grp', values='cmd').fillna(0)
    #dmop_hour.set_index('grp')

    dmop_hour['sum']=dmop_hour.sum(axis=1)
    dmop_hour.cols=['dmop_count_'+name+'_'+str(i) for i in dmop_hour.columns]
    target = pd.read_pickle(config.data_folder + '/target.pkl')
    df = target.join(dmop_hour.reindex(target.index,method='nearest'))
    svd_out=df.drop(config.target_cols, axis=1)
    svd_out.fillna(0).to_pickle(config.features_folder+'/dmop_evid_count_'+name+'.pkl')
    #test_df(df)

#generate_count(lambda dt: datetime(dt.year, dt.month, dt.day), '24h')
#generate_count(lambda dt: datetime(dt.year, dt.month, dt.day,dt.hour/3), '3h')
generate_count(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour), '1h')