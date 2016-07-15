"""Extract and interpolate states from EVTF log"""

import sys

sys.path.append("../")

from utils.utils import *


# Function to read a csv file and resample to hourly consumption
def parse_evtf(filename):
    df = pd.read_csv(config.data_folder + '/' + filename)
    df = convert_time(df)
    df = df.set_index('ut_ms')
    return df


evtf_train1 = parse_evtf('/train_set/context--2008-08-22_2010-07-10--evtf.csv')
evtf_train2 = parse_evtf('/train_set/context--2010-07-10_2012-05-27--evtf.csv')
evtf_train3 = parse_evtf('/train_set/context--2012-05-27_2014-04-14--evtf.csv')

evtf_train = pd.concat([evtf_train1, evtf_train2, evtf_train3])

evtf_test = parse_evtf('/test_set/context--2014-04-14_2016-03-01--evtf.csv')

evtf_all = pd.concat([evtf_train, evtf_test])

target = pd.read_pickle(config.data_folder + '/target.pkl')


def state_vector(start_ev, end_ev, fieldname):
    u = evtf_all[evtf_all['description'].isin([start_ev, end_ev])].copy()
    u[u['description'] == start_ev] = 1
    u[u['description'] == end_ev] = -1
    u['description'] = u['description'].astype(int)
    u = u.resample('60s').mean().fillna(0).cumsum().resample('1H').mean()
    u = u.rename(columns={'description': fieldname})
    return u

target = target.join(state_vector('MAR_UMBRA_START', 'MAR_UMBRA_END', 'evtf_umbra'))
target = target.join(state_vector('MAR_PENUMBRA_START', 'MAR_PENUMBRA_END', 'evtf_penumbra'))

#target['penumbra_period'] = pd.rolling_sum(target['evtf_penumbra'],20)>0.1

import re
def height_vector(regex, fieldname):
    heights1=evtf_all['description'].str.match(r'^(\d+)_KM').map(lambda x:x[0] if len(x)>0 else 0).astype(int).copy()
    heights2=evtf_all['description'].str.contains('APOCENTRE').map(lambda x: 10107 if x else 0).astype(int).copy()
    heights3=evtf_all['description'].str.contains('PERICENTRE').map(lambda x: 298 if x else 0).astype(int).copy()
    heights=heights1+heights2+heights3
    u=heights[heights>0]
    u=u.resample('60s').interpolate().resample('1H').mean()/10000.0
    #u = u.resample('1H').mean().fillna(0).cumsum().resample('1H').mean()
    u=u.to_frame(fieldname)
    return u

target = target.join(height_vector(r'(\d+)_KM_(.+)','evtf_height'))

evtf_out=target.drop(config.target_cols, axis=1)
evtf_out.fillna(method='ffill').fillna(method='bfill').to_pickle(config.features_folder+'/evtf_states.pkl')

test_df(target)



print "hello"
