import sys
import os

sys.path.append("../")

from utils.utils import *


folder = config.featuresets_folder
if not os.path.exists(folder):
    os.makedirs(folder)



def f12():
    target=pd.read_pickle(config.data_folder+'/target.pkl')
    target['NPWD2851']=target['NPWD2851']+target['NPWD2531']
    target['NPWD2531']=0.0005
    saaf=pd.read_pickle(config.features_folder+'/saaf.pkl')
    dmop_count=pd.read_pickle(config.features_folder+'/dmop_count_1h.pkl')
    ltdata=pd.read_pickle(config.features_folder+'/ltdata.pkl')
    ftlstates=pd.read_pickle(config.features_folder+'/ftl_states.pkl')
    evtf_states=pd.read_pickle(config.features_folder+'/evtf_states.pkl')
    dummies=pd.read_pickle(config.features_folder+'/dummies.pkl')
    time_features=pd.read_pickle(config.features_folder+'/time.pkl')
    df=target.join(saaf)
    df=df.join(dmop_count)
    df=df.join(evtf_states)
    df=df.join(ltdata)
    df=df.join(ftlstates)
    df=df.join(dummies)
    df=df.join(time_features)
    df.to_pickle(config.featuresets_folder+'/f12.pkl')


f12()

print "Done."