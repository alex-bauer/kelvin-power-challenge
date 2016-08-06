"""
Calculate days since 2008 to capture trends
"""
import sys
from datetime import datetime

sys.path.append("../")

from utils.utils import *

target = pd.read_pickle(config.data_folder + '/target.pkl')

target['days_since_2008']=target.index.map(lambda x:(x-datetime(2008,1,1)).days)

ltdata_out=target.drop(config.target_cols, axis=1)
ltdata_out.fillna(method='ffill').fillna(method='bfill').to_pickle(config.features_folder+'/time.pkl')

print "Done."