import sys

sys.path.append("../")

from utils.utils import *

target = pd.read_pickle(config.data_folder + '/target.pkl')

#Dummies for 2551 modes (fourth year modes are inferred from inspecting DMOP commands)
target['dummy_2551_mode']=0
target.loc['2008-08-25':'2009-02-08','dummy_2551_mode']=1
target.loc['2009-05-02':'2009-10-18','dummy_2551_mode']=1
target.loc['2010-05-02':'2010-09-19','dummy_2551_mode']=1
target.loc['2011-03-06':'2011-06-27','dummy_2551_mode']=1
target.loc['2011-12-12':'2012-04-29','dummy_2551_mode']=1
target.loc['2012-11-12':'2013-03-01','dummy_2551_mode']=1
target.loc['2014-07-08':'2014-11-10','dummy_2551_mode']=1


ltdata_out=target.drop(config.target_cols, axis=1)
ltdata_out.to_pickle(config.features_folder+'/dummies.pkl')

print "Done."