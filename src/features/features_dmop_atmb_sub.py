
import sys
import os
from datetime import datetime

sys.path.append("../")
from utils.utils import *

folder = config.features_folder
if not os.path.exists(folder):
    os.makedirs(folder)


# Function to read a csv file and resample to hourly
def parse_dmop(filename):
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


# Split numeric values and subsystem
dmop_split_df= dmop_all.subsystem.str.split('.',expand=True).rename(columns={0:'description',1:"param"})
# split the subsystem/command pattern
dmop_split_df['subsystem'] = dmop_split_df.description.str[0:4]
dmop_split_df['command'] = dmop_split_df.description.str[4:]

#
# ATMB - Subsystem
# ----------------

# Commands for atmb
ATMB_commands = dmop_split_df[(dmop_split_df.subsystem == "ATMB")].command.value_counts()

# Extract the "kelvins" per event
ATMB_df_events = dmop_split_df[dmop_split_df['subsystem']=='ATMB']
ATMB_df_events.loc[:,'kelvins'] = ATMB_df_events.command.str[0:3].astype(int)

# feature sampled by hour 
ATMB_df_events.kelvins.resample('h').ffill().fillna(method='bfill').to_pickle(config.features_folder + "ATMB_commands.pkl")
ATMB_df_events.kelvins.resample('h').ffill().fillna(method='bfill').to_csv(config.features_folder + "ATMB_commands.csv")