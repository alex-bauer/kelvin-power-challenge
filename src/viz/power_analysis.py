"""Visualizes top 5 power lines by variance"""

import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.append("../")

from utils.utils import *

interval = '1D'

def d_resample(df):
    if interval:
        return df.resample(interval).mean()
    else:
        return df

scale=lambda x: MinMaxScaler().fit_transform(x)

target = d_resample(pd.read_pickle(config.data_folder + '/target.pkl'))


cols=config.target_cols

cols=sorted(cols, key=lambda col:target[col].std(), reverse=True)

fig, axs = plt.subplots(5, 1, sharex=True)
for ix,col in enumerate(cols[0:5]):

    axs[ix].plot(target.index, target[col].values, label=col)

    axs[ix].legend()

plt.show()
