"""
ESA Kelvins - Mars Explorer Power Prediction Challenge

Baseline model (predicting the mean)

Author: Alexander Bauer (email@alexander-bauer.com)

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define the path to the unzipped data files
path_to_data='../data'


#Function to convert the utc timestamp to datetime
def convert_time(df):
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
    return df

#Function to resample the dataframe to hourly mean
def resample(df):
    df = df.set_index('ut_ms')
    df = df.resample('1H').mean()
    return df

#Function to read a csv file and resample to hourly consumption
def parse_power(filename, dropna=True):
    df = pd.read_csv(path_to_data + '/' + filename)
    df = convert_time(df)
    df = resample(df)
    if dropna:
        df = df.dropna()
    df['mars_time_of_year'] = df.index - df.index.min()
    return df


##Load the training power files
pow_train1 = parse_power('/train_set/power--2008-08-22_2010-07-10.csv')
pow_train2 = parse_power('/train_set/power--2010-07-10_2012-05-27.csv')
pow_train3 = parse_power('/train_set/power--2012-05-27_2014-04-14.csv')

#Concatenate the files
pow_train = pd.concat([pow_train1, pow_train2, pow_train3])

#Load the test sample submission file as template for prediction
pow_test = parse_power('power-prediction-sample-2014-04-14_2016-03-01.csv', False)

# Extract the columns that need to be predicted
p_cols = list(pow_test.columns)
p_cols.remove('mars_time_of_year')

# Let's have a look at the data, power values aggregated by day
#pow_train[p_cols].resample('1D').mean().plot(legend=False)
#plt.show()

# Defining the evaluation metric
def RMSE(val, pred):
    diff = (val - pred) ** 2
    rmse = np.mean(diff.values) ** 0.5
    return rmse


# Let's do some local validation of the model, take first two marsian years as train, and third year as validation set
pow_train = pd.concat([pow_train1, pow_train2])
pow_val = pow_train3

# The most simple approach is just to predict the mean per column
pow_pred = pow_train3.copy()
for p in p_cols:
    pow_pred[p] = pow_train[p].mean()
print 'Baseline mean prediction,  RMSE: ', RMSE(pow_val[p_cols], pow_pred[p_cols])

# 0.1278 not so bad for a start, let's predict for test set and prepare the submission
pow_train = pd.concat([pow_train1, pow_train2, pow_train2])

# Let's take the submission file as a template and put the trainset mean in each column
pow_pred = pow_test.copy()
for p in p_cols:
    pow_pred[p] = pow_train[p].mean()

#We need to convert the parsed data back to utc timestamp
pow_pred['ut_ms'] = (pow_pred.index.astype(np.int64) * 1e-6).astype(int)

#Prepare the submission file
pow_pred[['ut_ms']+p_cols].to_csv('mean_baseline.csv', index=False)

#Goto https://kelvins.esa.int/mars-express-power-challenge/ , create an account and submit the file!
