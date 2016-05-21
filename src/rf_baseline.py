"""
ESA Kelvins - Mars Explorer Power Prediction Challenge

Random Forest baseline model (using saaf and ltdata)

Author: Alexander Bauer (email@alexander-bauer.com)

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the path to the unzipped data files
from sklearn.ensemble import RandomForestRegressor

path_to_data = './data'


# Function to convert the utc timestamp to datetime
def convert_time(df):
    df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
    return df


# Function to resample the dataframe to hourly mean
def resample_1H(df):
    df = df.set_index('ut_ms')
    df = df.resample('1H').mean()
    return df


# Function to read a csv file and resample to hourly consumption
def parse_ts(filename, dropna=True):
    df = pd.read_csv(path_to_data + '/' + filename)
    df = convert_time(df)
    df = resample_1H(df)
    if dropna:
        df = df.dropna()
    return df


# Function to read the ltdata files
def parse_ltdata(filename):
    df = pd.read_csv(path_to_data + '/' + filename)
    df = convert_time(df)
    df = df.set_index('ut_ms')
    return df


##Load the power files
pow_train1 = parse_ts('/train_set/power--2008-08-22_2010-07-10.csv')
pow_train2 = parse_ts('/train_set/power--2010-07-10_2012-05-27.csv')
pow_train3 = parse_ts('/train_set/power--2012-05-27_2014-04-14.csv')
# Load the test sample submission file as template for prediction
pow_test = parse_ts('power-prediction-sample-2014-04-14_2016-03-01.csv', False)
# Concatenate the files
power_all = pd.concat([pow_train1, pow_train2, pow_train3, pow_test])

# Same for the saaf files
saaf_train1 = parse_ts('/train_set/context--2008-08-22_2010-07-10--saaf.csv')
saaf_train2 = parse_ts('/train_set/context--2010-07-10_2012-05-27--saaf.csv')
saaf_train3 = parse_ts('/train_set/context--2012-05-27_2014-04-14--saaf.csv')
saaf_test = parse_ts('/test_set/context--2014-04-14_2016-03-01--saaf.csv')
saaf_all = pd.concat([saaf_train1, saaf_train2, saaf_train3, saaf_test])

# Load the ltdata files
ltdata_train1 = parse_ltdata('/train_set/context--2008-08-22_2010-07-10--ltdata.csv')
ltdata_train2 = parse_ltdata('/train_set/context--2010-07-10_2012-05-27--ltdata.csv')
ltdata_train3 = parse_ltdata('/train_set/context--2012-05-27_2014-04-14--ltdata.csv')
ltdata_test = parse_ltdata('/test_set/context--2014-04-14_2016-03-01--ltdata.csv')
ltdata_all = pd.concat([ltdata_train1, ltdata_train2, ltdata_train3, ltdata_test])

# Extract the columns that need to be predicted
power_cols = list(power_all.columns)

# Now let's join everything together
df = power_all

# Make sure that saaf has the same sampling as the power, fill gaps with nearest value
saaf_all = saaf_all.reindex(df.index, method='nearest')
ltdata_all = ltdata_all.reindex(df.index, method='nearest')
df = df.join(saaf_all)
df = df.join(ltdata_all)

# Now we formulate the prediction problem X -> Y
# Y is the matrix that we want to predict
# X is everything else
Y = df[power_cols]
X = df.drop(power_cols, axis=1)


# Defining the evaluation metric
def RMSE(val, pred):
    diff = (val - pred) ** 2
    rmse = np.mean(diff.values) ** 0.5
    return rmse


# Splitting the dataset into train and test data
trainset = ~Y[power_cols[0]].isnull()
X_train, Y_train = X[trainset], Y[trainset]
X_test, Y_test = X[~trainset], Y[~trainset]

# Splitting the trainset further for cross-validation
cv_split = X_train.index < '2012-05-27'
X_train_cv, Y_train_cv = X_train[cv_split], Y_train[cv_split]
X_val_cv, Y_val_cv = X_train[~cv_split], Y_train[~cv_split]

# Here comes the machine learning
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, min_samples_leaf=5)
rf.fit(X_train_cv, Y_train_cv)
Y_val_cv_hat = rf.predict(X_val_cv)

# Showing the local prediction error and feature importances
rmse = RMSE(Y_val_cv, Y_val_cv_hat)
print "Local prediction error: {} ".format(rmse)
print "Feature importances:"
for feature, importance in sorted(zip(rf.feature_importances_, X_train.columns), key=lambda x: x[0], reverse=True):
    print feature, importance

# Now we do the training and prediction for the testset
rf.fit(X_train, Y_train)
Y_test_hat = rf.predict(X_test)

#Preparing the submission file:

#Converting the prediction matrix to a dataframe
Y_test_hat=pd.DataFrame(Y_test_hat, index=X_test.index, columns=power_cols)
# We need to convert the parsed datetime back to utc timestamp
Y_test_hat['ut_ms'] = (Y_test_hat.index.astype(np.int64) * 1e-6).astype(int)
# Writing the submission file as csv
Y_test_hat[['ut_ms'] + power_cols].to_csv('rf_baseline.csv', index=False)

# Goto https://kelvins.esa.int/mars-express-power-challenge/ , create an account and submit the file!
