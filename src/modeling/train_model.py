import sys

from sklearn.ensemble import RandomForestRegressor

from models import train_predict

sys.path.append("../")
import pandas as pd
from utils.utils import *
from config import config


params = {"features": ["ltdata.pkl", "saaf.pkl"], "model": "rf", "n_estimators": 100, "min_samples_leaf": 5}

# for lstm:
#params = {"features": ["ltdata.pkl", "saaf.pkl"], "model": "lstm",  "num_units": 64, 'sequence_length':8,'batch_size':64,'n_epochs':10}

power = pd.read_pickle(config.data_folder + '/target.pkl')

featuresets = []
for f in params['features']:
    f_df = pd.read_pickle(config.features_folder + '/' + f)
    featuresets.append(f_df)

features = pd.concat(featuresets, axis=1)

# Extract the columns that need to be predicted
power_cols = list(power.columns)

# Now let's join everything together
df = pd.concat((features, power), axis=1)

print df.columns
Y = df[power_cols]
X = df.drop(power_cols, axis=1)

# Splitting the dataset into train and test data
trainset = ~Y[power_cols[0]].isnull()
X_train, Y_train = X[trainset], Y[trainset]
X_test, Y_test = X[~trainset], Y[~trainset]

losses = []
oobs = []
for ix, (start_date, end_date) in enumerate(config.folds):
    # Splitting the trainset further for cross-validation
    X_train_cv, Y_train_cv = X_train.ix['2000-12-12':start_date], Y_train.ix['2000-12-12':start_date]
    X_val_cv, Y_val_cv = X_train.ix[start_date:end_date], Y_train.ix[start_date:end_date]


    Y_val_cv_hat= train_predict(params, X_train_cv, Y_train_cv, X_val_cv)

    oobs.append(Y_val_cv)
    # Showing the local prediction error and feature importances
    loss = RMSE(Y_val_cv, Y_val_cv_hat)
    losses.append(loss)
    print 'Fold', ix, loss

print "CV Loss:", np.mean(losses)

Y_test_hat=train_predict(params, X_train, Y_train, X_test)
# Preparing the submission file:

# Converting the prediction matrix to a dataframe
Y_test_hat = pd.DataFrame(Y_test_hat, index=X_test.index, columns=power_cols)
# We need to convert the parsed datetime back to utc timestamp
Y_test_hat['ut_ms'] = (Y_test_hat.index.astype(np.int64) * 1e-6).astype(int)
# Writing the submission file as csv
Y_test_hat[['ut_ms'] + power_cols].to_csv('submission.csv', index=False)

# Goto https://kelvins.esa.int/mars-express-power-challenge/ , create an account and submit the file!
