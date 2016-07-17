from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train_predict(params, X_train, Y_train, X_val):

    if params['model']=="rf":
        # Here comes the machine learning
        rf = RandomForestRegressor(n_estimators=params['n_estimators'], n_jobs=-1,
                                   min_samples_leaf=params['min_samples_leaf'])
        print "Feature importances:"
        rf.fit(X_train, Y_train)
        for feature, importance in sorted(zip(rf.feature_importances_, X_train.columns), key=lambda x: x[0], reverse=True):
            print feature, importance
        Y_val_cv_hat = rf.predict(X_val)
        return Y_val_cv_hat
    elif params['model'] == "lstm":
        from lstm import LSTM
        rnn = LSTM(params)
        rnn.fit(X_train, Y_train)
        Y_val_cv_hat = rnn.predict(X_val)
        
        return Y_val_cv_hat
