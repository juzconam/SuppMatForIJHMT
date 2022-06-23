# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:55:07 2020

@author: User
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def RFF_regr(X_train,y_train,X_test,y_test):
    
    RFF = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
                            max_features='auto', max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None,
                            oob_score=False, random_state=0, verbose=0, warm_start=False)
    RFF.fit(X_train, y_train.values.ravel())
    print(RFF.feature_importances_)
    predictions = RFF.predict(X_test)
    #print(predictions)
    actual = y_test.astype('float64').values.ravel()
    #print(actual)
    
    print(r2_score(actual, predictions))
    
    return