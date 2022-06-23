# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:07:59 2020

@author: Deepak Garg
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
import xlsxwriter as xlsw


def xgboost(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p):
    
    # XGBoost
    param = {
        'eta': [0.1,0.2,0.3]
        ,'max_depth': [3,6,10]
        #,'learning_rate': [0.1, 0.01, 0.05]
        #,'n_estimators': range(60, 220, 40)
        #,'colsample_bytree':[0.7,0.8]
        , 'reg_alpha':[1.1,1.2,1.3]
        , 'reg_lambda':[1.1,1.2,1.3]
        } 
    
    #steps = 20  # The number of training iterations
    
    # In order for XGBoost to be able to use our data, we’ll need to transform it into a specific format that XGBoost can handle. 
    # That format is called DMatrix. It’s a very simple one-linear to transform a numpy array of data to DMatrix format:
    #D_train = xgb.DMatrix(X_train, label=y_train
    #D_test = xgb.DMatrix(X_test, label=y_test)
    xgbr = xgb.XGBRegressor()
    pre_gs_inst = GridSearchCV(xgbr,param_grid = param,cv=5,scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = pre_gs_inst.fit(X_train, y_train['h'].values.ravel())
    
    best_params = grid_result.best_params_
    
    print('XGBoost Optimal Parameters', best_params['eta'],best_params['max_depth'],best_params['reg_alpha'],best_params['reg_lambda'])
    #XGBgs= xgbr(eta=best_params['eta'], 'max_depth'=best_params['max_depth'])
          
    xgbr1 = xgb.XGBRegressor(**best_params)
    
    # # Code to plot feature importance
    xgbr1.fit(X_train, y_train['h'].values.ravel())
    feat_importances = pd.Series(xgbr1.feature_importances_, index=X_train.columns)
    feat_importances.to_excel('XGBoost Feature Importance.xlsx', engine='xlsxwriter')
    
    
    scores = cross_val_score(xgbr1, X_train,y_train['h'].values.ravel(),cv=5)
    print("Mean cross-validation score: %.2f" % scores.mean())
    
    pred_xgbr = xgbr1.predict(X_test)
    actual = y_test['h'].astype('float64').values.ravel()
    n = actual.shape[0]
    
    print('XgBoost R2=',r2_score(actual, pred_xgbr))
    print('XgBoost Adj_R2_Score=',(1-(1-r2_score(actual, pred_xgbr))*(n-1)/(n-p-1)))
    
    # MAE of test dataset
    results = pd.DataFrame({'Pred':pred_xgbr, 'Act':actual
                            ,'htp':y_test['htp'].values.ravel(),'hann':y_test['hann'].values.ravel()
                            ,'filename':y_test['filename']})
    
    
    results['Diff'] = abs((1-(results['Pred']/results['Act']))*100)
    print('XgBoost MAE=',results['Diff'].mean())
    
    # Write results in a dataframe
    xlsfile = 'XGBoost.xlsx'
    results.to_excel(xlsfile, engine='xlsxwriter')    
    
    predictions1 = xgbr1.predict(X_1a)
    actual1 = df_tar_a['h'].astype('float64').values.ravel()

    results1 = pd.DataFrame({'Pred':predictions1, 'Act':actual1
                             ,'htp':df_tar_a['htp'].values.ravel(),'hann':df_tar_a['hann'].values.ravel()
                             ,'filename':df_tar_a['filename']})

    results1['Diff'] = abs((1-(results1['Pred']/results1['Act']))*100)
    results1.to_excel('XGBoost_excluded.xlsx',sheet_name = 'Sheet1', engine='xlsxwriter')    
 
    
    
    return