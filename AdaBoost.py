# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:34:02 2020

@author: Deepak Garg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
import xlsxwriter as xlsw

def AdaBoost(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p):
    param_dist = {
            'n_estimators': [10,100],
            'learning_rate' : [0.01,0.05,0.1,0.3,1],
            'loss' : ['linear', 'square', 'exponential']
            }
    
    
    pre_gs_inst = GridSearchCV(AdaBoostRegressor(),param_grid = param_dist,cv=5,scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)
    
    
    grid_result = pre_gs_inst.fit(X_train, y_train['h'].values.ravel())
    
    best_params = grid_result.best_params_
    print('AdaBoost Optimal Parameters', best_params['n_estimators'],best_params['learning_rate'],best_params['loss'])
    Adb = AdaBoostRegressor(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], loss=best_params['loss'], random_state=None)
    
    # # Code to plot feature importance
    Adb.fit(X_train, y_train['h'].values.ravel())
    feat_importances = pd.Series(Adb.feature_importances_, index=X_train.columns)
    feat_importances.to_excel('AdaBoost Feature Importance.xlsx', engine='xlsxwriter')
    
    pred_Adb = pre_gs_inst.predict(X_test)
    #act_Adb = y_test.astype('float64').values.ravel()
    actual = y_test['h'].values.ravel() 
    n = actual.shape[0]
    
    # MAE of test dataset
    results = pd.DataFrame({'Pred':pred_Adb, 'Act':actual
                            ,'htp':y_test['htp'].values.ravel(),'hann':y_test['hann'].values.ravel()
                            ,'filename':y_test['filename']})
    
    
    results['Diff'] = abs((1-(results['Pred']/results['Act']))*100)
    print('AdaBoost MAE=', results['Diff'].mean())
    print('AdaBoost R2=',r2_score(y_test['h'].astype('float64').values.ravel(), pred_Adb))
    print('AdaBoost Adj_R2_Score=',(1-(1-r2_score(actual, pred_Adb))*(n-1)/(n-p-1)))
    
    # Write results in a dataframe
    xlsfile = 'AdaBoost.xlsx'
    results.to_excel(xlsfile, engine='xlsxwriter')   
    
    predictions1 = pre_gs_inst.predict(X_1a)
    actual1 = df_tar_a['h'].astype('float64').values.ravel()
    
    results1 = pd.DataFrame({'Pred':predictions1, 'Act':actual1
                             ,'htp':df_tar_a['htp'].values.ravel(),'hann':df_tar_a['hann'].values.ravel()
                             ,'filename':df_tar_a['filename']})
    
    results1['Diff'] = abs((1-(results1['Pred']/results1['Act']))*100)
    results1.to_excel('AdaBoost_excluded.xlsx',sheet_name = 'Sheet1', engine='xlsxwriter')

    #-feat_importances = pd.Series(pre_gs_inst.feature_importances_, index=X_train.columns)
    #-feat_importances.to_excel('AdaBoost Feature Importance.xlsx', engine='xlsxwriter')    
 
    return