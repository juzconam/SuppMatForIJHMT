# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:30:19 2020

@author: Deepak Garg
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
import matplotlib.pyplot as plt
import xlsxwriter as xlsw

def RandFor(X_train,y_train,X_test,y_test,X_1a,df_tar,df_tar_a,p):
    gsc = GridSearchCV(estimator=RandomForestRegressor(),param_grid={'max_depth': range(3,7),'n_estimators': (10,40,50,60,70)}
                       ,cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X_train, y_train['h'].values.ravel())
    best_params = grid_result.best_params_
    
    print('best_params=',best_params)
    
    #rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
        
    RFF = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=best_params["max_depth"],
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=best_params["n_estimators"], n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)

    scores = cross_val_score(RFF, X_train, y_train['h'].values.ravel(), cv=10, scoring='neg_mean_absolute_error')
    RFF.fit(X_train, y_train['h'].values.ravel())
    pred_RFF = RFF.predict(X_test)
    act_RFF = y_test['h'].astype('float64').values.ravel()
    
    n = act_RFF.shape[0]    
    print('Random Forest R2_Score=',r2_score(act_RFF, pred_RFF))
    print('Random Forest Adj_R2_Score=',(1-(1-r2_score(act_RFF, pred_RFF))*(n-1)/(n-p-1)))
    
    #print('MAE=',mean_absolute_error(act_RFF, pred_RFF, multioutput='uniform_average')/X_train.shape[0])
    
    # # MAE of test dataset
    results = pd.DataFrame({'Pred':pred_RFF, 'Act':y_test['h'].values.ravel()
                            ,'htp':y_test['htp'].values.ravel(),'hann':y_test['hann'].values.ravel()
                            ,'filename':y_test['filename']})
    
    results['Diff'] = abs((1-(results['Pred']/results['Act']))*100)
    print('Random Forest MAE=',results['Diff'].mean())
    
    # Code to plot feature importance
    feat_importances = pd.Series(RFF.feature_importances_, index=X_train.columns)
    feat_importances.to_excel('Random Forest Feature Importance.xlsx', engine='xlsxwriter')
    feat_importances.nlargest(10).plot(kind='barh')
    plt.savefig('RFF_Feature_Importance.png')
    
    
    # Write results in a dataframe
    xlsfile = 'Random Forest.xlsx'
    results.to_excel(xlsfile,sheet_name ='Sheet1',engine='xlsxwriter')
    
    predictions1 = RFF.predict(X_1a)
    actual1 = df_tar_a['h'].astype('float64').values.ravel()
    
    results1 = pd.DataFrame({'Pred':predictions1, 'Act':actual1
                             ,'htp':df_tar_a['htp'].values.ravel(),'hann':df_tar_a['hann'].values.ravel()
                             ,'filename':df_tar_a['filename']})
    
    results1['Diff'] = abs((1-(results1['Pred']/results1['Act']))*100)
    results1.to_excel('RFF_excluded.xlsx',sheet_name = 'Sheet1', engine='xlsxwriter')

    return

