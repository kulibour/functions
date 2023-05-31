import pandas as pd
import numpy as np
import os, pickle
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold,GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as cal_mse
from sklearn.metrics import r2_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
cal_rmse = lambda y, y_pred: round(cal_mse(y, y_pred)**0.5,4)
cal_pcc = lambda y, y_pred: round(pearsonr(y, y_pred)[0],4)
cal_r2 = lambda y, y_pred: round(r2_score(y, y_pred),4)

hyper_space = {'n_estimators':hp.quniform('n_estimators', 10,301,1),
               'eta':hp.uniform('eta',0.01,0.2),
               'max_depth':hp.quniform('max_depth',1,10,1),
               'min_child_weight':hp.uniform('min_child_weight',1,10),
               'reg_alpha':hp.uniform('reg_alpha', 0,10),
               'reg_lambda':hp.uniform('reg_lambda', 0,10),
               'gamma':hp.uniform('gamma',0,0.5),
            #    'subsample':hp.uniform('subsample',0.5,1)##小数据集建模不需要调这个
            #    'colsample_bytree' : hp.uniform ('colsample_bytree', 0.4,0.99),
               }


cv = KFold(n_splits=5, shuffle=True, random_state=42)
def model_kfold_cv(model,train_x,train_y,cv,k=5):
    score_list = []
    for idx_train,idx_test in cv.split(train_x):
        x_train,y_train = train_x[idx_train],train_y[idx_train]
        x_test,y_test = train_x[idx_test],train_y[idx_test]
        model_fit = model.fit(x_train,y_train)
        score = model_fit.score(x_test,y_test)
        score_list.append(score)
    return np.array(score_list)

def objective(space):
    model = XGBRegressor(n_estimators = int(space['n_estimators']),
                         eta = space['eta'],
                         max_depth = int(space['max_depth']),
                         min_child_weight = space['min_child_weight'],
                         reg_alpha = space['reg_alpha'],
                         reg_lambda = space['reg_lambda'],
                         gamma = space['gamma'],
                        #  subsample = hyper_space['subsample'],
                        #  colsample_bytree = hyper_space['gamcolsample_bytreema'],
                         nthread = 10
                         )
    score_list = model_kfold_cv(model)
    score_mean = float('%.3f'%(np.mean(score_list)))
    score_std = float('%.3f'%(np.std(score_list)))
    print(f'score_mean:{score_mean}, score_std:{score_std}, status:{STATUS_OK}')
    return {'loss': -np.mean(score_list), 'status': STATUS_OK }

trials = Trials()
best_hyperparams = fmin(fn = objective,
                        space = hyper_space,
                        algo = tpe.suggest,
                        max_evals = 1000,
                        trials = trials)