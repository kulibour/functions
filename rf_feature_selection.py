import pandas as pd
import numpy as np
import os, re, json, ast, copy
from collections import defaultdict,Counter
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor(n_estimators=100,random_state=10,oob_score=True)

def cal_rfg_pcc_oob(train_set, base, y_name):
    train_x = np.array(train_set[base])
    train_y = np.array(train_set[y_name])
    model_fit = rfg.fit(train_x,train_y)
    fea_importance = model_fit.feature_importances_[-1]
    pcc_oob,pvalue = pearsonr(train_y,model_fit.oob_prediction_)
    return pcc_oob,fea_importance

class forward_feature_selection(object):
    def __init__(self, workdir, train_set, base_fea_list, features, y_name, cate, num_worker=10):
        self.num_worker = num_worker
        self.base_fea_list = base_fea_list
        self.workdir = workdir
        self.y_name = y_name
        self.train_set = train_set
        self.cate = cate
        self.features = features

    def _filter_and_select(self, candidate_data, criter_value):
        if candidate_data.shape[0]>0:
            ##选择
            candidate_data = candidate_data[candidate_data['value2']==max(candidate_data['value2'])] ###标准保留两位小数
            candidate_data = candidate_data[candidate_data['importance']==max(candidate_data['importance'])]###选择importance最高的fea
            candidate_data = candidate_data[candidate_data['value1']==max(candidate_data['value1'])]###标准不保留两位小数
            best_feature = list(candidate_data['feature'])[0]
            criter_value = list(candidate_data['value1'])[0]
            return best_feature,criter_value
        else:
            return 'meiyou',0

    def _forward(self, base, criter_value):
        judge = 0
        best_feature = ''
        candidate_data = pd.DataFrame()
        for fea in self.features:
            new_base = copy.deepcopy(base)
            if fea not in new_base:
                new_base.append(fea)
                criter_value1,fea_importance1 = cal_rfg_pcc_oob(self.train_set, new_base, self.y_name)
                if criter_value1-criter_value > 0.005:
                    judge = 1
                    fea_data = pd.DataFrame([[fea,criter_value1,round(criter_value1,2),fea_importance1]],
                                            columns=['feature','value1','value2','importance'])
                    candidate_data = pd.concat([candidate_data,fea_data],axis=0)
        if judge:
            best_feature2,criter_value2 = self._filter_and_select(candidate_data,criter_value)
            if best_feature2!='meiyou':
                best_feature = best_feature2
                criter_value = criter_value2
                return judge,best_feature,criter_value
            else:
                return -1,'no',criter_value
        else:
            return -1,'no',criter_value
        
    def _select_feature(self, base_fea):
        f = open(f'{self.workdir}rfg_{self.cate}_{base_fea}','w')
        f.write('features\tpcc\n')
        base = [base_fea]
        criter_value,base_importance = cal_rfg_pcc_oob(self.train_set, base, self.y_name)
        print(criter_value)
        f.write('%s\t%s\n'%(','.join(base),criter_value))
        judge = 0
        while judge != -1:
            judge,best_feature,criter_value = self._forward(base,criter_value)
            if judge != -1:
                base.append(best_feature)
                print(base)
                print(criter_value)
                f.write('%s\t%s\n'%(','.join(base),criter_value))
            else:
                print('end')
        f.close()

    def parallel_select(self):
        Parallel(n_jobs=self.num_worker)(delayed(self._select_feature)(base_fea) for base_fea in self.base_fea_list)