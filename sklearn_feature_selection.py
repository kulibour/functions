import pandas as pd
import numpy as np
import os, re, json, ast, copy
from collections import defaultdict,Counter
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as cal_mse
from sklearn.metrics import r2_score
cal_rmse = lambda y, y_pred: round(cal_mse(y, y_pred)**0.5,4)
cal_pcc = lambda y, y_pred: round(pearsonr(y, y_pred)[0],4)
cal_r2 = lambda y, y_pred: round(r2_score(y, y_pred),4)

class filter_features(object):
    def __init__(self, feature_file, workdir, cate='descriptors', num_worker=5):
        self.dataset_features = pd.read_csv(f'{workdir}info/{feature_file}',sep='\t')
        self.feature_names = list(self.dataset_features)[1:]
        self.num_worker = num_worker
        self.workdir = workdir
        self.cate = cate
        os.makedirs(f'{self.workdir}info',exist_ok=True)
        os.makedirs(f'{self.workdir}plot',exist_ok=True)
    
    def nan_filter(self):
        ##NaN value
        dataset_fea_nan_names = []
        for name in self.feature_names:
            if set(self.dataset_features[name].isna())=={False}:
                dataset_fea_nan_names.append(name)
        np.save(f'{self.workdir}info/dataset_{self.cate}_nan_names.npy',dataset_fea_nan_names)
        self.dataset_fea_nan = self.dataset_features[['unique_id']+dataset_fea_nan_names]
        self.dataset_fea_nan_names = dataset_fea_nan_names

    def variance_filter(self):
        ###Variance
        var_filter = VarianceThreshold()
        dataset_fea_var_values = var_filter.fit_transform(self.dataset_fea_nan.iloc[:,1:])
        self.dataset_fea_var_names = var_filter.get_feature_names_out(self.dataset_fea_nan_names)
        dataset_fea_var = pd.DataFrame(dataset_fea_var_values,columns=self.dataset_fea_var_names)
        dataset_fea_var['unique_id'] = list(self.dataset_fea_nan['unique_id'])
        dataset_fea_var = dataset_fea_var[['unique_id']+list(self.dataset_fea_var_names)]
        np.save(f'{self.workdir}info/dataset_{self.cate}_var_names.npy',self.dataset_fea_var_names)
        self.dataset_fea_var = dataset_fea_var
    
    def _generate_distance_matrix(self):
        ####pearsonr distance matrix
        pcc_distance_matrix = []
        for fea1 in self.dataset_fea_var_names:
            pcc_list = Parallel(n_jobs=self.num_worker)(delayed(cal_pcc)(self.dataset_fea_var[fea1],self.dataset_fea_var[fea2]) for fea2 in self.dataset_fea_var_names)
            pcc_distance_list = [1-abs(i) for i in pcc_list]
            pcc_distance_matrix.append(pcc_distance_list)
        self.pcc_distance_matrix = np.array(pcc_distance_matrix)
    
    def _draw_similarities_distribution(self):
        ####draw similarity distribution
        plt.figure(figsize=(10,6),dpi=300)
        distance_list = self.pcc_distance_matrix.flatten()
        sns.histplot(distance_list)
        plt.xlabel('Distance',fontsize=18,fontname='Arial')
        plt.ylabel('Count',fontsize=18,fontname='Arial')
        plt.title(f'The distribution of similarities among {len(self.dataset_fea_var_names)} features',fontsize=18,fontname='Arial')
        plt.savefig(f'{self.workdir}plot/{self.cate}_similarities',dpi=300,bbox_inches='tight')
    
    def _agglomerative_cluster(self, distance_threshold=0.2):
        ####AgglomerativeClustering
        agg_cluster = AgglomerativeClustering(affinity='precomputed',n_clusters=None,distance_threshold=distance_threshold,linkage='single')
        agg_cluster.fit(self.pcc_distance_matrix)
        fea_clusters = {}
        for idx,label in enumerate(agg_cluster.labels_):
            if label not in fea_clusters:
                fea_clusters[label] = []
            fea = self.dataset_fea_var_names[idx]
            if fea not in fea_clusters[label]:
                fea_clusters[label].append(fea)
        with open(f'{self.workdir}info/{self.cate}_clusters','wb') as f:
            pickle.dump(fea_clusters, f)
        self.fea_clusters = fea_clusters
    
    def similarity_filter(self):
        ###Similarity
        ####select cluster center
        distance_df = pd.DataFrame(self.pcc_distance_matrix,columns=self.dataset_fea_var_names)
        dataset_fea_clu_names = []
        for clu in self.fea_clusters:
            # clu=33
            dist_sum_list = []
            for fea in self.fea_clusters[clu]:
                fea_distance_sum = distance_df[fea].sum()
                dist_sum_list.append(fea_distance_sum)
            clu_center = self.fea_clusters[clu][dist_sum_list.index(min(dist_sum_list))]
            dataset_fea_clu_names.append(clu_center)
        np.save(f'{self.workdir}info/dataset_{self.cate}_clu_names.npy',dataset_fea_clu_names)
        self.dataset_fea_clu_names = dataset_fea_clu_names

def cal_rfr_r2_oob(model, train_set, base, y_name):
    train_x = np.array(train_set[base])
    train_y = np.array(train_set[y_name])
    model_fit = model.fit(train_x,train_y)
    fea_importance = model_fit.feature_importances_[-1]
    r2_oob = cal_r2(train_y,model_fit.oob_prediction_)
    return r2_oob,fea_importance

def model_kfold_cv(model, train_set, base, y_name, k=5):
    train_x = np.array(train_set[base])
    train_y = np.array(train_set[y_name])
    model_fit = model.fit(train_x,train_y)
    fea_importance = model_fit.feature_importances_[-1]
    score_list = []
    cv = KFold(n_splits=5, shuffle=True, random_state=10)
    for idx_train,idx_test in cv.split(train_x):
        x_train,y_train = train_x[idx_train],train_y[idx_train]
        x_test,y_test = train_x[idx_test],train_y[idx_test]
        model_fit = model.fit(x_train,y_train)
        score = model_fit.score(x_test,y_test)
        score_list.append(score)
    return np.array(score_list).mean(),fea_importance

class forward_feature_selection(object):
    def __init__(self, workdir, train_set, base_fea_list, features, y_name, model_type, eval_type,num_worker=10):
        self.num_worker = num_worker
        self.base_fea_list = base_fea_list
        self.workdir = workdir
        self.y_name = y_name
        self.train_set = train_set
        self.features = features
        self.model_type = model_type
        self.eval_type = eval_type
        if self.model_type=='xgbr':
            self.model = XGBRegressor(random_state=10)
        if self.model_type=='rfr':
            self.model = RandomForestRegressor(random_state=10,oob_score=True)
        if self.eval_type=='oob':
            self.features_eval = cal_rfr_r2_oob
        if self.eval_type=='kfold':
            self.features_eval = model_kfold_cv

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
                criter_value1,fea_importance1 = self.features_eval(self.model, self.train_set, new_base, self.y_name)
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
        f = open(f'{self.workdir}{self.model_type}_{self.eval_type}_{base_fea}','w')
        f.write('features\tmetric\n')
        base = [base_fea]
        criter_value,base_importance = self.features_eval(self.model, self.train_set, base, self.y_name)
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


def my_metrics(task, y, y_pred):
    if task=='regression':
        pcc = cal_pcc(y, y_pred)
        r2 = cal_r2(y, y_pred)
        rmse = cal_rmse(y, y_pred)
        return pcc, r2, rmse
    else:
        return 1