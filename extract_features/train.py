import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from config import config
import math
from data_preprocessing import calc_energy
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pickle

def save_model(data,filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_result(y, p):
    r = confusion_matrix(y, p,labels=[1.0, 0.0])
    print(r)
    print(accuracy_score(y, p))
    print(f1_score(y, p))
    return r

def read_class_name(path):
    df_thmia = pd.read_csv(path, header=None, sep="\t")
    name2idx = {name: i for i, name in enumerate(list(df_thmia.values.reshape(-1)))}
    idx2name = {idx: name for name, idx in name2idx.items()}
    return name2idx, idx2name


def get_train_info():
    info = []
    with open(config.train_label, 'r',encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            info.append(row[:3])
    df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
    return df_info
# 根据索引号，增强数据
def get_more_featrues(index_list):
    df_info = get_train_info()
    train_dir = config.train_dir
    level = 9
    count = 50
    resamples = 2048
    columns = [str(i) for i in range(1*(count+8)+1)]
    x = pd.DataFrame(columns=columns)
    x_index = 0
    for i in tqdm(range(len(index_list))):
        index = index_list[i]
        # print(i, index)
        file_name = os.path.join(train_dir, df_info.loc[index,'file'])
        from test1 import load_data
        raw_df = load_data(file_name)

        #print(df.columns.values[:1])
        df_list = create_data(raw_df,win_size=2500,move=500)

        for df in df_list:
            row_features = []
            for column in df.columns.values[:1]:
                data = df.loc[:, column].values
                from scipy import signal
                # data = signal.resample(data, resamples)
                # data = fiter_wave(data)
                # 每列都计算
                from time_domain_feature import psfeatureTime
                time_features = psfeatureTime(data)
                #wavelet_features = calc_energy(data, 'db10', level, "freq", count)
                #row_features = row_features + time_features + wavelet_features0000
                fxx, pxx = signal.welch(data, 500)
                row_features = row_features + time_features+ list(pxx[:count])
            x.loc[x_index] = [index]+row_features
            x_index += 1

    return x

def get_expansion(index_list):
    from multiprocessing.pool import Pool
    processes = 6

    row_len = len(index_list)

    windows = int(row_len/processes)
    index_sep = [int(i) for i in range(0,row_len,windows)][:processes+1]
    index_sep[-1] = row_len


    df_info_samples = []
    for i in range(processes):
        df_info_samples.append(index_list[index_sep[i]:index_sep[i+1]])
    from functools import partial
    pool = Pool(processes=processes)
    rest = pool.map(get_more_featrues,df_info_samples)
    # a = get_sub_featrues(df_info, train_dir)
    pool.close()
    pool.join()


    x = pd.concat(rest,ignore_index=True).iloc[:,1:]
    df_info = get_train_info()
    df_info = df_info.iloc[index_list,:].reset_index(drop=True)
    # onehot
    x['age'] = df_info['age']
    #c = df_info['gender'].apply(lambda x: 1 if x == "MALE" else 0)
    #d = df_info['gender'].apply(lambda x: 1 if x == "FEMALE" else 0)
    x['gender1'] = df_info['gender'].apply(lambda x: 1 if x == "MALE" else 0)
    x['gender2'] = df_info['gender'].apply(lambda x: 1 if x == "FEMALE" else 0)
    x.to_csv("x_train_exp.csv")
    x = pd.read_csv("x_train_exp.csv").iloc[:,1:]
    x = x.fillna(x.mean()['age'])
    print(x)
    return x

def create_data(data,win_size=2500,move=200):
    total_seconds = 10
    count = int((len(data) - win_size)/move)
    index = 0
    rest = []
    for i in range(count):
        new_data = data.iloc[index:index + win_size, :]
        index += move
        rest.append(new_data)
    return rest

if __name__ == '__main__':
    from test1_ex import extract_features
    #extract_features()
    #exit()
    name2idx, idx2name = read_class_name(config.arrythmia)
    y_train = pd.read_csv("y_train.csv").iloc[:,1:]
    columns_num = len(y_train.columns.values)
    x_train = pd.read_csv("x_train.csv").iloc[:,1:]
    #x_test = pd.read_csv(r"x_test.csv").iloc[:, 1:]
    #x_test['age'].fillna(42.627019408001736, inplace=True)
    """
    data = pd.concat([x_train,y_train],axis=1)
    data = data.drop_duplicates()
    data = data.reset_index().iloc[:,1:]
    x_train = data.iloc[:,:-34]
    y_train = data.iloc[:, -34:]
    """

    average_age = x_train.mean()['age']
    print(average_age)
    #x_train.fillna(average_age, inplace=True)
    x_train['age'].fillna(average_age, inplace=True)
    x_train = x_train.replace(np.nan,0)
    x_train = x_train.replace(np.inf, 0)

    print(x_train)
    x_deep_train = pd.read_csv("deep_features.csv").iloc[:,1:]
    x_train = pd.concat([x_train, x_deep_train], axis=1)
    print(x_train)

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = StandardScaler()
    #x_train = scaler.fit_transform(x_train)
    #x_test = scaler.transform(x_test)


    # y_train = y_train.iloc[:,1:2].values.reshape(-1)
    # RandomForestClassifier LinearSVC
    # clf = StackingCVClassifier(classifiers=[rfc,lr], meta_classifier=lr2, use_probas=True, cv=5, verbose=1)
    # clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=300, max_depth=12,criterion="entropy", class_weight="balanced",oob_score=True)
    # clf = LogisticRegression(class_weight="balanced")
    # clf = LogisticRegression()
    # clf = LinearSVC()
    # clf = SVC()
    # clf = GradientBoostingClassifier(learning_rate=1, n_estimators=100,max_depth=5,random_state=10)
    # clf= XGBClassifier()#调用XGBRegressor函数‍

    #test = pd.read_csv(config.test_label, sep='\t', header=None, dtype=str)
    #p = [[] for i in range(test.iloc[:,0].size)]

    # 分训练集和测试集
    x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=12)

    train_index = pd.read_csv(r'train_index.csv', header=None).values.reshape(-1)
    val_index = pd.read_csv(r'val_index.csv', header=None).values.reshape(-1)

    #quality_array = pd.read_csv("quality.csv", header=None).values.reshape(-1)
    #train_index = np.setdiff1d(train_index, np.argsort(quality_array)[:100])

    x_tra = x_train.loc[train_index,:]
    x_val = x_train.loc[val_index,:]
    y_tra = y_train.loc[train_index,:]
    y_val = y_train.loc[val_index,:]

    name2idx, idx2name = read_class_name(config.arrythmia)
    model_list = []
    pred_list = []
    cm = np.zeros((2,2))
    #clf_list = load_model("model.ml")
    #for i in tqdm(range(config.num_classes)):
    #for i in tqdm(range(25)):
    for i in tqdm(range(len(y_train.columns.values))):
        print(str(i)+":"+idx2name[i])
        y_tra1 = y_tra.iloc[:, [i]].values.reshape(-1)
        y_val1 = y_val.iloc[:, [i]].values.reshape(-1)

        # 特征选择
        from sklearn.feature_selection import SelectKBest, f_classif
        #selector = SelectKBest(score_func=f_classif, k=500)
        #x_train = selector.fit_transform(x_train, y1_train)
        #x_test = selector.transform(x_test)

        """
        clf = lgb.LGBMClassifier(num_leaves=90, learning_rate=0.1, n_estimators=145,
                                 is_unbalance=True, verbose=-1,objective='binary',metric='auc')
        """
        #clf = lgb.LGBMClassifier(num_leaves=100, learning_rate=0.1, n_estimators=100,feature_fraction=0.9,
        #                        is_unbalance=True, verbose=-1,objective='binary')

        clf = lgb.LGBMClassifier(num_leaves=120, learning_rate=0.1, n_estimators=600,
                                 is_unbalance=True, verbose=-1,objective='binary',metric='auc')

        from catboost import  CatBoostClassifier
        #clf = CatBoostClassifier(learning_rate=0.1, loss_function='Logloss', iterations=500, eval_metric="AUC")
        # 异常索引，数据增强
        # 读取数据 ，窗口滑动

        clf.fit(x_tra, y_tra1)
        #clf.fit(x_tra, y_tra1,eval_set=(x_val, y_val1), verbose=-1,eval_metric='auc')
        #clf.fit(x_tra, y_tra1, eval_set=(x_val, y_val1),early_stopping_rounds=50)
        #clf = clf_list[i]
        p_val = clf.predict(x_val)
        #p_test = clf.predict(x_test)
        #print(p_test)
        joblib.dump(clf, "ml%s.pkl"%(str(i)))
        r = get_result(y_val1, p_val)
        cm += r
        print(cm)
        pred_list.append(p_val)
        model_list.append(clf)

    y_pre = np.array(pred_list).T
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    print("micro f1:"+str(precision_score(y_val, y_pre, average='micro')))
    #print(f1_score(y_val, y_pre))
    print(cm)
    save_model(model_list,"model.ml")
    p = cm[0,0]/(cm[0,0]+cm[0,1])
    q = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print(2*p*q/(p+q))