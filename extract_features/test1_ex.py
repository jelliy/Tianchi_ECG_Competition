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
from sklearn import preprocessing
import numpy as np
from time_domain_feature import psfeatureTime
from intf import fiter_wave

label = preprocessing.LabelEncoder()
one_hot = preprocessing.OneHotEncoder(sparse = False)

def load_data(case):#定义更多参数，具体参考官方说明
    df = pd.read_csv(case,sep=" ")
    df['III'] = df['II']-df['I']
    #df['aVR']=-(df['II']+df['I'])/2
    df['aVR'] = (df['II'] + df['I']) / 2
    #df['aVL']=(df['I']-df['II'])/2
    #df['aVF']=(df['II']-df['I'])/2
    return df

def create_data(data,win_size=2500,move=200):
    total_seconds = 10
    count = int((len(data) - win_size)/move)
    index = 0
    rest = []
    for i in range(count):
        new_data = data[index:index+win_size]
        index += move
        rest.append(new_data)
    return rest

def get_sub_featrues(df_info,train_dir):
    level = 9
    count = 2429
    columns = [str(i) for i in range(count+1)]
    x = pd.DataFrame(columns=columns)
    index_list = df_info.index.values
    for i in tqdm(range(len(index_list))):
        index = index_list[i]
        # print(i, index)
        file_name = os.path.join(train_dir, df_info.loc[index,'file'])
        #print(file_name)
        df = load_data(file_name)

        row_features = []

        from features import feature_extractor5
        extractor = feature_extractor5
        # wave filtering
        #data = normalizer.normalize_ecg(data)
        row_features = extractor.features_for_row_ex(df, df_info.loc[index,'file'])

        x.loc[i] = [index]+row_features
    return x

def get_features(df_info, train_dir):
    from multiprocessing.pool import Pool
    processes = 6

    row_len = df_info.iloc[:,0].size

    windows = int(row_len/processes)
    index_list = [int(i) for i in range(0,row_len+1,windows)][:processes+1]
    index_list[-1] = row_len

    pool = Pool(processes=processes)
    df_info_samples = []
    for i in range(processes):
        df_info_samples.append(df_info.iloc[index_list[i]:index_list[i+1],:])
    from functools import partial
    rest = pool.map(partial(get_sub_featrues,train_dir=train_dir), df_info_samples)

    # a = get_sub_featrues(df_info, train_dir)
    pool.close()
    pool.join()

    x = pd.concat(rest,ignore_index=True).iloc[:,1:]
    #x = get_sub_featrues(df_info,train_dir).iloc[:,1:]
    # onehot
    x['age'] = df_info['age']
    #c = df_info['gender'].apply(lambda x: 1 if x == "MALE" else 0)
    #d = df_info['gender'].apply(lambda x: 1 if x == "FEMALE" else 0)
    x['gender1'] = df_info['gender'].apply(lambda x: 1 if x == "MALE" else 0)
    x['gender2'] = df_info['gender'].apply(lambda x: 1 if x == "FEMALE" else 0)
    print(x.head())
    return x

def read_class_name(path):
    df_thmia = pd.read_csv(path, header=None, sep="\t")
    name2idx = {name: i for i, name in enumerate(list(df_thmia.values.reshape(-1)))}
    idx2name = {idx: name for name, idx in name2idx.items()}
    return name2idx, idx2name

def read_test(test_label, test_dir):
    info = []
    with open(test_label, 'r',encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            info.append(row[:3])
    df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
    test_train = get_features(df_info, test_dir)
    return test_train

def read_train(trian_label, train_dir, name2idx):
    info = []
    label = []
    with open(trian_label, 'r',encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            info.append(row[:3])

            l = np.zeros(len(name2idx))
            for name in row[3:]:
                index = name2idx[name]
                l[index] = 1
            label.append(l)

    # 将列表中的Series合并转为DataFrame
    y_train = pd.DataFrame(label, columns=name2idx.keys()).fillna(0)
    print(y_train.sum())

    df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
    x_train = get_features(df_info, train_dir)
    return x_train, y_train

def extract_features():
    name2idx, idx2name = read_class_name(config.arrythmia)
    print("-----------train-----------")
    x_train, y_train = read_train(config.train_label, config.train_dir, name2idx)
    x_train.to_csv("x_train.csv")
    y_train.to_csv("y_train.csv")
"""
if __name__ == '__main__':
    name2idx, idx2name = read_class_name(config.arrythmia)
    print("-----------train-----------")
    x_train, y_train = read_train(config.train_label, config.train_dir, name2idx)
    x_train.to_csv("x_train.csv")
    y_train.to_csv("y_train.csv")
    exit()

    print("-----------test-----------")
    test_train = read_test(config.test_label, config.test_dir)
    test_train.to_csv("x_test.csv")


    y_train = y_train.iloc[:,1:2]
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=50, criterion="entropy")
    x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train.values.reshape(len(y_train), ), test_size=0.1, random_state=2011)
    from sklearn.model_selection import cross_val_score
    scores_clf_svc_cv = cross_val_score(clf,x_tra,y_tra,cv=5)
    print(scores_clf_svc_cv)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores_clf_svc_cv.mean(), scores_clf_svc_cv.std() * 2))

    clf.fit(x_tra, y_tra)
    p_val = clf.predict(x_val)

    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    def get_result(y, p):
        print(confusion_matrix(y, p))
        print(accuracy_score(y, p))
        print(f1_score(y, p))

    get_result(y_val, p_val)
    exit()
"""
