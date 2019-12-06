import numpy as np
from config import config
import math
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pickle

def save_model(data,filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_result(y, p):
    print(confusion_matrix(y, p))
    print(accuracy_score(y, p))
    print(f1_score(y, p))


def read_class_name(path):
    df_thmia = pd.read_csv(path, header=None, sep="\t")
    name2idx = {name: i for i, name in enumerate(list(df_thmia.values.reshape(-1)))}
    idx2name = {idx: name for name, idx in name2idx.items()}
    return name2idx, idx2name


if __name__ == '__main__':
    from process import preprocess

    preprocess()
    name2idx, idx2name = read_class_name(config.arrythmia)
    x_test = pd.read_csv(r"x_test.csv").iloc[:, 1:]
    x_test['age'].fillna(42.627019408001736, inplace=True)
    print(x_test)
    clf_list = load_model("model.ml")
    test = pd.read_csv(config.test_label, sep='\t', header=None, dtype=str)
    p = [[] for i in range(test.iloc[:, 0].size)]

    for i in tqdm(range(7)):
        print(str(i) + ":" + idx2name[i])
        from sklearn.externals import joblib
        clf = joblib.load("ml%s.pkl" % (str(i)))
        #clf = clf_list[i]
        p_test = clf.predict(x_test)
        print(p_test)
        for j in range(len(p_test)):
            if p_test[j] == 1:
                p[j].append(idx2name[i])

    a = [str("\t".join(k)) for k in p]
    test["result"] = a
    test.to_csv(r"subB.txt", sep='\t', header=None, index=None)

    str = ""
    with open('subB.txt', "r", encoding="utf8") as f:
        str = f.read().replace('"', '')

    with open(r"result.txt", 'w', encoding="utf8") as f:
        f.write(str)
