import numpy as np
import pandas as pd
from config import config
# 计算f1
def read_class_name(path):
    df_thmia = pd.read_csv(path, header=None, sep="\t")
    name2idx = {name: i for i, name in enumerate(list(df_thmia.values.reshape(-1)))}
    idx2name = {idx: name for name, idx in name2idx.items()}
    return name2idx, idx2name

def read_result(trian_label, name2idx):
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

    y_train = pd.DataFrame(label, columns=name2idx.keys(), dtype='int').fillna(0)
    return y_train

name2idx, idx2name = read_class_name(config.arrythmia)

y_act = read_result('../data/result/hefei_round1_ansA_20191008.txt', name2idx)
y_pre = read_result('../data/result/subA.txt', name2idx)

# print(y_act.values)
from sklearn.metrics import precision_score

# each class f1
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
for column in y_act.columns.values:
    print(y_act.loc[:,column].values)
    print(y_pre.loc[:, column].values)
    f1 = f1_score(y_act.loc[:,column].values,y_pre.loc[:,column].values)
    print(column+":"+str(f1))

print(precision_score(y_act, y_pre, average='micro'))