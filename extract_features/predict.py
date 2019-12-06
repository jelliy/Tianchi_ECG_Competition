import pickle
import lightgbm as lgb
import pandas as pd
import config
from tqdm import tqdm

def save_model(data,filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def read_class_name(path):
    df_thmia = pd.read_csv(path, header=None, sep="\t")
    name2idx = {name: i for i, name in enumerate(list(df_thmia.values.reshape(-1)))}
    idx2name = {idx: name for name, idx in name2idx.items()}
    return name2idx, idx2name

# extract 测试集特征
from test1_ex import extract_features
#extract_features()

x_test = pd.read_csv("x_test.csv").iloc[:,1:]
x_test = x_test.fillna(42.627019)

clf_list = load_model("model.ml")

test = pd.read_csv(config.test_label, sep='\t', header=None, dtype=str)
p = [[] for i in range(test.iloc[:,0].size)]

name2idx, idx2name = read_class_name(config.arrythmia)

for i in tqdm(range(34)):
    print(str(i) + ":" + idx2name[i])
    clf = clf_list[i]
    p_test = clf.predict(x_test)

    for j in range(len(p_test)):
        if p_test[j] == 1:
            p[j].append(idx2name[i])

a = [str("\t".join(k)) for k in p]
test["result"] = a
test.to_csv("subA.txt", sep='\t', header=None, index=None)