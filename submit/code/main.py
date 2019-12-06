# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch
import os
import models, utils
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from config import config

import pickle
import lightgbm as lgb
import pandas as pd
from tqdm import tqdm
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)


def save_model(data, filename):
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


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def deep_predict():
    from dataset import transform
    from data_process import name2index
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    # model
    model = getattr(models, config.model_name)()
    model.load_state_dict(torch.load(os.path.join(r'ckpt/resnet34', config.best_w), map_location='cpu')['state_dict'])
    # best_w = torch.load(os.path.join(r'ckpt/resnet34', config.best_w))
    # model.load_state_dict(best_w['state_dict'])
    model = model.to(device)
    model.eval()

    test_data = []
    for line in open(config.test_label, encoding='utf-8'):
        id = line.split('\t')[0]
        file_path = os.path.join(config.test_dir, id)
        test_data.append(file_path)

    print(len(test_data))
    from dataset import ECGDataTest
    test_dataset = ECGDataTest(test_data)

    from multiprocessing import cpu_count
    print(cpu_count())
    processes = cpu_count() - 1
    if processes < 0:
        processes = 1

    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=processes)

    output_list = np.empty(shape=[0, config.num_classes])
    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            output = torch.sigmoid(model(inputs)).cpu().numpy()
            output_list = np.vstack((output_list, output))
    return np.array(output_list)


def light_gbm_predict(do_not_pre_classes):
    from process import preprocess
    preprocess()

    x_test = pd.read_csv(r"../user_data/x_test.csv").iloc[:, 1:]
    x_test['age'].fillna(42.627019408001736, inplace=True)
    print(x_test)
    rows_number = x_test.iloc[:, 0].size
    clf_list = load_model("model.ml")

    pred_list = []

    name2idx, idx2name = read_class_name(config.arrythmia)

    for i in tqdm(range(34)):
        if i in do_not_pre_classes:
            pred_list.append(np.zeros(rows_number))
            print("skip .........")
            continue

        clf = clf_list[i]
        p_test = clf.predict(x_test)
        pred_list.append(p_test)

    return np.array(pred_list).T


def writ_to_file(out):
    test = pd.read_csv(config.test_label, sep='\t', header=None, dtype=str)
    name2idx, idx2name = read_class_name(config.arrythmia)
    p = [[] for i in range(test.iloc[:, 0].size)]
    for i in range(34):
        for j in range(len(out[:, i])):
            if out[j, i] == 1:
                p[j].append(idx2name[i])

    a = [str("\t".join(k)) for k in p]
    test["result"] = a
    test.to_csv(r"../user_data/subB.txt", sep='\t', header=None, index=None)

    str_result = ""
    with open('../user_data/subB.txt', "r", encoding="utf8") as f:
        str_result = f.read().replace('"', '')

    with open(r"../result.txt", 'w', encoding="utf8") as f:
        f.write(str_result)


def merge_result(r1, r2):
    # r1 is deep model predict
    """
    w1 = pd.read_csv("model_weight.csv", header=None).values.reshape(-1)
    w2 = 1-w1
    # r1 is deep model predict
    r = r1*w1+r2*w2
    """
    r3 = np.ones(r1.shape)
    w = pd.read_csv("model_weight3.csv", header=None).values
    r = w[:,0]*r1 + w[:,1]*r2 +w[:,2]*r3
    for i in range(config.num_classes):
        for index in range(r.shape[0]):
            r[index, i] = 1 if (r[index, i] > 0.5) else 0
    return r


if __name__ == '__main__':
    print(device)
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    do_not_pre_classes = [0, 2, 7, 8, 15, 19, 23, 25, 33]

    print(datetime.datetime.now())
    r2 = light_gbm_predict(do_not_pre_classes)
    print(datetime.datetime.now())
    print(r2)

    r1 = deep_predict()
    print(r1)
    print(datetime.datetime.now())
    out = merge_result(r1, r2)
    print(out)
    """
    r = np.zeros((r1.shape[0],34))
    for i in range(34):
        for index in range(r1.shape[0]):
            r[index,i] = 1 if(r1[index,i] > 0.5) else 0
    """
    writ_to_file(out)
