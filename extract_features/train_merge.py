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
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pickle
import torch, time, os, shutil
import models, utils
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_result(y, p):
    r = confusion_matrix(y, p, labels=[1.0, 0.0])
    print(r)
    # print(accuracy_score(y, p))
    print(f1_score(y, p))
    return r


def read_class_name(path):
    df_thmia = pd.read_csv(path, header=None, sep="\t")
    name2idx = {name: i for i, name in enumerate(list(df_thmia.values.reshape(-1)))}
    idx2name = {idx: name for name, idx in name2idx.items()}
    return name2idx, idx2name


def get_train_info():
    info = []
    with open(config.train_label, 'r', encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            info.append(row[:3])
    df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
    return df_info


def create_data(data, win_size=2500, move=200):
    total_seconds = 10
    count = int((len(data) - win_size) / move)
    index = 0
    rest = []
    for i in range(count):
        new_data = data.iloc[index:index + win_size, :]
        index += move
        rest.append(new_data)
    return rest


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    output_all = None
    target_all = None
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            if output_all is None:
                output_all = output
            else:
                output_all = torch.cat((output_all, output), 0)
            if target_all is None:
                target_all = target
            else:
                target_all = torch.cat((target_all, target), 0)
            # f1 = utils.calc_f1(target, output, threshold)
            # f1_meter += f1
        # utils.calc_f1_ex(target_all, output_all, threshold)
    return output_all


def deep_predict():
    from torch import nn
    from torch.utils.data import DataLoader
    from dataset import ECGDataset
    list_threhold = [0.5]
    model = getattr(models, config.model_name)()
    checkpoint = torch.load(os.path.join(r'ckpt\resnet34', config.best_w), map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6)
    for threshold in list_threhold:
        output_all = val_epoch(model, criterion, val_dataloader, threshold)
        # print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))
    return output_all.cpu().numpy()


def light_gbm_predict(x_test, do_not_pre_classes):
    rows_number = x_test.iloc[:, 0].size
    clf_list = load_model("model.ml")

    pred_list = []

    name2idx, idx2name = read_class_name(config.arrythmia)

    for i in tqdm(range(34)):
        print(str(i) + ":" + idx2name[i])
        if i in do_not_pre_classes:
            pred_list.append(np.zeros(rows_number))
            print("skip .........")
            continue

        clf = clf_list[i]
        p_test = clf.predict_proba(x_test)
        pred_list.append(p_test[:, 1])

    return np.array(pred_list).T


def merge_result(r1, r2, deep_model_pre_classes):
    # r1 is deep model predict
    r2 = (r1+r2)/2
    """
    for i in deep_model_pre_classes:
        for index in range(r2.shape[0]):
            r2[index, i] = 1 if (r1[index, i] > 0.5) else 0
    """
    return r2


def calc_f1_all(y_true, y_pre):
    cm = np.zeros((2, 2))
    for i in range(34):
        print("------- " + str(i) + " --------")
        predict = [0. if i < 0.5 else 1. for i in y_pre[:, i]]
        r = get_result(y_true[:, i], predict)
        cm += r
    print(cm)
    p = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    q = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print(2 * p * q / (p + q))


if __name__ == '__main__':
    name2idx, idx2name = read_class_name(config.arrythmia)
    y_train = pd.read_csv("y_train.csv").iloc[:, 1:]
    columns_num = len(y_train.columns.values)
    x_train = pd.read_csv("x_train.csv").iloc[:, 1:]

    average_age = x_train.mean()['age']
    print(average_age)
    x_train['age'].fillna(average_age, inplace=True)

    train_index = pd.read_csv(r'train_index.csv', header=None).values.reshape(-1)
    val_index = pd.read_csv(r'val_index.csv', header=None).values.reshape(-1)

    x_tra = x_train.loc[train_index, :]
    x_val = x_train.loc[val_index, :]
    y_tra = y_train.loc[train_index, :]
    y_val = y_train.loc[val_index, :]

    y_val.to_csv("y.csv", header=None, index=None)
    deep_model_pre_classes = [1, 4, 9, 10, 12, 22, 26]
    # deep_model_pre_classes = [4, 10, 12, 17, 22, 24, 31, 32]
    do_not_pre_classes = [0, 2, 7, 8, 15, 19, 23, 25, 33]

    r1 = deep_predict()
    print("deep-------------")
    calc_f1_all(y_val.values, r1)
    df_r1 = pd.DataFrame(r1)
    df_r1.to_csv("r1.csv", header=None, index=None)
    r2 = light_gbm_predict(x_val, do_not_pre_classes)
    print("lightgbm-------------")
    calc_f1_all(y_val.values, r2)
    df_r2 = pd.DataFrame(r2)
    df_r2.to_csv("r2.csv", header=None, index=None)
    out = merge_result(r1, r2, deep_model_pre_classes)
    print("merge-------------")
    calc_f1_all(y_val.values, out)
