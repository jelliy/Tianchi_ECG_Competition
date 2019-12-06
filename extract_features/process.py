import pandas as pd
from tqdm import tqdm
from config import config
import os
from sklearn import preprocessing
import numpy as np


def load_data(case):
    df = pd.read_csv(case, sep=" ")
    df['III'] = df['II'] - df['I']
    # df['aVR']=-(df['II']+df['I'])/2
    df['aVR'] = -(df['II'] + df['I']) / 2
    # df['aVL']=(df['I']-df['II'])/2
    # df['aVF']=(df['II']-df['I'])/2
    return df


def create_data(data, win_size=2500, move=200):
    total_seconds = 10
    count = int((len(data) - win_size) / move)
    index = 0
    rest = []
    for i in range(count):
        new_data = data[index:index + win_size]
        index += move
        rest.append(new_data)
    return rest


def get_sub_featrues(df_info, train_dir):
    level = 9
    count = 207
    columns = [str(i) for i in range(10 * count + 1)]
    x = pd.DataFrame(columns=columns)
    index_list = df_info.index.values
    for i in tqdm(range(len(index_list))):
        index = index_list[i]
        # print(i, index)
        file_name = os.path.join(train_dir, df_info.loc[index, 'file'])
        # print(file_name)
        df = load_data(file_name)

        from features import feature_extractor5
        extractor = feature_extractor5
        # wave filtering
        # data = normalizer.normalize_ecg(data)
        row_features = extractor.features_for_row_ex(df, file_name)

        x.loc[i] = [index] + row_features
    return x


def get_features(df_info, train_dir):
    from multiprocessing.pool import Pool
    from multiprocessing import cpu_count
    print(cpu_count())
    processes = cpu_count() - 2
    if processes < 0:
        processes = 1

    row_len = df_info.iloc[:, 0].size

    windows = int(row_len / processes)
    index_list = [int(i) for i in range(0, row_len + 1, windows)][:processes + 1]
    index_list[-1] = row_len

    pool = Pool(processes=processes)
    df_info_samples = []
    for i in range(processes):
        df_info_samples.append(df_info.iloc[index_list[i]:index_list[i + 1], :])
    from functools import partial
    rest = pool.map(partial(get_sub_featrues, train_dir=train_dir), df_info_samples)

    # a = get_sub_featrues(df_info, train_dir)
    pool.close()
    pool.join()

    x = pd.concat(rest, ignore_index=True).iloc[:, 1:]
    # x = get_sub_featrues(df_info,train_dir).iloc[:,1:]
    # onehot
    x['age'] = df_info['age']
    # c = df_info['gender'].apply(lambda x: 1 if x == "MALE" else 0)
    # d = df_info['gender'].apply(lambda x: 1 if x == "FEMALE" else 0)
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
    with open(test_label, 'r', encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            info.append(row[:3])
    df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
    test_train = get_features(df_info, test_dir)
    return test_train


def read_train(trian_label, train_dir, name2idx):
    info = []
    label = []
    with open(trian_label, 'r', encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            info.append(row[:3])

            l = np.zeros(len(name2idx))
            for name in row[3:]:
                index = name2idx[name]
                l[index] = 1
            label.append(l)

    y_train = pd.DataFrame(label, columns=name2idx.keys()).fillna(0)
    print(y_train.sum())

    df_info = pd.DataFrame(info, columns=['file', 'age', 'gender'])
    x_train = get_features(df_info, train_dir)
    return x_train, y_train


def preprocess():
    # name2idx, idx2name = read_class_name(config.arrythmia)
    print("-----------test-----------")
    test_train = read_test(config.test_label, config.test_dir)
    test_train.to_csv("x_test.csv")


"""
def preprocess():
    name2idx, idx2name = read_class_name(config.arrythmia)
    print("-----------train-----------")
    x_train1, y_train1 = read_train(config.train_label, config.train_dir, name2idx)
    x_trainA, y_trainA = read_train(config.trainA_label, config.trainA_dir, name2idx)
    x_train = pd.concat([x_train1, x_trainA],ignore_index=True)
    y_train = pd.concat([y_train1, y_trainA], ignore_index=True)
    x_train.to_csv(r"../user_data/x_train.csv")
    y_train.to_csv("../user_data/y_train.csv")
    print("-----------test-----------")
    test_train = read_test(config.test_label, config.test_dir)
    test_train.to_csv("../user_data/x_test.csv")
"""
