# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:44
数据预处理：
    1.构建label2index和index2label
    2.划分数据集
@ author: javis
'''
import os, torch
import numpy as np
from config import config
import pandas as pd
# 保证每次划分数据一致
np.random.seed(41)


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def split_data(file2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(config.train_dir))
    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    id2index = dict()
    index2id = dict()
    i = 0
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
        id2index[id] = i
        index2id[i] = id
        i = i+1
    return file2index, id2index, index2id


def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def train(name2idx, idx2name):
    file2idx, id2index, index2id = file2index(config.train_label, name2idx)

    if os.path.exists(r'train_index.csv'):
        train_index = pd.read_csv(r'train_index.csv', header=None).values.reshape(-1)
        val_index = pd.read_csv(r'val_index.csv', header=None).values.reshape(-1)
        train = [index2id[index] for index in train_index]
        val = [index2id[index] for index in val_index]
    else:
        train, val = split_data(file2idx)
    wc=count_labels(train,file2idx)
    print(wc)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
    torch.save(dd, config.train_data)

    train_index = [id2index[i] for i in train]
    val_index = [id2index[i] for i in val]

    df_train_index = pd.DataFrame(train_index)
    df_train_index.to_csv("train_index.csv", header=None, index=None)
    df_val_index = pd.DataFrame(val_index)
    df_val_index.to_csv("val_index.csv", header=None, index=None)

if __name__ == '__main__':
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train(name2idx, idx2name)
