# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal



def transform(sig, train=False):
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.train_data)
        self.train = train
        self.data_index = dd["train2index"] if train else dd["val2index"]
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

        x_train = pd.read_csv("x_train.csv").iloc[:, 1:]
        average_age = x_train.mean()['age']
        # x_train.fillna(average_age, inplace=True)
        x_train['age'].fillna(average_age, inplace=True)
        x_train = x_train.replace(np.nan, 0)
        x_train = x_train.replace(np.inf, 0)
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import MinMaxScaler
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        self.train = x_train[self.data_index,:]


    def __getitem__(self, index):
        fid = self.data[index]

        x = self.train[index,:]
        x = torch.tensor(x, dtype=torch.float32)
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0])
