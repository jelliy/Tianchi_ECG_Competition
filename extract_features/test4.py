import pandas as pd
import numpy as np
import os
from config import config
from tqdm import tqdm
#import warnings

def get_all_files(train_label,train_dir):
    filenames = []
    with open(train_label, 'r', encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            filenames.append(os.path.join(train_dir, row[0]))
    return filenames

filenames = get_all_files(config.train_label, config.train_dir)

quality_array = pd.read_csv("quality.csv", header=None).values.reshape(-1)

for i in np.argsort(quality_array)[:100]:
    print(quality_array[i])
    print(filenames[i])