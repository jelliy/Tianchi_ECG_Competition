from neurokit import ecg_preprocess, ecg_signal_quality
import pandas as pd
import numpy as np
import os
from config import config
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
def ecg_process(ecg, sampling_rate=1000, filter_type="FIR", filter_band="bandpass", filter_frequency=[3, 45],
                segmenter="hamilton", quality_model="default"):

    processed_ecg = ecg_preprocess(ecg,
                               sampling_rate=sampling_rate,
                               filter_type=filter_type,
                               filter_band=filter_band,
                               filter_frequency=filter_frequency,
                               segmenter=segmenter)

    quality_array = ecg_signal_quality(processed_ecg["ECG"]["Cardiac_Cycles"], sampling_rate, quality_model=quality_model)['Cardiac_Cycles_Signal_Quality']

    quality_average = 0
    if len(quality_array) > 2:
        quality_list = list(quality_array)
        quality_list.remove(max(quality_array))
        quality_list.remove(min(quality_array))
        quality_average = np.mean(quality_array)

    return quality_average

# ergod train data
def get_all_files(train_label,train_dir):
    filenames = []
    with open(train_label, 'r', encoding="utf8") as f:
        for l in f.readlines():
            row = l.rstrip().split('\t')
            filenames.append(os.path.join(train_dir, row[0]))
    return filenames

filenames = get_all_files(config.train_label, config.train_dir)

quality_list = []
for f in tqdm(filenames):
    df = pd.read_csv(f, sep=" ")
    quality = 0
    for column in df.columns:
        y = df.loc[:,column].values
        try:
            quality += ecg_process(y, sampling_rate=500)
        except:
            quality += 0
    quality_list.append(quality/8)
    if quality < 0.3:
        print(str(quality)+" ------- "+filenames)

pd.DataFrame(quality_list).to_csv("quality.csv", header=None,index=None)