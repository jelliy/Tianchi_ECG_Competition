"""
Plots records and detected R peaks
applying both QRS detection algorithms

See common/qrs_detect.py, common/qrs_detect2.py
"""

import random

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

from features.qrs_detect import *

plt.rcParams["figure.figsize"] = (20, 4)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

from loading import loader

from features.qrs_detect2 import *


def plot_with_detected_peaks(row, clazz):
    r = qrs_detect(normalizer.normalize_ecg(row))

    print('R', len(r), r)
    times = np.diff(r)
    print(np.mean(times), np.std(times))
    plt.subplot(2, 1, 1)
    plt.plot(range(len(row)), row, 'g-',
             r, [row[x] for x in r], 'r^')
    plt.ylabel(clazz + "QRS 1")

    r = qrs_detect2(row, fs=300, thres=0.4, ref_period=0.2)

    print('R', len(r), r)
    times = np.diff(r)
    print(np.mean(times), np.std(times))
    plt.subplot(2, 1, 2)
    plt.plot(range(len(row)), row, 'g-',
             r, [row[x] for x in r], 'r^')
    plt.ylabel(clazz + " QRS 2")

    plt.show()


# Normal: A00001, A00002, A0003, A00006
plot_with_detected_peaks(loader.load_data_from_file("A00001"), "Normal")
# AF: A00004, A00009, A00015, A00027
plot_with_detected_peaks(loader.load_data_from_file("A00004"), "AF")
# Other: A00005, A00008, A00013, A00017
plot_with_detected_peaks(loader.load_data_from_file("A00005"), "Other")
# Noisy: A00205, A00585, A01006, A01070
plot_with_detected_peaks(loader.load_data_from_file("A00205"), "Noisy")
