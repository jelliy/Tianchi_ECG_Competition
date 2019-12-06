"""
This script plots different stages of normalization of the signal
"""

import random

import matplotlib
from scipy.fftpack import fft

from biosppyex.signals import ecg

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from loading import loader
from features.qrs_detect import *

plt.rcParams["figure.figsize"] = (4, 7)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)


def plot_with_detected_peaks(x, clazz):
    total = 7

    x = ecg.ecg(x, sampling_rate=300, show=False)['templates']

    m = [np.median(col) for col in x.T]

    dists = [np.sum(np.square(s-m)) for s in x]
    pmin = np.argmin(dists)

    x = x[pmin]

    p = x[:45]
    qrs = x[45:80]
    t = x[80:]

    plt.subplot(total, 1, 1)
    plt.ylabel(clazz)
    plt.plot(x)

    plt.subplot(total, 1, 2)
    plt.ylabel(clazz)
    plt.plot(p)

    plt.subplot(total, 1, 3)
    plt.ylabel("P DFT")
    ff = fft(p)[:len(p) // 2]
    plt.plot(ff, 'r')

    plt.subplot(total, 1, 4)
    plt.ylabel(clazz)
    plt.plot(qrs)

    plt.subplot(total, 1, 5)
    plt.ylabel("QRS DFT")
    ff = fft(qrs)[:len(qrs) // 2]
    plt.plot(ff, 'r')

    plt.subplot(total, 1, 6)
    plt.ylabel(clazz)
    plt.plot(t)

    plt.subplot(total, 1, 7)
    plt.ylabel("T DFT")
    ff = fft(t)[:len(t) // 2]
    plt.plot(ff, 'r')

    plt.show()


# Normal: A00001, A00002, A0003, A00006
plot_with_detected_peaks(loader.load_data_from_file("A07088"), "Normal")
plot_with_detected_peaks(loader.load_data_from_file("A00006"), "Normal")
plot_with_detected_peaks(loader.load_data_from_file("A08128"), "Normal")
# AF: A00004, A00009, A00015, A00027
plot_with_detected_peaks(loader.load_data_from_file("A00004"), "AF rhythm")
plot_with_detected_peaks(loader.load_data_from_file("A00009"), "AF rhythm")
# Other: A00005, A00008, A00013, A00017
plot_with_detected_peaks(loader.load_data_from_file("A00005"), "Other rhythm")
plot_with_detected_peaks(loader.load_data_from_file("A00008"), "Other rhythm")
# Noisy: A00205, A00585, A01006, A01070
plot_with_detected_peaks(loader.load_data_from_file("A00205"), "Noisy signal")
plot_with_detected_peaks(loader.load_data_from_file("A00585"), "Noisy signal")
