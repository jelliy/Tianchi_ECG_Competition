"""
Reads answers.txt file and REFERENCE.csv file
and compares correct labes with predicted
Than outputs the list of wrongly classified training samples

NOTE:
    this script provides you an ability to plot wrongly classified entries
NOTE:
    make sure you have generated the answers.txt file
"""

import csv

import matplotlib
from pywt import wavedec

from biosppyex.signals import ecg
import numpy as np
from features import heartbeats
from features import qrs_detect
from preprocessing import normalizer

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.figsize"] = (20, 6)

from loading import loader


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window wit`h the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

with open('../answers.txt') as predicted, \
        open('../validation/REFERENCE.csv') as correct:
    preader = csv.reader(predicted)
    creader = csv.reader(correct)

    ypred = []
    ytrue = []

    acc = dict()
    info = dict()

    misclassified = []

    for (p, c) in zip(preader, creader):
        (record, pred_label) = p
        true_label = c[1]
        ypred.append(pred_label)
        ytrue.append(true_label)

        if p != c:
            info[record] = "Pred=" + pred_label + " True=" + true_label
            misclassified.append((true_label, pred_label, record))
        else:
            info[record] = "Correct=" + true_label

    misclassified = sorted(misclassified, key=lambda t: t[0] + t[1])
    for item in misclassified:
        print(item[2], 'is of class', item[0], 'but was classified as', item[1])

    print(classification_report(ytrue, ypred))

    matrix = confusion_matrix(ytrue, ypred)
    print(matrix)
    for row in matrix:
        amax = sum(row)
        if amax > 0:
            for i in range(len(row)):
                row[i] = row[i] * 100.0 / amax

    print(matrix)

    while (True):
        name = input("Enter an entry name to plot: ")
        name = name.strip()
        if len(name) == 0:
            print("Finishing")
            break

        if not loader.check_has_example(name):
            print("File Not Found")
            continue

        row = loader.load_data_from_file(name)

        print(info[name])
        [ts, fts, rpeaks, tts, thb, hrts, hr] = ecg.ecg(signal=row, sampling_rate=loader.FREQUENCY, show=True)
        b = heartbeats.median_heartbeat(thb)
        t = b[int(0.3 * loader.FREQUENCY):int(0.55 * loader.FREQUENCY)]

        a = normalizer.normalize_ecg(t)

        a[a < 0] = 0

        a = np.square(a)

        a -= np.mean(a)

        a[a < 0] = 0

        a, d1 = wavedec(a, 'sym4', level=1)

        # a = normalizer.normalize_ecg(a)

        plt.subplot(3, 1, 1)
        plt.plot(b)

        plt.subplot(3, 1, 2)
        plt.plot(t)

        plt.subplot(3, 1, 3)
        plt.plot(a)
        plt.show()