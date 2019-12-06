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

from biosppyex.signals import ecg
from utils.system import mkdir

matplotlib.use("Qt5Agg")

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.figsize"] = (4, 10)
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

from loading import loader


def __draw_to_file(file, times, values):
    plt.subplot(3, 1, 1)
    plt.plot(times, values.T, 'm', linewidth=1.5, alpha=0.7)
    plt.title("Templates")

    plt.subplot(3, 1, 2)
    plt.plot(times, [np.mean(x) for x in values.T], 'm', linewidth=1.5, alpha=0.7)
    plt.title("Mean")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(times, [np.median(x) for x in values.T], 'm', linewidth=1.5, alpha=0.7)
    plt.title("Median")
    plt.grid()

    # plt.show()
    plt.savefig(file, bbox_inches='tight', pad_inches=0.1)
    plt.close()


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

    print("Plotting misclassified")
    base_dir = "../outputs/misclassified"

    mkdir(base_dir)

    misclassified = sorted(misclassified, key=lambda t: t[2])
    for item in misclassified:
        print("Saving to " + item[2])
        row = loader.load_data_from_file(item[2])
        [ts, fts, rpeaks, tts, thb, hrts, hr] = ecg.ecg(signal=row, sampling_rate=loader.FREQUENCY, show=False)
        __draw_to_file(
            base_dir + "/" + item[2] + "_" + item[0] + "_" + item[1] + ".png",
            tts,
            thb
        )
