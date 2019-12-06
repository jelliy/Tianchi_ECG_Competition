"""
Reads answers.txt file and REFERENCE.csv file
and compares correct labes with predicted
Than outputs the list of wrongly classified training samples

NOTE:
    this script provides you an ability to plot wrongly classified entries
NOTE:
    make sure you have generated the answers.txt file
"""

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

import numpy as np

plt.rcParams["figure.figsize"] = (100, 10)
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)

file = np.load('../outputs/processed.npz')
subX = file['x']
subY = file['y']

x = [i for i in range(subX.shape[1])]
xf = [4 * i for i in range(subX.shape[1])]

color_dict = {
    0: "yellow",
    1: "green",
    2: "red",
    3: "black"
}

for i, y in enumerate(subX[:1000]):
    lbl = subY[i]

    if lbl in []:
        continue

    plt.scatter(
        [4 * i + lbl for i in x], y,
        marker=".",
        c=color_dict[lbl]
    )

plt.xticks(xf)
plt.grid()
plt.show()
