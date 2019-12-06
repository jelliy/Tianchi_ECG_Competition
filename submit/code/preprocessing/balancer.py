from collections import Counter

import numpy as np

from utils import matlab
from utils.common import shuffle_data


def show_balancing(y):
    counter = Counter(y)
    for key in sorted(list(counter.keys())):
        print(key, counter[key])


def balance(x, y):
    uniq = np.unique(y)

    selected = dict()

    for val in uniq:
        selected[val] = [x[i] for i in matlab.find(y, lambda v: v == val)]

    min_len = min([len(x) for x in selected.values()])

    x = []
    y = []
    for (key, value) in selected.items():
        x += value[:min_len]
        y += [key for i in range(min_len)]

    x, y = shuffle_data(x, y)

    return x, y


def balance2(x, y):
    uniq = np.unique(y)

    selected = dict()

    for val in uniq:
        selected[val] = [x[i] for i in matlab.find(y, lambda v: v == val)]

    min_len = 6 * min([len(x) for x in selected.values()])

    x = []
    y = []
    for (key, value) in selected.items():
        slen = min(len(value), min_len)
        x += value[:slen]
        y += [key for i in range(slen)]

    x, y = shuffle_data(x, y)

    return x, y
