import random
import numpy as np
from collections import Iterable

from scipy import stats


def set_seed(seed=None):
    if seed is None:
        seed = int(random.random() * 1e6)
    random.seed(seed)
    np.random.seed(seed)
    print("Seed =", seed)


def mode(a):
    return stats.mode(a, axis=None)[0][0]


def shuffle_data(data, labels):
    """
    Shuffles input data

    In some cases input data might be distributed sorted which might create a hidden error
    in training/validation process so it's better to always shuffle input data before usage
    :return: Shuffled input data
    """
    data_shuf = []
    labels_shuf = []
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        labels_shuf.append(labels[i])
    return (np.array(data_shuf), labels_shuf)


def trimboth(x: Iterable, percent = 0.1):
    n = int(percent * len(x))
