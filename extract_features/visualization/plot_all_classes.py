"""
This script plots different stages of normalization of the signal
"""

import random

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from loading import loader
from features.qrs_detect import *

plt.rcParams["figure.figsize"] = (14, 5)

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

a_ecg = loader.load_data_from_file("A01171")
n_ecg = loader.load_data_from_file("A03823")
o_ecg = loader.load_data_from_file("A07915")
s_ecg = loader.load_data_from_file("A04946")

total = 4
plt.subplot(total, 1, 1)
plt.ylabel("A-Fib")
plt.plot(a_ecg)

plt.subplot(total, 1, 2)
plt.ylabel("Normal")
plt.plot(n_ecg)

plt.subplot(total, 1, 3)
plt.ylabel("Other")
plt.plot(o_ecg)

plt.subplot(total, 1, 4)
plt.ylabel("Noisy")
plt.plot(s_ecg)

plt.show()