import matplotlib
from scipy import signal

from biosppyex.signals import ecg
from features import hrv
from features.qrs_detect import low_pass_filtering, high_pass_filtering
from loading import loader
from preprocessing import normalizer

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 3)


def plot_powers(x):
    x = normalizer.normalize_ecg(x)
    x = low_pass_filtering(x)
    x = high_pass_filtering(x)

    # templates = ecg.ecg(x, sampling_rate=300, show=False)['templates']

    fs, powers = signal.welch(x, loader.FREQUENCY)

    fs = fs[:40]
    powers = powers[:40]

    print(hrv.frequency_domain(x, fs=300))

    plt.plot(fs, powers)
    plt.xticks([2 * i for i in range(len(fs) // 2)])
    plt.grid()
    plt.show()


plot_powers(loader.load_data_from_file("A00001"))
plot_powers(loader.load_data_from_file("A07718"))
plot_powers(loader.load_data_from_file("A08523"))

plot_powers(loader.load_data_from_file("A00004"))
plot_powers(loader.load_data_from_file("A06746"))
plot_powers(loader.load_data_from_file("A07707"))

plot_powers(loader.load_data_from_file("A00013"))
plot_powers(loader.load_data_from_file("A06245"))
plot_powers(loader.load_data_from_file("A07908"))

plot_powers(loader.load_data_from_file("A01006"))
plot_powers(loader.load_data_from_file("A08402"))
plot_powers(loader.load_data_from_file("A04853"))
