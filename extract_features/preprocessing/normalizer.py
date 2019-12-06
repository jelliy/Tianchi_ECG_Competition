import numpy as np

from utils import parallel


def __remove_dc_component(ecg):
    mean = np.mean(ecg)
    # cancel DC components
    return ecg - mean


def max_normalization(ecg):
    return ecg / max(np.fabs(np.amin(ecg)), np.fabs(np.amax(ecg)))


def normalize_batch(array):
    return parallel.apply_async(array, normalize_ecg)


def normalize_ecg(ecg):
    """
    Normalizes to a range of [-1; 1]
    :param ecg: input signal
    :return: normalized signal
    """
    ecg = __remove_dc_component(ecg)
    ecg = max_normalization(ecg)
    return ecg
