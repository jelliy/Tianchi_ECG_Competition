import numpy as np
from collections import Iterable
from scipy import interpolate
from scipy.signal import welch


def time_domain(rri: Iterable):
    """
    Computes time domain characteristics of heart rate:

    - RMSSD, Root mean square of successive differences
    - NN50, Number of pairs of successive NN intervals that differ by more than 50ms
    - pNN50, Proportion of NN50 divided by total number of NN intervals
    - NN20, Number of pairs of successive NN intervals that differ by more than 20ms
    - pNN20, Proportion of NN20 divided by total number of NN intervals
    - SDNN, standard deviation of NN intervals
    - mRRi, mean length of RR interval
    - stdRRi, mean length of RR intervals
    - mHR, mean heart rate

    :param rri: RR intervals in ms
    :return: dictionary with computed characteristics
    :rtype: dict
    """
    rmssd = 0
    sdnn = 0
    nn20 = 0
    pnn20 = 0
    nn50 = 0
    pnn50 = 0
    mrri = 0
    stdrri = 0
    mhr = 0

    if len(rri) > 0:
        diff_rri = np.diff(rri)
        if len(diff_rri) > 0:
            # Root mean square of successive differences
            rmssd = np.sqrt(np.mean(diff_rri ** 2))
            # Number of pairs of successive NNs that differ by more than 50ms
            nn50 = sum(abs(diff_rri) > 50)
            # Proportion of NN50 divided by total number of NNs
            pnn50 = (nn50 / len(diff_rri)) * 100

            # Number of pairs of successive NNs that differe by more than 20ms
            nn20 = sum(abs(diff_rri) > 20)
            # Proportion of NN20 divided by total number of NNs
            pnn20 = (nn20 / len(diff_rri)) * 100

        # Standard deviation of NN intervals
        sdnn = np.std(rri, ddof=1)  # make it calculates N-1
        # Mean of RR intervals
        mrri = np.mean(rri)
        # Std of RR intervals
        stdrri = np.std(rri)
        # Mean heart rate, in ms
        mhr = 60 * 1000.0 / mrri

    keys = ['rmssd', 'sdnn', 'nn20', 'pnn20', 'nn50', 'pnn50', 'mrri', 'stdrri', 'mhr']
    values = [rmssd, sdnn, nn20, pnn20, nn50, pnn50, mrri, stdrri, mhr]
    values = np.round(values, 2)
    values = np.nan_to_num(values)

    return dict(zip(keys, values))


def frequency_domain(x: Iterable, fs: float = 1000,
                     vlf_band=(0, 4),
                     lf_band=(4, 15),
                     hf_band=(15, 40)
                     ) -> dict:
    """
    Interpolates a signal and performs Welch estimation of power spectrum
    for VLF, LF and HF bands

    :param x: signal
    :param fs: signal frequency
    :param vlf_band: very low frequencies band interval
    :param lf_band: low frequencies band interval
    :param hf_band: high frequencies band interval
    :return: computed frequency domain components of heart rate variability
    """

    fxx, pxx = welch(x, fs=fs)

    vlf_indexes = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
    lf_indexes = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
    hf_indexes = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])

    vlf = np.sum(pxx[vlf_indexes])
    lf = np.sum(pxx[lf_indexes])
    hf = np.sum(pxx[hf_indexes])

    total_power = vlf + lf + hf

    if hf != 0:
        lf_hf = lf / hf
    else:
        lf_hf = 0

    if total_power - vlf != 0:
        lfnu = (lf / (total_power - vlf)) * 100
        hfnu = (hf / (total_power - vlf)) * 100
    else:
        lfnu = 0
        hfnu = 0

    keys = ['total_power', 'vlf', 'lf', 'hf', 'lf_hf', 'lfnu', 'hfnu']
    values = [total_power, vlf, lf, hf, lf_hf, lfnu, hfnu]
    values = np.round(values, 2)
    values = np.nan_to_num(values)

    return dict(zip(keys, values))
