# -*- coding: utf-8 -*-
"""
biosppyex.signals.ecg
-------------------

This module provides methods to process Electrocardiographic (ECG) signals.
Implemented code assumes a single-channel Lead I like ECG signal.

:copyright: (c) 2015-2017 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range, zip

# 3rd party
import numpy as np
import scipy.signal as ss

# local
from . import tools as st
from .. import utils

from math import ceil

from scipy.fftpack import diff
from scipy.signal import medfilt, lfilter
from scipy.signal import resample, filtfilt
from scipy.stats import mode

from utils import matlab, common
from utils.matlab import *

import pywt
import numpy as np
import scipy
def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.

    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.

    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma

def _universal_thresh(img, sigma):
    """ Universal threshold used by the VisuShrink method """
    return sigma*np.sqrt(2*np.log(img.size))

def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh

def _wavelet_threshold(data, wavelet, method=None, threshold=None,
                       sigma=None, mode='soft', wavelet_levels=None):
    """Perform wavelet thresholding.

    Parameters
    ----------
    image : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. If it is set to None, a
        user-specified ``threshold`` must be supplied instead.
    threshold : float, optional
        The thresholding value to apply during wavelet coefficient
        thresholding. The default value (None) uses the selected ``method`` to
        estimate appropriate threshold(s) for noise removal.
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels
        (see Notes below).

    Returns
    -------
    out : ndarray
        Denoised image.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`
    """
    wavelet = pywt.Wavelet(wavelet)

    data = np.array(data)
    # original_extent is used to workaround PyWavelets issue #80
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = tuple(slice(s) for s in data.shape)

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        dlen = wavelet.dec_len
        wavelet_levels = np.min(
            [pywt.dwt_max_level(s, dlen) for s in data.shape])

        # Skip coarsest wavelet scales (see Notes in docstring).
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(data, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(len(coeffs)+1, 1, 1)
    plt.plot(coeffs[0])
    for i in range(1,len(coeffs)):
        plt.subplot(len(coeffs)+1, 1, i+1)
        plt.plot(coeffs[i]['d'])
    plt.show()
    """

    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]['d' * data.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if method is not None and threshold is not None:
        print(("Thresholding method {} selected.  The user-specified threshold "
              "will be ignored.").format(method))

    if threshold is None:
        var = sigma**2
        if method is None:
            raise ValueError(
                "If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # The BayesShrink thresholds from [1]_ in docstring
            threshold = [{key: _bayes_thresh(level[key], var) for key in level}
                         for level in dcoeffs]
        elif method == "VisuShrink":
            # The VisuShrink thresholds from [2]_ in docstring
            threshold = _universal_thresh(data, sigma)
        else:
            raise ValueError("Unrecognized method: {}".format(method))

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode=mode) for key in level}
                           for level in dcoeffs]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode=mode) for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    denoised_coeffs[0] = np.zeros(len(denoised_coeffs[0]))
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def qrs_detect2(ecg, thres=0.6, ref_period=0.25, fs=300):
    """
    QRS detector based on the P&T method
    See: https://github.com/alistairewj/peak-detector/blob/master/sources/qrs_detect2.m
    :param ecg: one ecg channel on which to run the detector
    :param thres: energy threshold of the detector [arbitrary units]
    :param ref_period: refractory period in sec between two R-peaks [ms]
    :param fs: sampling frequency [Hz]
    :return: list, positions of R picks
    """

    WIN_SAMP_SZ = 7
    MED_SMOOTH_NB_COEFF = int(round(fs / 100))
    INT_NB_COEFF = int(round(WIN_SAMP_SZ * fs / 256))
    SEARCH_BACK = True
    MAX_FORCE = []
    MIN_AMP = 0.1

    NB_SAMP = len(ecg)
    ecg = np.maximum(ecg, 0)
    tm = list(frange(1 / fs, ceil(NB_SAMP / fs), 1 / fs))

    """
    # == Bandpass filtering for ECG signal
    # this sombrero hat has shown to give slightly better results than a
    # standard band-pass filter. Plot the frequency response to convince
    # yourself of what it does
    b1 = [-7.757327341237223e-05, -2.357742589814283e-04, -6.689305101192819e-04, -0.001770119249103,
          -0.004364327211358, -0.010013251577232, -0.021344241245400, -0.042182820580118, -0.077080889653194,
          -0.129740392318591, -0.200064921294891, -0.280328573340852, -0.352139052257134, - 0.386867664739069,
          -0.351974030208595, -0.223363323458050, 0, 0.286427448595213, 0.574058766243311,
          0.788100265785590, 0.867325070584078, 0.788100265785590, 0.574058766243311, 0.286427448595213, 0,
          -0.223363323458050, -0.351974030208595, -0.386867664739069, -0.352139052257134,
          -0.280328573340852, -0.200064921294891, -0.129740392318591, -0.077080889653194, -0.042182820580118,
          -0.021344241245400, -0.010013251577232, -0.004364327211358, -0.001770119249103, -6.689305101192819e-04,
          -2.357742589814283e-04, -7.757327341237223e-05]

    # NOTE: resample works differently than in matlab
    b1 = resample(b1, int(ceil(len(b1) * fs / 250)))
    bpfecg = np.transpose(filtfilt(b1, 1, ecg))
    """
    #sm_size = int(0.08 * fs)
    #bpfecg , _ = st.smoother(signal=ecg, kernel='hamming', size=sm_size, mirror=True)
    bpfecg = ecg
    #if (sum(abs(ecg - common.mode(ecg)) > MIN_AMP) / NB_SAMP) > 0.2:
    if True:
        """
        if 20% of the samples have an absolute amplitude which is higher
        than MIN_AMP then we are good to go
        """

        # == P&T operations
        dffecg = matlab.diff(np.transpose(bpfecg))
        sqrecg = [x * x for x in dffecg]
        intecg = lfilter(np.ones(INT_NB_COEFF), 1, sqrecg)
        mdfint = medfilt(intecg, [MED_SMOOTH_NB_COEFF])
        delay = int(ceil(INT_NB_COEFF / 2))
        mdfint = np.roll(mdfint, -delay)

        mdfintFidel = mdfint

        if NB_SAMP / fs > 90:
            xs = np.sort(mdfintFidel[fs:fs * 90])
        else:
            xs = np.sort(mdfintFidel[fs:])

        if len(MAX_FORCE) == 0:
            if NB_SAMP / fs > 10:
                ind_xs = ceil(98 / 100 * len(xs))
                en_thres = xs[ind_xs]
            else:
                ind_xs = ceil(99 / 100 * len(xs))
                en_thres = xs[ind_xs]
        else:
            en_thres = MAX_FORCE

        poss_reg = apply(mdfint, lambda x: x > (thres * en_thres))

        if len(poss_reg) == 0:
            poss_reg[10] = 1

        if SEARCH_BACK:
            try:
                # ind of samples above threshold
                indAboveThreshold = find(poss_reg, lambda x: x > 0)
                # compute RRv
                RRv = np.diff([tm[i] for i in indAboveThreshold])
                medRRv = mode([RRv[i] for i in find(RRv, lambda x: x > 0.01)]).mode[0]
                # missed a peak?
                indMissedBeat = find(RRv, lambda x: x > 1.5 * medRRv)
                # find interval onto which a beat might have been missed
                indStart = [indAboveThreshold[i] for i in indMissedBeat]
                indEnd = [indAboveThreshold[i + 1] for i in indMissedBeat]

                for i in range(len(indStart)):
                    # look for a peak on this interval by lowering the energy threshold
                    poss_reg[indStart[i]:indEnd[i]] = apply(mdfint[indStart[i]:indEnd[i]],
                                                            lambda x: x > 0.3 * thres * en_thres)
            except IndexError:
                pass

        left = find(diff(np.append([0], np.transpose(poss_reg))), lambda x: x == 1)
        right = find(diff(np.append(np.transpose(poss_reg), [0])), lambda x: x == -1)

        all = [(left, right) for left, right in zip(left, right) if left != right]

        left = [x[0] for x in all]
        right = [x[1] for x in all]

        nb_s = len(apply(left, lambda x: x < 30 * fs))
        loc = np.zeros(nb_s, dtype=np.int32)
        for j in range(nb_s):
            a, loc[j] = np_max(abs(bpfecg[left[j]:right[j]]))
            loc[j] = loc[j] + left[j]
        sign = np.mean([ecg[i] for i in loc])

        compt = 0
        NB_PEAKS = len(left)
        maxval = np.zeros(NB_PEAKS)
        maxloc = np.zeros(NB_PEAKS, dtype=np.int32)

        for i in range(NB_PEAKS):
            if sign > 0:
                v, l = np_max(ecg[left[i]:right[i]])
            else:
                v, l = np_min(ecg[left[i]:right[i]])

            maxval[compt] = v
            maxloc[compt] = l + left[i]

            if compt > 0:
                if maxloc[compt] - maxloc[compt - 1] < fs * ref_period and abs(maxval[compt]) < abs(maxval[compt - 1]):
                    continue
                elif maxloc[compt] - maxloc[compt - 1] < fs * ref_period and abs(maxval[compt]) >= abs(
                        maxval[compt - 1]):
                    maxloc[compt - 1] = maxloc[compt]
                    maxval[compt - 1] = maxval[compt]
                else:
                    compt += 1
            else:
                # if first peak then increment
                compt += 1

        # datapoints QRS positions
        qrs_pos = maxloc[:compt]
    else:
        qrs_pos = []
        sign = None
        en_thres = None

    return qrs_pos

def plot(ts, signal, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr):
    from .. import plotting
    plotting.plot_ecg(ts=ts,
                      raw=signal,
                      filtered=filtered,
                      rpeaks=rpeaks,
                      templates_ts=ts_tmpl,
                      templates=templates,
                      heart_rate_ts=ts_hr,
                      heart_rate=hr,
                      path=None,
                      show=True)


def find_max(signal,beats,sampling_rate):
    lim = 20
    r_beats = []
    sampling_rate = float(sampling_rate)
    length = len(signal)
    thres_ch = 0.85
    adjacency = 0.05 * sampling_rate
    for i in beats:
        error = [False, False]
        if i - lim < 0:
            window = signal[0:i + lim]
            add = 0
        elif i + lim >= length:
            window = signal[i - lim:length]
            add = i - lim
        else:
            window = signal[i - lim:i + lim]
            add = i - lim
        # meanval = np.mean(window)
        #w_peaks, _ = st.find_extrema(signal=window, mode='max')
        #w_negpeaks, _ = st.find_extrema(signal=window, mode='min')
        r_beats.append(np.argmax(window)+ add)

    return r_beats

def ecgex(signals=None, sampling_rate=500., show=True, min_num=3, filename=None):
    """Process a raw ECG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).

    """

    # check inputs
    if signals is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)

    filtered_list = []
    rpeaks_list = []
    for i in range(signals.shape[1]):
        signal = signals.iloc[:,[i]].values.reshape(-1)
        # normalizer
        from preprocessing import categorizer, balancer, normalizer
        # signal = normalizer.normalize_ecg(signal)
        # ensure numpy
        #signal = np.array(signal)

        # filter signal
        filtered = _wavelet_threshold(signal,'coif5',method='VisuShrink', mode='soft',wavelet_levels=8)
        filtered_list.append(filtered)

        # segment
        #rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
        rpeaks = qrs_detect2(filtered, thres=0.6, ref_period=0.25, fs=int(sampling_rate))
        # import wfdb
        # wfdb.plot_items(signal=filtered, ann_samp=[np.array(rpeaks)])
        if i in [0,1,4,5,6,7]:
            rpeaks_list += list(rpeaks)

    rpeaks_list.sort()
    d = np.diff(rpeaks_list)
    index_list = np.array([i +1 for i in np.where(d > 20)])
    index_list = np.insert(index_list,0,0)
    index_list = np.append(index_list,len(rpeaks_list))
    rpeaks = []
    for i in range(len(index_list)-1):
        a = np.array(rpeaks_list[index_list[i]:index_list[i+1]])
        if len(a) < min_num:
            continue
        rpeaks.append(int(np.median(a)))

    #if len(rpeaks) < 4:
    #    print(filename + "   --------------")

    # import wfdb
    #wfdb.plot_items(signal=filtered, ann_samp=[np.array(rpeaks)])
    """
    if len(rpeaks) > 3:
        dif = np.diff(rpeaks)
        std = np.array(dif).std()
        if dif[1] < 200:
            print(filename + "   " +str(std))
    """
    #寻找等差数列  漏了一个补上
    """
    dif = np.diff(rpeaks)
    tmp = list(dif)
    tmp.remove(max(tmp))
    tmp.remove(min(tmp))
    tmp_mean = int(np.array(tmp).mean())
    index = np.where(dif > tmp_mean*1.8)
    if len(index) > 0:
        a = index[0][0]
        rpeaks = np.insert(rpeaks,a+1,rpeaks[a]+tmp_mean)
    """
    out_list = []
    for i in range(signals.shape[1]):
        signal = signals.iloc[:, [i]].values.reshape(-1)
        filtered = filtered_list[i]

        """
        if len(rpeaks) < 4:
            new_rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
        else:
            new_rpeaks = find_max(signal,rpeaks,500)
        """
        new_rpeaks = find_max(signal, rpeaks, 500)
        # extract templates
        templates, new_rpeaks = extract_heartbeats(signal=filtered,
                                               rpeaks=new_rpeaks,
                                               sampling_rate=sampling_rate,
                                               before=0.2,
                                               after=0.4)

        # compute heart rate
        hr_idx, hr = st.get_heart_rate(beats=new_rpeaks,
                                       sampling_rate=sampling_rate,
                                       smooth=True,
                                       size=3)

        if len(hr_idx) > 0:
            # get time vectors
            length = len(signal)
            T = (length - 1) / sampling_rate
            ts = np.linspace(0, T, length, endpoint=False)
            ts_hr = ts[hr_idx]
            ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)
        else:
            ts = np.array([])
            ts_hr = np.array([])
            ts_tmpl = np.array([])

        # plot
        if show:
            plot(ts=ts,
                 signal=signal,
                 filtered=filtered,
                 rpeaks=new_rpeaks,
                 ts_tmpl=ts_tmpl,
                 templates=templates,
                 ts_hr=ts_hr,
                 hr=hr)

        # output
        args = (ts, filtered, new_rpeaks, ts_tmpl, templates, ts_hr, hr)
        names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
                 'heart_rate_ts', 'heart_rate')
        out = utils.ReturnTuple(args, names)
        out_list.append(out)
    return out_list, rpeaks

def ecg(signal=None, sampling_rate=500., show=True):
    """Process a raw ECG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered = _wavelet_threshold(signal,'coif5',method='VisuShrink', mode='soft',wavelet_levels=8)
    # segment
    #rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    rpeaks = qrs_detect2(filtered, thres=0.6, ref_period=0.25, fs=int(sampling_rate))
    """
    dif1 = np.diff(rpeaks1)
    std1 = np.array(dif1).std()
    rpeaks = rpeaks1
    if std1 > 30:
        rpeaks2 = qrs_detect2(filtered, thres=0.6, ref_period=0.25, fs=int(sampling_rate))
        if len(rpeaks2) > 2:
            dif2 = np.diff(rpeaks2)
            std2 = np.array(dif2).std()
            if std2 < std1:
                rpeaks = rpeaks2
                print("算法1:"+str(std1)+"    算法2:"+str(std2))
    """
    # extract templates
    templates, rpeaks = extract_heartbeats(signal=filtered,
                                           rpeaks=rpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.2,
                                           after=0.4)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=rpeaks,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    if len(hr_idx) > 0:
        # get time vectors
        length = len(signal)
        T = (length - 1) / sampling_rate
        ts = np.linspace(0, T, length, endpoint=False)
        ts_hr = ts[hr_idx]
        ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)
    else:
        ts = np.array([])
        ts_hr = np.array([])
        ts_tmpl = np.array([])

    # plot
    if show:
        plot(ts=ts,
             signal=signal,
             filtered=filtered,
             rpeaks=rpeaks,
             ts_tmpl=ts_tmpl,
             templates=templates,
             ts_hr=ts_hr,
             hr=hr)

    # output
    args = (ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr)
    names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
             'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def _extract_heartbeats(signal=None, rpeaks=None, before=200, after=400):
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    before : int, optional
        Number of samples to include before the R peak.
    after : int, optional
        Number of samples to include after the R peak.

    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.

    """

    R = np.sort(rpeaks)
    length = len(signal)
    templates = []
    newR = []

    for r in R:
        a = r - before
        if a < 0:
            continue
        b = r + after
        if b > length:
            break
        templates.append(signal[a:b])
        newR.append(r)

    templates = np.array(templates)
    newR = np.array(newR, dtype='int')

    return templates, newR


def extract_heartbeats(signal=None, rpeaks=None, sampling_rate=1000.,
                       before=0.2, after=0.4):
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    before : float, optional
        Window size to include before the R peak (seconds).
    after : int, optional
        Window size to include after the R peak (seconds).

    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rpeaks is None:
        raise TypeError("Please specify the input R-peak locations.")

    if before < 0:
        raise ValueError("Please specify a non-negative 'before' value.")
    if after < 0:
        raise ValueError("Please specify a non-negative 'after' value.")

    # convert delimiters to samples
    before = int(before * sampling_rate)
    after = int(after * sampling_rate)

    # get heartbeats
    templates, newR = _extract_heartbeats(signal=signal,
                                          rpeaks=rpeaks,
                                          before=before,
                                          after=after)

    return utils.ReturnTuple((templates, newR), ('templates', 'rpeaks'))


def compare_segmentation(reference=None, test=None, sampling_rate=1000.,
                         offset=0, minRR=None, tol=0.05):
    """Compare the segmentation performance of a list of R-peak positions
    against a reference list.

    Parameters
    ----------
    reference : array
        Reference R-peak location indices.
    test : array
        Test R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    offset : int, optional
        Constant a priori offset (number of samples) between reference and
        test R-peak locations.
    minRR : float, optional
        Minimum admissible RR interval (seconds).
    tol : float, optional
        Tolerance between corresponding reference and test R-peak
        locations (seconds).

    Returns
    -------
    TP : int
        Number of true positive R-peaks.
    FP : int
        Number of false positive R-peaks.
    performance : float
        Test performance; TP / len(reference).
    acc : float
        Accuracy rate; TP / (TP + FP).
    err : float
        Error rate; FP / (TP + FP).
    match : list
        Indices of the elements of 'test' that match to an R-peak
        from 'reference'.
    deviation : array
        Absolute errors of the matched R-peaks (seconds).
    mean_deviation : float
        Mean error (seconds).
    std_deviation : float
        Standard deviation of error (seconds).
    mean_ref_ibi : float
        Mean of the reference interbeat intervals (seconds).
    std_ref_ibi : float
        Standard deviation of the reference interbeat intervals (seconds).
    mean_test_ibi : float
        Mean of the test interbeat intervals (seconds).
    std_test_ibi : float
        Standard deviation of the test interbeat intervals (seconds).

    """

    # check inputs
    if reference is None:
        raise TypeError("Please specify an input reference list of R-peak \
                        locations.")

    if test is None:
        raise TypeError("Please specify an input test list of R-peak \
                        locations.")

    if minRR is None:
        minRR = np.inf

    sampling_rate = float(sampling_rate)

    # ensure numpy
    reference = np.array(reference)
    test = np.array(test)

    # convert to samples
    minRR = minRR * sampling_rate
    tol = tol * sampling_rate

    TP = 0
    FP = 0

    matchIdx = []
    dev = []

    for i, r in enumerate(test):
        # deviation to closest R in reference
        ref = reference[np.argmin(np.abs(reference - (r + offset)))]
        error = np.abs(ref - (r + offset))

        if error < tol:
            TP += 1
            matchIdx.append(i)
            dev.append(error)
        else:
            if len(matchIdx) > 0:
                bdf = r - test[matchIdx[-1]]
                if bdf < minRR:
                    # false positive, but removable with RR interval check
                    pass
                else:
                    FP += 1
            else:
                FP += 1

    # convert deviations to time
    dev = np.array(dev, dtype='float')
    dev /= sampling_rate
    nd = len(dev)
    if nd == 0:
        mdev = np.nan
        sdev = np.nan
    elif nd == 1:
        mdev = np.mean(dev)
        sdev = 0.
    else:
        mdev = np.mean(dev)
        sdev = np.std(dev, ddof=1)

    # interbeat interval
    th1 = 1.5  # 40 bpm
    th2 = 0.3  # 200 bpm

    rIBI = np.diff(reference)
    rIBI = np.array(rIBI, dtype='float')
    rIBI /= sampling_rate

    good = np.nonzero((rIBI < th1) & (rIBI > th2))[0]
    rIBI = rIBI[good]

    nr = len(rIBI)
    if nr == 0:
        rIBIm = np.nan
        rIBIs = np.nan
    elif nr == 1:
        rIBIm = np.mean(rIBI)
        rIBIs = 0.
    else:
        rIBIm = np.mean(rIBI)
        rIBIs = np.std(rIBI, ddof=1)

    tIBI = np.diff(test[matchIdx])
    tIBI = np.array(tIBI, dtype='float')
    tIBI /= sampling_rate

    good = np.nonzero((tIBI < th1) & (tIBI > th2))[0]
    tIBI = tIBI[good]

    nt = len(tIBI)
    if nt == 0:
        tIBIm = np.nan
        tIBIs = np.nan
    elif nt == 1:
        tIBIm = np.mean(tIBI)
        tIBIs = 0.
    else:
        tIBIm = np.mean(tIBI)
        tIBIs = np.std(tIBI, ddof=1)

    # output
    perf = float(TP) / len(reference)
    acc = float(TP) / (TP + FP)
    err = float(FP) / (TP + FP)

    args = (TP, FP, perf, acc, err, matchIdx, dev, mdev, sdev, rIBIm, rIBIs,
            tIBIm, tIBIs)
    names = ('TP', 'FP', 'performance', 'acc', 'err', 'match', 'deviation',
             'mean_deviation', 'std_deviation', 'mean_ref_ibi', 'std_ref_ibi',
             'mean_test_ibi', 'std_test_ibi',)

    return utils.ReturnTuple(args, names)


def ssf_segmenter(signal=None, sampling_rate=1000., threshold=20, before=0.03,
                  after=0.01):
    """ECG R-peak segmentation based on the Slope Sum Function (SSF).

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float, optional
        SSF threshold.
    before : float, optional
        Search window size before R-peak candidate (seconds).
    after : float, optional
        Search window size after R-peak candidate (seconds).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # convert to samples
    winB = int(before * sampling_rate)
    winA = int(after * sampling_rate)

    Rset = set()
    length = len(signal)

    # diff
    dx = np.diff(signal)
    dx[dx >= 0] = 0
    dx = dx ** 2

    # detection
    idx, = np.nonzero(dx > threshold)
    idx0 = np.hstack(([0], idx))
    didx = np.diff(idx0)

    # search
    sidx = idx[didx > 1]
    for item in sidx:
        a = item - winB
        if a < 0:
            a = 0
        b = item + winA
        if b > length:
            continue

        r = np.argmax(signal[a:b]) + a
        Rset.add(r)

    # output
    rpeaks = list(Rset)
    rpeaks.sort()
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def christov_segmenter(signal=None, sampling_rate=1000.):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Christov [Chri04]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [Chri04] Ivaylo I. Christov, "Real time electrocardiogram QRS
       detection using combined adaptive threshold", BioMedical Engineering
       OnLine 2004, vol. 3:28, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    length = len(signal)

    # algorithm parameters
    v100ms = int(0.1 * sampling_rate)
    v50ms = int(0.050 * sampling_rate)
    v300ms = int(0.300 * sampling_rate)
    v350ms = int(0.350 * sampling_rate)
    v200ms = int(0.2 * sampling_rate)
    v1200ms = int(1.2 * sampling_rate)
    M_th = 0.4  # paper is 0.6

    # Pre-processing
    # 1. Moving averaging filter for power-line interference suppression:
    # averages samples in one period of the powerline
    # interference frequency with a first zero at this frequency.
    b = np.ones(int(0.02 * sampling_rate)) / 50.
    a = [1]
    X = ss.filtfilt(b, a, signal)
    # 2. Moving averaging of samples in 28 ms interval for electromyogram
    # noise suppression a filter with first zero at about 35 Hz.
    b = np.ones(sampling_rate / 35.) / 35.
    X = ss.filtfilt(b, a, X)
    X, _, _ = st.filter_signal(signal=X,
                               ftype='butter',
                               band='lowpass',
                               order=7,
                               frequency=40.,
                               sampling_rate=sampling_rate)
    X, _, _ = st.filter_signal(signal=X,
                               ftype='butter',
                               band='highpass',
                               order=7,
                               frequency=9.,
                               sampling_rate=sampling_rate)

    k, Y, L = 1, [], len(X)
    for n in range(k + 1, L - k):
        Y.append(X[n] ** 2 - X[n - k] * X[n + k])
    Y = np.array(Y)
    Y[Y < 0] = 0

    # Complex lead
    # Y = abs(scipy.diff(X)) # 1-lead
    # 3. Moving averaging of a complex lead (the sintesis is
    # explained in the next section) in 40 ms intervals a filter
    # with first zero at about 25 Hz. It is suppressing the noise
    # magnified by the differentiation procedure used in the
    # process of the complex lead sintesis.
    b = np.ones(sampling_rate / 25.) / 25.
    Y = ss.lfilter(b, a, Y)

    # Init
    MM = M_th * np.max(Y[:5 * sampling_rate]) * np.ones(5)
    MMidx = 0
    M = np.mean(MM)
    slope = np.linspace(1.0, 0.6, int(sampling_rate))
    Rdec = 0
    R = 0
    RR = np.zeros(5)
    RRidx = 0
    Rm = 0
    QRS = []
    Rpeak = []
    current_sample = 0
    skip = False
    F = np.mean(Y[:v350ms])

    # Go through each sample
    while current_sample < len(Y):
        if QRS:
            # No detection is allowed 200 ms after the current one. In
            # the interval QRS to QRS+200ms a new value of M5 is calculated: newM5 = 0.6*max(Yi)
            if current_sample <= QRS[-1] + v200ms:
                Mnew = M_th * max(Y[QRS[-1]:QRS[-1] + v200ms])
                # The estimated newM5 value can become quite high, if
                # steep slope premature ventricular contraction or artifact
                # appeared, and for that reason it is limited to newM5 = 1.1*M5 if newM5 > 1.5* M5
                # The MM buffer is refreshed excluding the oldest component, and including M5 = newM5.
                Mnew = Mnew if Mnew <= 1.5 * MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                MM[MMidx] = Mnew
                MMidx = np.mod(MMidx + 1, 5)
                # M is calculated as an average value of MM.
                Mtemp = np.mean(MM)
                M = Mtemp
                skip = True
            # M is decreased in an interval 200 to 1200 ms following
            # the last QRS detection at a low slope, reaching 60 % of its
            # refreshed value at 1200 ms.
            elif current_sample >= QRS[-1] + v200ms and current_sample < QRS[-1] + v1200ms:
                M = Mtemp * slope[current_sample - QRS[-1] - v200ms]
            # After 1200 ms M remains unchanged.
            # R = 0 V in the interval from the last detected QRS to 2/3 of the expected Rm.
            if current_sample >= QRS[-1] and current_sample < QRS[-1] + (2 / 3.) * Rm:
                R = 0
            # In the interval QRS + Rm * 2/3 to QRS + Rm, R decreases
            # 1.4 times slower then the decrease of the previously discussed
            # steep slope threshold (M in the 200 to 1200 ms interval).
            elif current_sample >= QRS[-1] + (2 / 3.) * Rm and current_sample < QRS[-1] + Rm:
                R += Rdec
                # After QRS + Rm the decrease of R is stopped
                # MFR = M + F + R
        MFR = M + F + R
        # QRS or beat complex is detected if Yi = MFR
        if not skip and Y[current_sample] >= MFR:
            QRS += [current_sample]
            Rpeak += [QRS[-1] + np.argmax(Y[QRS[-1]:QRS[-1] + v300ms])]
            if len(QRS) >= 2:
                # A buffer with the 5 last RR intervals is updated at any new QRS detection.
                RR[RRidx] = QRS[-1] - QRS[-2]
                RRidx = np.mod(RRidx + 1, 5)
        skip = False
        # With every signal sample, F is updated adding the maximum
        # of Y in the latest 50 ms of the 350 ms interval and
        # subtracting maxY in the earliest 50 ms of the interval.
        if current_sample >= v350ms:
            Y_latest50 = Y[current_sample - v50ms:current_sample]
            Y_earliest50 = Y[current_sample - v350ms:current_sample - v300ms]
            F += (max(Y_latest50) - max(Y_earliest50)) / 1000.
        # Rm is the mean value of the buffer RR.
        Rm = np.mean(RR)
        current_sample += 1

    rpeaks = []
    for i in Rpeak:
        a, b = i - v100ms, i + v100ms
        if a < 0:
            a = 0
        if b > length:
            b = length
        rpeaks.append(np.argmax(signal[a:b]) + a)

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def engzee_segmenter(signal=None, sampling_rate=1000., threshold=0.48):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Engelse and Zeelenberg [EnZe79]_ with the
    modifications by Lourenco *et al.* [LSLL12]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float, optional
        Detection threshold.

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [EnZe79] W. Engelse and C. Zeelenberg, "A single scan algorithm for
       QRS detection and feature extraction", IEEE Comp. in Cardiology,
       vol. 6, pp. 37-42, 1979
    .. [LSLL12] A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred,
       "Real Time Electrocardiogram Segmentation for Finger Based ECG
       Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # algorithm parameters
    changeM = int(0.75 * sampling_rate)
    Miterate = int(1.75 * sampling_rate)
    v250ms = int(0.25 * sampling_rate)
    v1200ms = int(1.2 * sampling_rate)
    v1500ms = int(1.5 * sampling_rate)
    v180ms = int(0.18 * sampling_rate)
    p10ms = int(np.ceil(0.01 * sampling_rate))
    p20ms = int(np.ceil(0.02 * sampling_rate))
    err_kill = int(0.01 * sampling_rate)
    inc = 1
    mmth = threshold
    mmp = 0.2

    # Differentiator (1)
    y1 = [signal[i] - signal[i - 4] for i in range(4, len(signal))]

    # Low pass filter (2)
    c = [1, 4, 6, 4, 1, -1, -4, -6, -4, -1]
    y2 = np.array([np.dot(c, y1[n - 9:n + 1]) for n in range(9, len(y1))])
    y2_len = len(y2)

    # vars
    MM = mmth * max(y2[:Miterate]) * np.ones(3)
    MMidx = 0
    Th = np.mean(MM)
    NN = mmp * min(y2[:Miterate]) * np.ones(2)
    NNidx = 0
    ThNew = np.mean(NN)
    update = False
    nthfpluss = []
    rpeaks = []

    # Find nthf+ point
    while True:
        # If a previous intersection was found, continue the analysis from there
        if update:
            if inc * changeM + Miterate < y2_len:
                a = (inc - 1) * changeM
                b = inc * changeM + Miterate
                Mnew = mmth * max(y2[a:b])
                Nnew = mmp * min(y2[a:b])
            elif y2_len - (inc - 1) * changeM > v1500ms:
                a = (inc - 1) * changeM
                Mnew = mmth * max(y2[a:])
                Nnew = mmp * min(y2[a:])
            if len(y2) - inc * changeM > Miterate:
                MM[MMidx] = Mnew if Mnew <= 1.5 * MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                NN[NNidx] = Nnew if abs(Nnew) <= 1.5 * abs(NN[NNidx - 1]) else 1.1 * NN[NNidx - 1]
            MMidx = np.mod(MMidx + 1, len(MM))
            NNidx = np.mod(NNidx + 1, len(NN))
            Th = np.mean(MM)
            ThNew = np.mean(NN)
            inc += 1
            update = False
        if nthfpluss:
            lastp = nthfpluss[-1] + 1
            if lastp < (inc - 1) * changeM:
                lastp = (inc - 1) * changeM
            y22 = y2[lastp:inc * changeM + err_kill]
            # find intersection with Th
            try:
                nthfplus = np.intersect1d(np.nonzero(y22 > Th)[0], np.nonzero(y22 < Th)[0] - 1)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
            # adjust index
            nthfplus += int(lastp)
            # if a previous R peak was found:
            if rpeaks:
                # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                if nthfplus - rpeaks[-1] > v250ms and nthfplus - rpeaks[-1] < v1200ms:
                    pass
                # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                elif nthfplus - rpeaks[-1] < v250ms:
                    nthfpluss += [nthfplus]
                    continue
        # no previous intersection, find the first one
        else:
            try:
                aux = np.nonzero(y2[(inc - 1) * changeM:inc * changeM + err_kill] > Th)[0]
                bux = np.nonzero(y2[(inc - 1) * changeM:inc * changeM + err_kill] < Th)[0] - 1
                nthfplus = int((inc - 1) * changeM) + np.intersect1d(aux, bux)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
        nthfpluss += [nthfplus]
        # Define 160ms search region
        windowW = np.arange(nthfplus, nthfplus + v180ms)
        # Check if the condition y2[n] < Th holds for a specified
        # number of consecutive points (experimentally we found this number to be at least 10 points)"
        i, f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
        hold_points = np.diff(np.nonzero(y2[i:f] < ThNew)[0])
        cont = 0
        for hp in hold_points:
            if hp == 1:
                cont += 1
                if cont == p10ms - 1:  # -1 is because diff eats a sample
                    max_shift = p20ms  # looks for X's max a bit to the right
                    if nthfpluss[-1] > max_shift:
                        rpeaks += [np.argmax(signal[i - max_shift:f]) + i - max_shift]
                    else:
                        rpeaks += [np.argmax(signal[i:f]) + i]
                    break
            else:
                cont = 0

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def gamboa_segmenter(signal=None, sampling_rate=1000., tol=0.002):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Gamboa.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    tol : float, optional
        Tolerance parameter.

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # convert to samples
    v_100ms = int(0.1 * sampling_rate)
    v_300ms = int(0.3 * sampling_rate)
    hist, edges = np.histogram(signal, 100, density=True)

    TH = 0.01
    F = np.cumsum(hist)

    v0 = edges[np.nonzero(F > TH)[0][0]]
    v1 = edges[np.nonzero(F < (1 - TH))[0][-1]]

    nrm = max([abs(v0), abs(v1)])
    norm_signal = signal / float(nrm)

    d2 = np.diff(norm_signal, 2)

    b = np.nonzero((np.diff(np.sign(np.diff(-d2)))) == -2)[0] + 2
    b = np.intersect1d(b, np.nonzero(-d2 > tol)[0])

    if len(b) < 3:
        rpeaks = []
    else:
        b = b.astype('float')
        rpeaks = []
        previous = b[0]
        for i in b[1:]:
            if i - previous > v_300ms:
                previous = i
                rpeaks.append(np.argmax(signal[i:i + v_100ms]) + i)

    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def hamilton_segmenter(signal=None, sampling_rate=1000.):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Hamilton [Hami02]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [Hami02] P.S. Hamilton, "Open Source ECG Analysis Software
       Documentation", E.P.Limited, 2002

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)
    length = len(signal)
    dur = length / sampling_rate

    # algorithm parameters
    v1s = int(1. * sampling_rate)
    v100ms = int(0.1 * sampling_rate)
    TH_elapsed = np.ceil(0.36 * sampling_rate)
    sm_size = int(0.08 * sampling_rate)
    init_ecg = 10  # seconds for initialization
    if dur < init_ecg:
        init_ecg = int(dur)

    """
    # filtering
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='lowpass',
                                      order=4,
                                      frequency=25.,
                                      sampling_rate=sampling_rate)
    filtered, _, _ = st.filter_signal(signal=filtered,
                                      ftype='butter',
                                      band='highpass',
                                      order=4,
                                      frequency=3.,
                                      sampling_rate=sampling_rate)

    # diff
    dx = np.abs(np.diff(filtered, 1) * sampling_rate)
    """
    #signal = np.maximum(signal, 0)
    dx = signal
    # smoothing
    dx, _ = st.smoother(signal=dx, kernel='hamming', size=sm_size, mirror=True)
    #dx, _ = st.smoother(signal=dx, kernel='flattop', size=sm_size, mirror=True)
    # buffers
    qrspeakbuffer = np.zeros(init_ecg)
    noisepeakbuffer = np.zeros(init_ecg)
    peak_idx_test = np.zeros(init_ecg, dtype="int")
    noise_idx = np.zeros(init_ecg)
    rrinterval = sampling_rate * np.ones(init_ecg)

    a, b = 0, v1s
    all_peaks, _ = st.find_extrema(signal=dx, mode='max')
    # import wfdb
    #wfdb.plot_items(signal=dx, ann_samp=[all_peaks])

    for i in range(init_ecg):
        peaks, values = st.find_extrema(signal=dx[a:b], mode='max')
        try:
            ind = np.argmax(values)
        except ValueError:
            pass
        else:
            # peak amplitude
            qrspeakbuffer[i] = values[ind]
            # peak location
            peak_idx_test[i] = peaks[ind] + a

        a += v1s
        b += v1s

    #wfdb.plot_items(signal=dx, ann_samp=[peak_idx_test])
    # thresholds
    ANP = np.median(noisepeakbuffer)
    AQRSP = np.median(qrspeakbuffer)
    TH = 0.475
    DT = ANP + TH * (AQRSP - ANP)
    DT_vec = []
    indexqrs = 0
    indexnoise = 0
    indexrr = 0
    npeaks = 0
    offset = 0

    beats = []

    # detection rules
    # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
    lim = int(np.ceil(0.3 * sampling_rate))
    diff_nr = int(np.ceil(0.045 * sampling_rate))
    bpsi, bpe = offset, 0

    for f in all_peaks:
        DT_vec += [DT]
        # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = np.array((all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f))
        peaks_within = all_peaks[peak_cond]
        if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
            continue

        # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
        if dx[f] > DT:
            # 2 - look for both positive and negative slopes in raw signal
            if f < diff_nr:
                diff_now = np.diff(signal[0:f + diff_nr])
            elif f + diff_nr >= len(signal):
                diff_now = np.diff(signal[f - diff_nr:len(dx)])
            else:
                diff_now = np.diff(signal[f - diff_nr:f + diff_nr])
            diff_signer = diff_now[diff_now > 0]
            if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                continue
            # RR INTERVALS
            if npeaks > 0:
                # 3 - in here we check point 3 of the Hamilton paper
                # that is, we check whether our current peak is a valid R-peak.
                prev_rpeak = beats[npeaks - 1]

                elapsed = f - prev_rpeak
                # if the previous peak was within 360 ms interval
                if elapsed < TH_elapsed:
                    # check current and previous slopes
                    if prev_rpeak < diff_nr:
                        diff_prev = np.diff(signal[0:prev_rpeak + diff_nr])
                    elif prev_rpeak + diff_nr >= len(signal):
                        diff_prev = np.diff(signal[prev_rpeak - diff_nr:len(dx)])
                    else:
                        diff_prev = np.diff(signal[prev_rpeak - diff_nr:prev_rpeak + diff_nr])

                    slope_now = max(diff_now)
                    slope_prev = max(diff_prev)

                    if (slope_now < 0.5 * slope_prev):
                        # if current slope is smaller than half the previous one, then it is a T-wave
                        continue
                if dx[f] < 3. * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                    beats += [int(f) + bpsi]
                else:
                    continue

                if bpe == 0:
                    rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                    indexrr += 1
                    if indexrr == init_ecg:
                        indexrr = 0
                else:
                    if beats[npeaks] > beats[bpe - 1] + v100ms:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0

            elif dx[f] < 3. * np.median(qrspeakbuffer):
                beats += [int(f) + bpsi]
            else:
                continue

            npeaks += 1
            qrspeakbuffer[indexqrs] = dx[f]
            peak_idx_test[indexqrs] = f
            indexqrs += 1
            if indexqrs == init_ecg:
                indexqrs = 0
        if dx[f] <= DT:
            # 4 - not valid
            # 5 - If no QRS has been detected within 1.5 R-to-R intervals,
            # there was a peak that was larger than half the detection threshold,
            # and the peak followed the preceding detection by at least 360 ms,
            # classify that peak as a QRS complex
            tf = f + bpsi
            # RR interval median
            RRM = np.median(rrinterval)  # initial values are good?

            if len(beats) >= 2:
                elapsed = tf - beats[npeaks - 1]

                if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                    if dx[f] > 0.5 * DT:
                        beats += [int(f) + offset]
                        # RR INTERVALS
                        if npeaks > 0:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0
                        npeaks += 1
                        qrspeakbuffer[indexqrs] = dx[f]
                        peak_idx_test[indexqrs] = f
                        indexqrs += 1
                        if indexqrs == init_ecg:
                            indexqrs = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0
            else:
                noisepeakbuffer[indexnoise] = dx[f]
                noise_idx[indexnoise] = f
                indexnoise += 1
                if indexnoise == init_ecg:
                    indexnoise = 0

        # Update Detection Threshold
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        DT = ANP + 0.475 * (AQRSP - ANP)

    beats = np.array(beats)
    #wfdb.plot_items(signal=dx, ann_samp=[beats])

    lim = lim
    r_beats = []
    thres_ch = 0.85
    adjacency = 0.05 * sampling_rate
    for i in beats:
        error = [False, False]
        if i - lim < 0:
            window = signal[0:i + lim]
            add = 0
        elif i + lim >= length:
            window = signal[i - lim:length]
            add = i - lim
        else:
            window = signal[i - lim:i + lim]
            add = i - lim
        # meanval = np.mean(window)
        w_peaks, _ = st.find_extrema(signal=window, mode='max')
        w_negpeaks, _ = st.find_extrema(signal=window, mode='min')
        zerdiffs = np.where(np.diff(window) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

        pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
        negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))

        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            pass
        try:
            twonegpeaks = [negpeaks[0]]
        except IndexError:
            pass

        # getting positive peaks
        for i in range(len(pospeaks) - 1):
            if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                twopeaks.append(pospeaks[i + 1])
                break
        try:
            posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
        except IndexError:
            error[0] = True

        # getting negative peaks
        for i in range(len(negpeaks) - 1):
            if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                twonegpeaks.append(negpeaks[i + 1])
                break
        try:
            negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
        except:
            error[1] = True

        r_beats.append(twopeaks[0][1] + add)
        """
        # choosing type of R-peak
        if not sum(error):
            if posdiv > thres_ch * negdiv:
                # pos noerr
                r_beats.append(twopeaks[0][1] + add)
            else:
                # neg noerr
                r_beats.append(twonegpeaks[0][1] + add)
        elif sum(error) == 2:
            try:
                if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                    # pos allerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg allerr
                    r_beats.append(twonegpeaks[0][1] + add)
            except:
                pass
        elif error[0]:
            # pos poserr
            r_beats.append(twopeaks[0][1] + add)
        else:
            # neg negerr
            r_beats.append(twonegpeaks[0][1] + add)
        """
    rpeaks = sorted(list(set(r_beats)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))
