from math import ceil

from scipy.fftpack import diff
from scipy.signal import medfilt, lfilter
from scipy.signal import resample, filtfilt
from scipy.stats import mode

from utils import matlab, common
from utils.matlab import *


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

    tm = list(frange(1 / fs, ceil(NB_SAMP / fs), 1 / fs))

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

    if (sum(abs(ecg - common.mode(ecg)) > MIN_AMP) / NB_SAMP) > 0.2:
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
