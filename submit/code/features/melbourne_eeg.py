import numpy as np


def calcNormalizedFFT(epoch, lvl, nt, fs):
    """
    Calculates the FFT of the epoch signal.
    Removes the DC component and normalizes the area to 1
    :param epoch - signal
    :param lvl - frequency levels
    :param nt - length of the signal
    :param fs - sampling frequency
    """
    lseg = np.round(nt / fs * lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    D[0, :] = 0  # set the DC component to zero
    D /= D.sum()  # Normalize each channel

    return D


def calcDSpect(epoch, lvl, nt, nc, fs):
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    lseg = np.round(nt / fs * lvl).astype('int')

    dspect = np.zeros((len(lvl) - 1, nc))
    for j in range(len(dspect)):
        dspect[j, :] = 2 * np.sum(D[lseg[j]:lseg[j + 1], :], axis=0)

    return dspect


def calcShannonEntropy(epoch, lvl, nt, nc, fs):
    """
    Computes Shannon Entropy
    """
    # compute Shannon's entropy, spectral edge and correlation matrix
    # segments corresponding to frequency bands
    dspect = calcDSpect(epoch, lvl, nt, nc, fs)

    # Find the shannon's entropy
    spentropy = -1 * np.sum(np.multiply(dspect, np.log(dspect)), axis=0)

    return spentropy


def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs):
    """
    Compute spectral edge frequency
    """
    # Find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = 0.5

    topfreq = int(round(nt / sfreq * tfreq)) + 1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq, :], axis=0)
    B = A - (A.max() * ppow)
    spedge = np.min(np.abs(B), axis=0)
    spedge = (spedge - 1) / (topfreq - 1) * tfreq

    return spedge


def corr(data, type_corr):
    """
    Calculate cross-correlation matrix
    """
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0
    w, v = np.linalg.eig(C)
    # print(w)
    x = np.sort(w)
    x = np.real(x)
    return x


def calcActivity(epoch):
    """
    Calculate Hjorth activity over epoch
    """
    return np.nanvar(epoch, axis=0)


def calcMobility(epoch):
    """
    Calculate the Hjorth mobility parameter over epoch
    """
    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    return np.divide(
        np.nanstd(np.diff(epoch, axis=0)),
        np.nanstd(epoch, axis=0))


def calcComplexity(epoch):
    """
    Calculate Hjorth complexity over epoch
    """
    return np.divide(
        calcMobility(np.diff(epoch, axis=0)),
        calcMobility(epoch))


def hurstFD(epoch):
    """
    Returns the Hurst Exponent of the time series vector ts
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.nanstd(np.subtract(epoch[lag:], epoch[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0
