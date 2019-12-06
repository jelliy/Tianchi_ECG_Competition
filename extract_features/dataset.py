# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal


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

from biosppy.signals.tools import filter_signal
def _apply_filter(signal, filter_bandwidth):

    # Calculate filter order
    order = int(0.3 * 500)

    # Filter signal
    signal, _, _ = filter_signal(signal=signal,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=filter_bandwidth,
                                 sampling_rate=500)

    return signal

def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    #sig1 = signal.resample(sig, target_point_num) if target_point_num else sig
    """
    sig_list = []
    for i in range(sig.shape[1]):
        a = sig[:,i]
        #sig_list.append(_apply_filter(a, [3, 45]))
        sig_list.append(_wavelet_threshold(a, 'coif5', method='VisuShrink', mode='soft', wavelet_levels=7))
    sig = np.array(sig_list).T
    """
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)
    # # 数据增强
    """
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    """
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(config.train_data)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(config.train_dir, fid)
        df = pd.read_csv(file_path, sep=' ').values
        x = transform(df, self.train)
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    print(d[0])
