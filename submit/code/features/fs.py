from scipy.fftpack import rfft
import numpy as np


def extract_fft(x):
    return rfft(x)[:len(x) // 2]