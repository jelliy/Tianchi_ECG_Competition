import pywt
from sklearn.decomposition import TruncatedSVD
import numpy as np


def gray_code(n):
    res = [0]
    i = 0
    while i < n:  # 从2的0次方开始，
        res_inv = res[::-1]  # 求res的反向list
        res_inv = [x + pow(2, i) for x in res_inv]
        res = res + res_inv
        i += 1
    return res

def calc_energy(data,wavelet,level,order,count):
    # Construct wavelet packet
    wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    nodes = wp.get_level(level, order=order)
    #labels = [n.path for n in nodes]
    #values = np.array([n.data for n in nodes], 'd')

    values = np.array([np.square(n.data) for n in nodes])

    values = values.sum(axis=1)
    total_e = values.sum() + 0.0000001
    total_e_s = values[: count].sum()
    # print(v1/total_e)
    # print(total_e)
    array = [v / total_e for v in values]
    # sort by GrayCode
    #index = gray_code(level)
    #res = [array[i] for i in index]
    return array[: count]


def clac_waveletpacket_coefficient(data, wavelet, level, order,n_components=6):
    # Construct wavelet packet
    wp = pywt.WaveletPacket(data, wavelet, 'symmetric', maxlevel=level)
    nodes = wp.get_level(level, order=order)
    labels = [n.path for n in nodes]
    # values = np.array([n.data for n in nodes], 'd')
    values = np.array([n.data for n in nodes])
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd.fit(values)
    features = svd.singular_values_
    return features

'''
client = HbaseReader("127.0.0.1", "9090", "t1")

satrtTime = '2018-10-05 15:32:43'
endTime = '2018-10-05 15:36:43'
data = client.read(satrtTime, endTime)
'''