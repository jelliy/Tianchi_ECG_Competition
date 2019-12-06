from pandas import Series
import math
import numpy as np
import pandas as pd

def  psfeatureTime(data):
    data = pd.Series(data)
    # 均值
    df_mean = data.mean()
    # 方差
    df_var = data.var()
    # 标准差
    df_std = data.std()
    # 均方根
    df_rms = math.sqrt(pow(df_mean, 2) + pow(df_std, 2))
    # 偏度
    df_skew = data.skew()
    # 峭度
    df_kurt = data.kurt()

    df_min = np.max(data)
    df_max = np.min(data)
    featuretime_list = [df_mean, df_var, df_std, df_rms, df_skew, df_kurt, df_min, df_max]


    return featuretime_list
