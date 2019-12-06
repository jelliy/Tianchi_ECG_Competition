import itertools
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

from biosppyex.signals import ecg
from features import hrv, heartbeats, fs
from features.melbourne_eeg import calcActivity, calcMobility, calcComplexity
from loading import loader
from utils import common, matlab


def frequency_powers(x, n_power_features=40):
    fxx, pxx = signal.welch(x, loader.FREQUENCY)
    features = dict()
    for i, v in enumerate(pxx[:n_power_features]):
        features['welch' + str(i)] = v

    return features


def frequency_powers_summary(x):
    ecg_fs_range = (0, 50)
    band_size = 5

    features = dict()

    fxx, pxx = signal.welch(x, loader.FREQUENCY)
    for i in range((ecg_fs_range[1] - ecg_fs_range[0]) // 5):
        fs_min = i * band_size
        fs_max = fs_min + band_size
        indices = np.logical_and(fxx >= fs_min, fxx < fs_max)
        bp = np.sum(pxx[indices])
        features["power_" + str(fs_min) + "_" + str(fs_max)] = bp

    return features


def fft_features(beat):
    pff = fs.extract_fft(beat[:int(0.13 * loader.FREQUENCY)])
    rff = fs.extract_fft(beat[int(0.13 * loader.FREQUENCY):int(0.27 * loader.FREQUENCY)])
    tff = fs.extract_fft(beat[int(0.27 * loader.FREQUENCY):])

    features = dict()
    for i, v in enumerate(pff[:10]):
        features['pft' + str(i)] = v

    for i, v in enumerate(rff[:10]):
        features['rft' + str(i)] = v

    for i, v in enumerate(tff[:20]):
        features['tft' + str(i)] = v

    return features


def heart_rate_features(hr):
    features = {
        'hr_max': 0,
        'hr_min': 0,
        'hr_mean': 0,
        'hr_median': 0,
        'hr_mode': 0,
        'hr_std': 0,
        'hr_skew': 0,
        'hr_kurtosis': 0,
        'hr_range': 0,
        'hr_count': 0
    }

    if len(hr) > 0:
        features['hr_max'] = np.amax(hr)
        features['hr_min'] = np.amin(hr)
        features['hr_mean'] = np.mean(hr)
        features['hr_median'] = np.median(hr)
        features['hr_mode'] = common.mode(hr)
        features['hr_std'] = np.std(hr)
        features['hr_skew'] = skew(hr)
        features['hr_kurtosis'] = kurtosis(hr)
        features['hr_range'] = np.amax(hr) - np.amin(hr)
        features['hr_count'] = len(hr)
    return features


def heart_beats_features(thb):
    means = heartbeats.median_heartbeat(thb)
    mins = np.array([col.min() for col in thb.T])
    maxs = np.array([col.max() for col in thb.T])
    # stds = np.array([col.std() for col in thb.T])
    diff = maxs - mins

    features = dict()
    for i, v in enumerate(means):
        features['median' + str(i)] = v

    for i, v in enumerate(diff):
        features['hbdiff' + str(i)] = v

    return features


def heart_beats_features2(thb, rpeaks):
    if len(thb) == 0:
        thb = np.zeros((1, int(0.6 * loader.FREQUENCY)), dtype=np.int32)

    diff_rpeak = np.median(np.diff(rpeaks))
    means = heartbeats.median_heartbeat(thb)
    r_pos = int(0.2 * loader.FREQUENCY)

    if diff_rpeak < 350:
        tmp1 = 100-int(diff_rpeak/3)
        tmp2 = 100 + int(2 * diff_rpeak / 3)
        for i in range(len(means)):
            if i < tmp1:
                means[i] = 0
            elif i > tmp2:
                means[i] = 0
        freq_cof = diff_rpeak/350 * loader.FREQUENCY
    else:
        freq_cof = loader.FREQUENCY

    a = dict()
    r_list = []
    p_list = []
    q_list = []
    s_list = []
    t_list = []

    for i in range(np.shape(thb)[0]):
        r_pos = int(0.2 * loader.FREQUENCY)
        template = thb[i,:]
        PQ = template[r_pos - int(0.2 * freq_cof):r_pos - int(0.05 * freq_cof)]
        ST = template[r_pos + int(0.05 * freq_cof):r_pos + int(0.4 * freq_cof)]
        QR = template[r_pos - int(0.07 * freq_cof):r_pos]
        RS = template[r_pos:r_pos + int(0.07 * freq_cof)]

        q_list.append(np.min(QR))
        s_list.append(np.min(RS))

        p_list.append(np.max(PQ))
        t_list.append(np.max(ST))
        r_list.append(template[r_pos])

    a['T_P_max'] = np.max(p_list)
    a['T_P_min'] = np.min(p_list)
    a['T_P_std'] = np.std(p_list)
    a['T_Q_max'] = np.max(q_list)
    a['T_Q_min'] = np.min(q_list)
    a['T_Q_std'] = np.std(q_list)
    a['T_R_max'] = np.max(r_list)
    a['T_R_min'] = np.min(r_list)
    a['T_R_std'] = np.std(r_list)
    a['T_S_max'] = np.max(s_list)
    a['T_S_min'] = np.min(s_list)
    a['T_S_std'] = np.std(s_list)
    a['T_T_max'] = np.max(t_list)
    a['T_T_min'] = np.min(t_list)
    a['T_T_std'] = np.std(t_list)



    # PQ = means[:int(0.15 * loader.FREQUENCY)]
    PQ = means[r_pos -int(0.2 * freq_cof):r_pos-int(0.05 * freq_cof)]
    #ST = means[int(0.25 * loader.FREQUENCY):]
    #ST = means[r_pos + int(0.05 * freq_cof):r_pos + int(0.4 * freq_cof)]
    #QR = means[int(0.13 * loader.FREQUENCY):r_pos]
    QR = means[r_pos-int(0.07 * freq_cof):r_pos]
    RS = means[r_pos:r_pos+int(0.07 * freq_cof)]

    #q_pos = int(0.13 * loader.FREQUENCY) + np.argmin(QR)
    q_pos = r_pos - len(QR) + np.argmin(QR)
    s_pos = r_pos + np.argmin(RS)

    ST = means[s_pos + int(0.08 * freq_cof):r_pos + int(0.35 * freq_cof)]
    p_pos = np.argmax(PQ)

    t_pos = np.argmax(ST)
    if t_pos / len(ST) < 0.2 or t_pos / len(ST) > 0.8:
        t_pos = np.argmin(ST)
        if t_pos / len(ST) < 0.2 or t_pos / len(ST) > 0.8:
            t_pos = int(0.12 * freq_cof)
    t_pos = t_pos + s_pos + int(0.08 * freq_cof)
    t_begin = max(s_pos + int(0.08 * freq_cof), t_pos - int(0.12 * freq_cof))
    st_begin = max(s_pos + int(0.035 * freq_cof) ,t_begin - int(0.06 * freq_cof))
    #t_begin = int(0.035 * freq_cof) + s_pos
    #t_end = int(0.24 * freq_cof) + t_begin
    t_end = min(len(means)-1, int(t_pos + int(0.12 * freq_cof)))

    #t_wave = ST[max(0, t_pos - int(0.08 * freq_cof)):min(len(ST), t_pos + int(0.08 * freq_cof))]
    t_wave = means[t_begin:t_end]
    p_wave = PQ[max(0, p_pos - int(0.06 * freq_cof)):min(len(PQ), p_pos + int(0.06 * freq_cof))]
    st_wave = means[st_begin:t_begin]
    p_pos = p_pos+r_pos -int(0.2 * freq_cof)

    QRS = means[q_pos:s_pos]
    #
    a['PR_interval'] = (r_pos - p_pos)/500
    a['PQ_interval'] = (p_pos - q_pos)/500
    a['RT_interval'] = (t_pos - r_pos) / 500
    a['PT_interval'] = (t_pos - p_pos) / 500
    a['ST_interval'] = (t_pos - s_pos) / 500
    a['T_slope1'] = (means[t_pos]-means[t_begin])/len(t_wave)
    a['T_slope2'] = (means[t_pos]-means[t_end])/len(t_wave)
    a['ST_slope'] = (means[t_begin] - means[st_begin]) / len(st_wave)
    a['RS_interval'] = (r_pos - q_pos) / 500
    a['QR_interval'] = (q_pos - r_pos) / 500
    a['R_time'] = len(p_wave)/500
    a['T_time'] = len(t_wave)/500
    a['t_activity'] = calcActivity(t_wave)
    a['t_mobility'] = calcMobility(t_wave)
    a['t_complexity'] = calcComplexity(t_wave)
    a['p_activity'] = calcActivity(p_wave)
    a['p_mobility'] = calcMobility(p_wave)
    a['p_complexity'] = calcComplexity(p_wave)
    #

    a['P_max'] = means[p_pos]
    a['P_min'] = min(p_wave)
    a['P_to_R'] = means[p_pos] / means[r_pos]
    a['P_to_Q'] = means[p_pos] - means[q_pos]

    a['T_am'] = means[t_pos]
    a['T_max'] = max(t_wave)
    a['T_min'] = min(t_wave)
    a['T_to_R'] = means[t_pos] / means[r_pos]
    a['T_to_S'] = means[t_pos] / means[s_pos]
    a['P_to_T'] = means[p_pos] / means[t_pos]
    a['P_skew'] = skew(p_wave)
    a['P_kurt'] = kurtosis(p_wave)
    a['T_skew'] = skew(t_wave)
    a['T_kurt'] = kurtosis(t_wave)
    a['activity'] = calcActivity(means)
    a['mobility'] = calcMobility(means)
    a['complexity'] = calcComplexity(means)

    qrs_min = abs(min(QRS))
    qrs_max = abs(max(QRS))
    qrs_abs = max(qrs_min, qrs_max)
    sign = -1 if qrs_min > qrs_max else 1

    a['QRS_diff'] = sign * abs(qrs_min / qrs_abs)
    a['QS_diff'] = abs(means[s_pos] - means[q_pos])
    a['QRS_kurt'] = kurtosis(QRS)
    a['QRS_skew'] = skew(QRS)
    a['QRS_minmax'] = qrs_max - qrs_min


    a['T_wave_time'] = len(t_wave)/500
    a['P_wave_time'] = len(p_wave)/500

    a['Temp_Median_p_5'] = np.percentile(means, 5)
    a['Temp_Median_p_25'] = np.percentile(means, 25)
    a['Temp_Median_p_75'] = np.percentile(means, 75)
    a['Temp_Median_p_95'] = np.percentile(means, 95)

    a['T_Median_p_5'] = np.percentile(t_wave, 5)
    a['T_Median_p_25'] = np.percentile(t_wave, 25)
    a['T_Median_p_75'] = np.percentile(t_wave, 75)
    a['T_Median_p_95'] = np.percentile(t_wave, 95)

    a.update(fft_features(means))
    return a


def heart_beats_features3(thb):
    means = np.array([np.mean(col) for col in thb.T])
    medians = np.array([np.median(col) for col in thb.T])

    diff = np.subtract(means, medians)
    diff = np.power(diff, 2)

    return {
        'mean_median_diff_mean': np.mean(diff),
        'mean_median_diff_std': np.std(diff)
    }


def cross_beats(s, peaks):
    fs = loader.FREQUENCY
    r_after = int(0.06 * fs)
    r_before = int(0.06 * fs)

    crossbeats = []
    for i in range(1, len(peaks)):
        start = peaks[i - 1] + r_after
        end = peaks[i] - r_before
        if start >= end:
            continue

        crossbeats.append(s[start:end])

    features = dict()
    f_peaks = [sign_changes(x) for x in crossbeats]
    features['cb_p_mean'] = np.mean(f_peaks)
    features['cb_p_min'] = np.min(f_peaks)
    features['cb_p_max'] = np.max(f_peaks)

    return features


def sign_changes(x):
    return len(list(itertools.groupby(x, lambda x: x > 0))) - (x[0] > 0)

import biosppyex
import pandas as pd
from features.statistics import mad
import nolds
import mne
from features.complexity import complexity_entropy_shannon, complexity_entropy_multiscale, complexity_entropy_svd, \
    complexity_entropy_svd, complexity_entropy_spectral, complexity_fisher_info, complexity_fd_petrosian, \
    complexity_fd_higushi

def r_features(r_peaks):
    # Sanity check after artifact removal
    if len(r_peaks) < 5:
        print("NeuroKit Warning: ecg_hrv(): Not enough normal R peaks to compute HRV :/")
        r_peaks = [1000,2000,3000,4000]

    hrv_dict = dict()
    RRis = np.diff(r_peaks)

    RRis = RRis / 500
    RRis = RRis.astype(float)

    # Artifact detection - Statistical
    rr1 = 0
    rr2 = 0
    rr3 = 0
    rr4 = 0
    median_rr = np.median(RRis)
    for index, rr in enumerate(RRis):
        # Remove RR intervals that differ more than 25% from the previous one

        if rr < 0.6:
            rr1 += 1

        if rr > 1.3:
            rr2 += 1

        if rr < median_rr*0.75:
            rr3 +=1

        if rr > median_rr*1.25:
            rr4 +=1

    # Artifacts treatment
    hrv_dict["n_Artifacts1"] = rr1 / len(RRis)
    hrv_dict["n_Artifacts2"] = rr2 / len(RRis)
    hrv_dict["n_Artifacts3"] = rr3 / len(RRis)
    hrv_dict["n_Artifacts4"] = rr4 / len(RRis)

    hrv_dict["RMSSD"] = np.sqrt(np.mean(np.diff(RRis) ** 2))
    hrv_dict["meanNN"] = np.mean(RRis)
    hrv_dict["sdNN"] = np.std(RRis, ddof=1)  # make it calculate N-1
    hrv_dict["cvNN"] = hrv_dict["sdNN"] / hrv_dict["meanNN"]
    hrv_dict["CVSD"] = hrv_dict["RMSSD"] / hrv_dict["meanNN"]
    hrv_dict["medianNN"] = np.median(abs(RRis))
    hrv_dict["madNN"] = mad(RRis, constant=1)
    hrv_dict["mcvNN"] = hrv_dict["madNN"] / hrv_dict["medianNN"]
    nn50 = sum(abs(np.diff(RRis)) > 50)
    nn20 = sum(abs(np.diff(RRis)) > 20)
    hrv_dict["pNN50"] = nn50 / len(RRis) * 100
    hrv_dict["pNN20"] = nn20 / len(RRis) * 100


    hrv_dict["Shannon"] = complexity_entropy_shannon(RRis)
    hrv_dict["Sample_Entropy"] = nolds.sampen(RRis, emb_dim=2)

    #mse = complexity_entropy_multiscale(RRis, max_scale_factor=20, m=2)
    #hrv_dict["Entropy_Multiscale_AUC"] = mse["MSE_AUC"]
    hrv_dict["Entropy_SVD"] = complexity_entropy_svd(RRis, emb_dim=2)
    hrv_dict["Entropy_Spectral_VLF"] = complexity_entropy_spectral(RRis, 500, bands=np.arange(0.0033, 0.04, 0.001))
    hrv_dict["Entropy_Spectral_LF"] = complexity_entropy_spectral(RRis, 500, bands=np.arange(0.04, 0.15, 0.001))
    hrv_dict["Entropy_Spectral_HF"] = complexity_entropy_spectral(RRis, 500, bands=np.arange(0.15, 0.40, 0.001))
    hrv_dict["Fisher_Info"] = complexity_fisher_info(RRis, tau=1, emb_dim=2)
    #hrv_dict["FD_Petrosian"] = complexity_fd_petrosian(RRis)
    #hrv_dict["FD_Higushi"] = complexity_fd_higushi(RRis, k_max=16)


    hrv_dict.update(hrv.time_domain(RRis))
    hrv_dict.update(hrv.frequency_domain(RRis))

    # RRI Velocity
    diff_rri = np.diff(RRis)
    hrv_dict.update(add_suffix(hrv.time_domain(diff_rri), "fil1"))
    hrv_dict.update(add_suffix(hrv.frequency_domain(diff_rri), "fil1"))
    # RRI Acceleration
    diff2_rri = np.diff(diff_rri)
    hrv_dict.update(add_suffix(hrv.time_domain(diff2_rri), "fil2"))
    hrv_dict.update(add_suffix(hrv.frequency_domain(diff2_rri), "fil2"))
    return hrv_dict


def add_suffix(dic, suffix):
    keys = list(dic.keys())
    for key in keys:
        dic[key + suffix] = dic.pop(key)
    return dic


def SampEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(B)

    N = len(U)

    return -np.log(_phi(m + 1) / _phi(m))

def get_features_dict_ex(signals, filename):
    """
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
    :param x:
    :return:
    """
    out, rpeaks = ecg.ecgex(signals=signals, sampling_rate=loader.FREQUENCY, show=False, filename=filename)

    result = []
    templates = []
    for i in range(len(out)):
        x = signals.iloc[:,[i]].values.reshape(-1)
        [ts, fts, new_rpeaks, tts, thb, hrts, hr] = out[i]
        fx = dict()
        fx.update(heart_rate_features(hr))
        fx.update(frequency_powers(x, n_power_features=60))
        fx.update(add_suffix(frequency_powers(fts), "fil"))
        fx.update(frequency_powers_summary(fts))
        fx.update(heart_beats_features2(thb, rpeaks))
        #fx.update(fft_features(heartbeats.median_heartbeat(thb)))
        templates.append(heartbeats.median_heartbeat(thb))

        fx['PRbyST'] = fx['PR_interval'] * fx['ST_interval']
        fx['P_form'] = fx['P_kurt'] * fx['P_skew']
        fx['T_form'] = fx['T_kurt'] * fx['T_skew']


        for key, value in fx.items():
            if np.math.isnan(value):
                value = 0
            fx[key] = value
        from time_domain_feature import psfeatureTime
        result += psfeatureTime(x)
        result += list(np.array([fx[key] for key in sorted(list(fx.keys()))], dtype=np.float32))
    rx = r_features(rpeaks)
    result += list(np.array([rx[key] for key in sorted(list(rx.keys()))], dtype=np.float32))
    import os
    #path = os.path.join('data', filename)
    #pd.DataFrame(np.array(templates).T).to_csv(path,index=None,header=None)
    return result

def get_features_dict(x):
    """
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
    :param x:
    :return:
    """
    [ts, fts, rpeaks, tts, thb, hrts, hr] = ecg.ecg(signal=x, sampling_rate=loader.FREQUENCY, show=False)

    """
    Returns:	

    ts (array) – Signal time axis reference (seconds).
    filtered (array) – Filtered ECG signal.
    rpeaks (array) – R-peak location indices.
    templates_ts (array) – Templates time axis reference (seconds).
    templates (array) – Extracted heartbeat templates.
    heart_rate_ts (array) – Heart rate time axis reference (seconds).
    heart_rate (array) – Instantaneous heart rate (bpm).
    """

    fx = dict()
    fx.update(heart_rate_features(hr))
    fx.update(frequency_powers(x, n_power_features=60))
    fx.update(add_suffix(frequency_powers(fts), "fil"))
    fx.update(frequency_powers_summary(fts))
    fx.update(heart_beats_features2(thb))
    fx.update(fft_features(heartbeats.median_heartbeat(thb)))
    # fx.update(heart_beats_features3(thb))
    fx.update(r_features(fts, rpeaks))

    fx['PRbyST'] = fx['PR_interval'] * fx['ST_interval']
    fx['P_form'] = fx['P_kurt'] * fx['P_skew']
    fx['T_form'] = fx['T_kurt'] * fx['T_skew']

    """
    from features.template_statistics import TemplateStatistics
    # Get features
    template_statistics = TemplateStatistics(
        ts=ts,
        signal_raw=x,
        signal_filtered=fts,
        rpeaks=rpeaks,
        templates_ts=tts,
        templates=np.array(thb).T,
        fs=loader.FREQUENCY,
        template_before=0.25,
        template_after=0.4
    )
    template_statistics.calculate_template_statistics()

    # Update feature dictionary
    fx.update(template_statistics.get_template_statistics())
    """
    for key, value in fx.items():
        if np.math.isnan(value):
            value = 0
        fx[key] = value

    return fx


def get_feature_names(x):
    features = get_features_dict(x)
    return sorted(list(features.keys()))


def features_for_row(x):
    features = get_features_dict(x)
    return np.array([features[key] for key in sorted(list(features.keys()))], dtype=np.float32)

def features_for_row_ex(x, file_name):
    features = get_features_dict_ex(x, file_name)
    return features