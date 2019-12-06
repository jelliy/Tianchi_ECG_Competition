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

def three_points_median(ts):
    new_ts = []
    if len(ts) < 3:
        return new_ts
    for i in range(len(ts)-2):
        new_ts.append(np.median([ts[i], ts[i+1], ts[i+2]]))
    return new_ts

def autocorr(ts, t):
    return np.corrcoef(np.array([ts[0:len(ts)-t], ts[t:len(ts)]]))[0,1]

def SampleEn(ts):
    '''
    sample entropy on QRS interval
    '''


def CoeffOfVariation(ts):
    '''
    analysis of cumulative distribution functions [17],
    '''
    if len(ts) >= 3:
        tmp_ts = ts[1:-1]
        if np.mean(tmp_ts) == 0:
            coeff_ts = 0.0
        else:
            coeff_ts = np.std(tmp_ts) / np.mean(tmp_ts)
    else:
        coeff_ts = 0.0

    if len(ts) >= 4:
        tmp_ts = ts[1:-1]
        tmp_ts = np.diff(tmp_ts)
        if np.mean(tmp_ts) == 0:
            coeff_dts = 0.0
        else:
            coeff_dts = np.std(tmp_ts) / np.mean(tmp_ts)
    else:
        coeff_dts = 0.0

    return [coeff_ts, coeff_dts]

def heart_beats_features(thb, diff_rpeak):
    means = heartbeats.median_heartbeat(thb)
    r_pos = int(0.2 * loader.FREQUENCY)

    if diff_rpeak < 280:
        tmp1 = 100-int(diff_rpeak/3)
        tmp2 = 100 + int(2 * diff_rpeak / 3)
        for i in range(len(means)):
            if i < tmp1:
                means[i] = 0
            elif i > tmp2:
                means[i] = 0
        freq_cof = diff_rpeak/300 * loader.FREQUENCY
    else:
        freq_cof = loader.FREQUENCY

    # PQ = means[:int(0.15 * loader.FREQUENCY)]
    PQ = means[r_pos -int(0.2 * freq_cof):r_pos-int(0.05 * freq_cof)]
    #ST = means[int(0.25 * loader.FREQUENCY):]
    ST = means[r_pos + int(0.05 * freq_cof):r_pos + int(0.4 * freq_cof)]
    #QR = means[int(0.13 * loader.FREQUENCY):r_pos]
    QR = means[r_pos-int(0.07 * freq_cof):r_pos]
    RS = means[r_pos:r_pos+int(0.07 * freq_cof)]

    #q_pos = int(0.13 * loader.FREQUENCY) + np.argmin(QR)
    q_pos = r_pos - len(QR) + np.argmin(QR)
    s_pos = r_pos + np.argmin(RS)

    p_pos = np.argmax(PQ)
    t_pos = np.argmax(ST)

    t_wave = ST[max(0, t_pos - int(0.08 * freq_cof)):min(len(ST), t_pos + int(0.08 * freq_cof))]
    p_wave = PQ[max(0, p_pos - int(0.06 * freq_cof)):min(len(PQ), p_pos + int(0.06 * freq_cof))]

    #r_plus = sum(1 if b[r_pos] > 0 else 0 for b in thb)
    #r_minus = len(thb) - r_plus

    QRS = means[q_pos:s_pos]


    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(means)
    plt.axvline(p_pos+r_pos -int(0.2 * freq_cof), color='red')
    plt.axvline(q_pos, color='green')
    plt.axvline(r_pos, color='blue')
    plt.axvline(s_pos, color='black')
    plt.axvline(t_pos+r_pos + int(0.05 * freq_cof), color='yellow')
    plt.show()

    a = dict()
    a['PR_interval'] = r_pos - p_pos
    a['P_max'] = PQ[p_pos]
    a['P_to_R'] = PQ[p_pos] / means[r_pos]
    a['P_to_Q'] = PQ[p_pos] - means[q_pos]
    a['ST_interval'] = t_pos
    a['T_max'] = ST[t_pos]
    #a['R_plus'] = r_plus / max(1, len(thb))
    #a['R_minus'] = r_minus / max(1, len(thb))
    a['T_to_R'] = ST[t_pos] / means[r_pos]
    a['T_to_S'] = ST[t_pos] - means[s_pos]
    a['P_to_T'] = PQ[p_pos] / ST[t_pos]
    a['P_skew'] = skew(p_wave)
    a['P_kurt'] = kurtosis(p_wave)
    a['T_skew'] = skew(t_wave)
    a['T_kurt'] = kurtosis(t_wave)
    a['activity'] = calcActivity(means)
    a['mobility'] = calcMobility(means)
    a['complexity'] = calcComplexity(means)
    a['QRS_len'] = s_pos - q_pos

    qrs_min = abs(min(QRS))
    qrs_max = abs(max(QRS))
    qrs_abs = max(qrs_min, qrs_max)
    sign = -1 if qrs_min > qrs_max else 1

    a['QRS_diff'] = sign * abs(qrs_min / qrs_abs)
    a['QS_diff'] = abs(means[s_pos] - means[q_pos])
    a['QRS_kurt'] = kurtosis(QRS)
    a['QRS_skew'] = skew(QRS)
    a['QRS_minmax'] = qrs_max - qrs_min
    #a['P_std'] = np.mean(stds[:q_pos])
    #a['T_std'] = np.mean(stds[s_pos:])

    a['Temp_Median_p_5'] = np.percentile(means, 5)
    a['Temp_Median_p_25'] = np.percentile(means, 25)
    a['Temp_Median_p_75'] = np.percentile(means, 75)
    a['Temp_Median_p_95'] = np.percentile(means, 95)

    return a

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = np.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

def heart_beats_features3(thb):
    means = heartbeats.median_heartbeat(thb)
    from fastdtw  import fastdtw
    dist_list = []
    std_list = []
    mean_list = []
    skew_list = []
    kurtosis_list = []

    for i in range(len(thb)):
        template = thb[i]
        distance = cos_dist(template, means)
        dist_list.append(distance)
        std_list.append(template.std())
        mean_list.append(template.mean())
        skew_list.append(skew(template))
        kurtosis_list.append(kurtosis(template))

    heart_beats_dict = {}
    if len(thb) > 0:
        outlier_index = dist_list.index(min(dist_list))
        max_template = thb[outlier_index]

        dict1 = heart_beats_features(means)
        heart_beats_dict.update(dict1)
        dict2 = add_suffix(heart_beats_features(max_template), "fil")
        heart_beats_dict.update(dict2)

        dist_array = np.array(dist_list)
        dist_array.mean()
        heart_beats_dict['Templates_Dist_Mean'] = dist_array.mean()
        heart_beats_dict['Templates_Dist_Std'] = dist_array.std()
        std_list = np.array(std_list)
        heart_beats_dict['Templates_Std_Mean'] = std_list.mean()
        heart_beats_dict['Templates_Std_Std'] = std_list.std()
        mean_list = np.array(mean_list)
        heart_beats_dict['Templates_Mean_Mean'] = mean_list.mean()
        heart_beats_dict['Templates_Mean_Std'] = mean_list.std()
        skew_list = np.array(skew_list)
        heart_beats_dict['Templates_Skew_Mean'] = skew_list.mean()
        heart_beats_dict['Templates_Skew_Std'] = skew_list.std()
        kurtosis_list = np.array(kurtosis_list)
        heart_beats_dict['Templates_Kurtosis_Mean'] = kurtosis_list.mean()
        heart_beats_dict['Templates_Kurtosis_Std'] = kurtosis_list.std()
    else:
        dict1 = heart_beats_features(means)
        dict2 = add_suffix(heart_beats_features(means), "fil")
        heart_beats_dict.update(dict1)
        heart_beats_dict.update(dict2)

        heart_beats_dict['Templates_Dist_Mean'] = 0
        heart_beats_dict['Templates_Dist_Std'] = 0

        heart_beats_dict['Templates_Std_Mean'] = 0
        heart_beats_dict['Templates_Std_Std'] = 0

        heart_beats_dict['Templates_Mean_Mean'] = 0
        heart_beats_dict['Templates_Mean_Std'] = 0

        heart_beats_dict['Templates_Skew_Mean'] = 0
        heart_beats_dict['Templates_Skew_Std'] = 0

        heart_beats_dict['Templates_Kurtosis_Mean'] = 0
        heart_beats_dict['Templates_Kurtosis_Std'] = 0

    return heart_beats_dict

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


def r_features(s, r_peaks):
    r_vals = [s[i] for i in r_peaks]

    times = np.diff(r_peaks)
    avg = np.mean(times)
    filtered = sum([1 if i < 0.5 * avg else 0 for i in times])

    total = len(r_vals) if len(r_vals) > 0 else 1

    data = hrv.time_domain(times)

    data['beats_to_length'] = len(r_peaks) / len(s)
    data['r_mean'] = np.mean(r_vals)
    data['r_std'] = np.std(r_vals)
    data['filtered_r'] = filtered
    data['rel_filtered_r'] = filtered / total

    return data


def add_suffix(dic, suffix):
    keys = list(dic.keys())
    for key in keys:
        dic[key + suffix] = dic.pop(key)
    return dic

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
    for i in range(len(out)):
        x = signals.iloc[:,[i]].values.reshape(-1)
        [ts, fts, new_rpeaks, tts, thb, hrts, hr] = out[i]
        import neurokit as nk
        rsa = nk.respiratory_sinus_arrhythmia(rpeaks, rsp_cycles, rsp_signal)
        fx = dict()
        fx.update(add_suffix(frequency_powers(fts), "fil"))
        fx.update(frequency_powers_summary(fts))
        fx.update(heart_beats_features(thb,np.median(np.diff(new_rpeaks))))
        fx.update(fft_features(heartbeats.median_heartbeat(thb)))
        # fx.update(heart_beats_features3(thb))
        # gloal feature
        fx.update(frequency_powers(x, n_power_features=60))
        fx.update(heart_rate_features(hr))
        fx.update(r_features(fts,new_rpeaks))

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
    # delete Outliers
    fx.update(add_suffix(heart_rate_features(three_points_median(hr)), "_3"))
    fx.update(frequency_powers(x, n_power_features=50))
    fx.update(add_suffix(frequency_powers(fts), "fil"))
    fx.update(frequency_powers_summary(fts))
    fx.update(heart_beats_features(thb, np.median(np.diff(rpeaks))))
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