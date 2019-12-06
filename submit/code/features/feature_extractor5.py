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
    means = heartbeats.median_heartbeat(thb)
    diff_rpeak = np.median(np.diff(rpeaks))
    freq_cof = loader.FREQUENCY
    if diff_rpeak < 280:
        tmp1 = 100 - int(diff_rpeak / 3)
        tmp2 = 100 + int(2 * diff_rpeak / 3)
        for i in range(len(means)):
            if i < tmp1:
                means[i] = 0
            elif i > tmp2:
                means[i] = 0
        freq_cof = diff_rpeak / 300 * loader.FREQUENCY

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

    stds = np.array([np.std(col) for col in thb.T])

    r_pos = int(0.2 * loader.FREQUENCY)

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


    a['PR_interval'] = r_pos - p_pos
    a['P_max'] = PQ[p_pos]
    a['P_min'] = min(PQ)
    a['P_to_R'] = PQ[p_pos] / means[r_pos]
    a['P_to_Q'] = PQ[p_pos] - means[q_pos]
    a['ST_interval'] = t_pos
    a['T_max'] = ST[t_pos]
    a['T_min'] = min(ST)
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
    a['P_std'] = np.mean(stds[:q_pos])
    a['T_std'] = np.mean(stds[s_pos:])

    a['T_wave_time'] = len(t_wave)/500
    a['P_wave_time'] = len(p_wave)/500

    a['Temp_Median_p_5'] = np.percentile(means, 5)
    a['Temp_Median_p_25'] = np.percentile(means, 25)
    a['Temp_Median_p_75'] = np.percentile(means, 75)
    a['Temp_Median_p_95'] = np.percentile(means, 95)

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


def r_features(r_peaks):
    times = np.diff(r_peaks)
    avg = np.mean(times)
    filtered = sum([1 if i < 0.5 * avg else 0 for i in times])

    total = len(r_peaks)

    data = hrv.time_domain(times)

    data['filtered_r'] = filtered
    data['rel_filtered_r'] = filtered / total

    # RRI Velocity
    diff_rri = np.diff(times)
    data.update(add_suffix(hrv.time_domain(diff_rri), "fil1"))
    # RRI Acceleration
    diff2_rri = np.diff(diff_rri)
    data.update(add_suffix(hrv.time_domain(diff2_rri), "fil2"))

    return data


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