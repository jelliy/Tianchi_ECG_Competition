"""
template_statistics.py
--------------------
This module provides a class and methods for extracting template statistics from ECG templates.
Implemented code assumes a single-channel lead ECG signal.
:copyright: (c) 2017 by Goodfellow Analytics
--------------------
By: Sebastian D. Goodfellow, Ph.D.
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np
import scipy as sp
from pyentrp import entropy as ent
import pyeeg
from scipy import signal
import pywt

# Local imports
from utils.pyrem_univariate import *


class TemplateStatistics:

    """
    Generate a dictionary of template statistics for one ECG signal.

    Parameters
    ----------
    ts : numpy array
        Full waveform time array.
    signal_raw : numpy array
        Raw full waveform.
    signal_filtered : numpy array
        Filtered full waveform.
    rpeaks : numpy array
        Array indices of R-Peaks
    templates_ts : numpy array
        Template waveform time array
    templates : numpy array
        Template waveforms
    fs : int, float
        Sampling frequency (Hz).
    template_before : float, seconds
            Time before R-Peak to start template.
    template_after : float, seconds
        Time after R-Peak to end template.

    Returns
    -------
    template_statistics : dictionary
        Template statistics.
    """

    def __init__(self, ts, signal_raw, signal_filtered, rpeaks,
                 templates_ts, templates, fs, template_before,
                 template_after):

        # Input variables
        self.ts = ts
        self.signal_raw = signal_raw
        self.signal_filtered = signal_filtered
        self.rpeaks = rpeaks
        self.templates_ts = templates_ts
        self.templates = templates
        self.fs = fs
        self.template_before_ts = template_before
        self.template_after_ts = template_after
        self.template_before_sp = int(self.template_before_ts * self.fs)
        self.template_after_sp = int(self.template_after_ts * self.fs)

        # Set future variables
        self.templates_filtered = None
        self.templates_filtered_index = None
        self.qrs_start_sp = None
        self.qrs_start_ts = None
        self.qrs_end_sp = None
        self.qrs_end_ts = None
        self.q_times_sp = None
        self.q_amps = None
        self.q_time_sp = None
        self.q_amp = None
        self.p_times_sp = None
        self.p_amps = None
        self.p_time_sp = None
        self.p_amp = None
        self.s_times_sp = None
        self.s_amps = None
        self.s_time_sp = None
        self.s_amp = None
        self.t_times_sp = None
        self.t_amps = None
        self.t_time_sp = None
        self.t_amp = None
        self.templates_good = None
        self.templates_bad = None
        self.median_template = None
        self.median_template_good = None
        self.median_template_bad = None
        self.rpeaks_good = None
        self.rpeaks_bad = None

        # Set QRS start and end points
        self.qrs_start_sp_manual = 30  # R - qrs_start_sp_manual
        self.qrs_end_sp_manual = 40  # R + qrs_start_sp_manual

        # R-Peak calculations
        self.template_rpeak_sp = self.template_before_sp

        # Calculate median template
        self.median_template = np.median(self.templates, axis=1)

        # Correct R-Peak picks
        self.r_peak_check(correlation_threshold=0.9)

        # RR interval calculations
        self.rpeaks_ts = self.ts[self.rpeaks]
        self.filter_rpeaks(correlation_threshold=0.9)

        # QRS calculations
        self.calculate_qrs_bounds()

        # PQRST Calculations
        self.preprocess_pqrst()

        # Feature dictionary
        self.template_statistics = dict()

    """
    Compile Features
    """
    def get_template_statistics(self):
        return self.template_statistics

    def calculate_template_statistics(self):
        self.template_statistics.update(self.calculate_p_wave_statistics())
        self.template_statistics.update(self.calculate_q_wave_statistics())
        self.template_statistics.update(self.calculate_t_wave_statistics())
        self.template_statistics.update(self.calculate_s_wave_statistics())
        self.template_statistics.update(self.calculate_pqrst_wave_statistics())
        self.template_statistics.update((self.calculate_r_peak_polarity_statistics()))
        self.template_statistics.update(self.calculate_r_peak_amplitude_statistics())
        self.template_statistics.update(self.calculate_template_correlation_statistics())
        self.template_statistics.update(self.calculate_qrs_correlation_statistics())
        #self.template_statistics.update(self.calculate_p_wave_correlation_statistics())
        #self.template_statistics.update(self.calculate_t_wave_correlation_statistics())
        #self.template_statistics.update(self.calculate_bad_template_correlation_statistics())
        #self.template_statistics.update(self.calculate_stationary_wavelet_transform_statistics())

    """
    Pre Processing
    """
    def r_peak_check(self, correlation_threshold=0.9):

        # Check lengths
        #print(self.templates.shape[1])
        assert len(self.rpeaks) == self.templates.shape[1]

        # Loop through rpeaks
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                # Compute cross correlation
                cross_correlation = signal.correlate(
                    self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                    self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
                )

                # Correct rpeak
                rpeak_corrected = \
                    self.rpeaks[template_id] - \
                    (np.argmax(cross_correlation) -
                     len(self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]))

                # Check to see if shifting the R-Peak improved the correlation coefficient
                if self.check_improvement(rpeak_corrected, correlation_threshold):

                    # Update rpeaks array
                    self.rpeaks[template_id] = rpeak_corrected

        # Re-extract templates
        self.templates, self.rpeaks = self.extract_templates(self.rpeaks)

        # Re-compute median template
        self.median_template = np.median(self.templates, axis=1)

        # Check lengths
        assert len(self.rpeaks) == self.templates.shape[1]

    def extract_templates(self, rpeaks):

        # Sort R-Peaks in ascending order
        rpeaks = np.sort(rpeaks)

        # Get number of sample points in waveform
        length = len(self.signal_filtered)

        # Create empty list for templates
        templates = []

        # Create empty list for new rpeaks that match templates dimension
        rpeaks_new = np.empty(0, dtype=int)

        # Loop through R-Peaks
        for rpeak in rpeaks:

            # Before R-Peak
            a = rpeak - self.template_before_sp
            if a < 0:
                continue

            # After R-Peak
            b = rpeak + self.template_after_sp
            if b > length:
                break

            # Append template list
            templates.append(self.signal_filtered[a:b])

            # Append new rpeaks list
            rpeaks_new = np.append(rpeaks_new, rpeak)

        # Convert list to numpy array
        templates = np.array(templates).T

        return templates, rpeaks_new

    def check_improvement(self, rpeak_corrected, correlation_threshold):

        # Before R-Peak
        a = rpeak_corrected - self.template_before_sp

        # After R-Peak
        b = rpeak_corrected + self.template_after_sp

        if a >= 0 and b < len(self.signal_filtered):

            # Update template
            template_corrected = self.signal_filtered[a:b]

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                template_corrected[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25]
            )

            # Check new correlation
            if correlation_coefficient[0, 1] >= correlation_threshold:
                return True
            else:
                return False
        else:
            return False

    def filter_rpeaks(self, correlation_threshold=0.9):

        # Get rpeaks is floats
        rpeaks = self.rpeaks.astype(float)

        # Loop through templates
        for template_id in range(self.templates.shape[1]):

            # Calculate correlation coefficient
            correlation_coefficient = np.corrcoef(
                self.median_template[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25],
                self.templates[self.template_rpeak_sp - 25:self.template_rpeak_sp + 25, template_id]
            )

            # Check correlation
            if correlation_coefficient[0, 1] < correlation_threshold:

                rpeaks[template_id] = np.nan

        # Get good and bad rpeaks
        self.rpeaks_good = rpeaks[np.isfinite(rpeaks)]
        self.rpeaks_bad = rpeaks[~np.isfinite(rpeaks)]

        # Get good and bad
        self.templates_good = self.templates[:, np.where(np.isfinite(rpeaks))[0]]
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.templates_bad = self.templates[:, np.where(~np.isfinite(rpeaks))[0]]

        # Get median templates
        self.median_template_good = np.median(self.templates_good, axis=1)
        if len(np.where(~np.isfinite(rpeaks))[0]) > 0:
            self.median_template_bad = np.median(self.templates_bad, axis=1)

    def calculate_qrs_bounds(self):

        # Empty lists of QRS start and end times
        qrs_starts_sp = []
        qrs_ends_sp = []

        # Loop through templates
        for template in range(self.templates_good.shape[1]):

            # Get zero crossings before the R-Peak
            pre_qrs_zero_crossings = np.where(
                np.diff(np.sign(self.templates_good[0:self.template_rpeak_sp, template]))
            )[0]

            # Check length
            if len(pre_qrs_zero_crossings) >= 2:

                # Append QRS starting index
                qrs_starts_sp = np.append(qrs_starts_sp, pre_qrs_zero_crossings[-2])

            if len(qrs_starts_sp) > 0:

                self.qrs_start_sp = int(np.median(qrs_starts_sp))
                self.qrs_start_ts = self.qrs_start_sp / self.fs

            else:
                self.qrs_start_sp = int(self.template_before_sp / 2.0)

            # Get zero crossings after the R-Peak
            post_qrs_zero_crossings = np.where(
                np.diff(np.sign(self.templates_good[self.template_rpeak_sp:-1, template]))
            )[0]

            # Check length
            if len(post_qrs_zero_crossings) >= 2:

                # Append QRS ending index
                qrs_ends_sp = np.append(qrs_ends_sp, post_qrs_zero_crossings[-2])

            if len(qrs_ends_sp) > 0:

                self.qrs_end_sp = int(self.template_before_sp + np.median(qrs_ends_sp))
                self.qrs_end_ts = self.qrs_end_sp / self.fs

            else:
                self.qrs_end_sp = int(self.template_before_sp + self.template_after_sp / 2.0)

    def preprocess_pqrst(self):

        # Get QRS start point
        qrs_start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Get QRS end point
        qrs_end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Get QR median template
        qr_median_template = self.median_template_good[qrs_start_sp:self.template_rpeak_sp]

        # Get RS median template
        rs_median_template = self.median_template_good[self.template_rpeak_sp:qrs_end_sp]

        # Get QR templates
        qr_templates = self.templates_good[qrs_start_sp:self.template_rpeak_sp, :]

        # Get RS templates
        rs_templates = self.templates_good[self.template_rpeak_sp:qrs_end_sp, :]

        """
        Q-Wave
        """
        # Get array of Q-wave times (sp)
        self.q_times_sp = np.array(
            [qrs_start_sp + np.argmin(qr_templates[:, col]) for col in range(qr_templates.shape[1])]
        )

        # Get array of Q-wave amplitudes
        self.q_amps = np.array(
            [self.templates_good[self.q_times_sp[col], col] for col in range(self.templates_good.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.q_time_sp = qrs_start_sp + np.argmin(qr_median_template)

        # Get array of Q-wave amplitudes
        self.q_amp = self.median_template_good[self.q_time_sp]

        """
        P-Wave
        """
        # Get array of Q-wave times (sp)
        self.p_times_sp = np.array([
            np.argmax(self.templates_good[0:self.q_times_sp[col], col])
            for col in range(self.templates_good.shape[1])
        ])

        # Get array of Q-wave amplitudes
        self.p_amps = np.array(
            [self.templates_good[self.p_times_sp[col], col] for col in range(self.templates_good.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.p_time_sp = np.argmax(self.median_template_good[0:self.q_time_sp])

        # Get array of Q-wave amplitudes
        self.p_amp = self.median_template_good[self.p_time_sp]

        """
        S-Wave
        """
        # Get array of Q-wave times (sp)
        self.s_times_sp = np.array([
            self.template_rpeak_sp + np.argmin(rs_templates[:, col])
            for col in range(rs_templates.shape[1])
        ])

        # Get array of Q-wave amplitudes
        self.s_amps = np.array(
            [self.templates_good[self.s_times_sp[col], col] for col in range(self.templates_good.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.s_time_sp = self.template_rpeak_sp + np.argmin(rs_median_template)

        # Get array of Q-wave amplitudes
        self.s_amp = self.median_template_good[self.s_time_sp]

        """
        T-Wave
        """
        # Get array of Q-wave times (sp)
        self.t_times_sp = np.array([
            self.s_times_sp[col] + np.argmax(self.templates[self.s_times_sp[col]:, col])
            for col in range(self.templates_good.shape[1])
        ])

        # Get array of Q-wave amplitudes
        self.t_amps = np.array(
            [self.templates_good[self.t_times_sp[col], col] for col in range(self.templates_good.shape[1])]
        )

        # Get array of Q-wave times (sp)
        self.t_time_sp = self.s_time_sp + np.argmax(self.median_template_good[self.s_time_sp:])

        # Get array of Q-wave amplitudes
        self.t_amp = self.median_template_good[self.t_time_sp]

        """
        Debug
        """
        # plt.plot([qrs_start_sp, qrs_start_sp], [self.median_template_good.min(), self.median_template_good.max()], '-g')
        # plt.plot([qrs_end_sp, qrs_end_sp], [self.median_template_good.min(), self.median_template_good.max()], '-g')
        #
        # fig = plt.figure(figsize=(10, 5))
        # templates_ts = np.linspace(-250, 400, self.templates_good.shape[0], endpoint=False)
        #
        # plt.plot(templates_ts, self.templates_good, '-', c=[0.7, 0.7, 0.7], alpha=0.75)
        # plt.plot(templates_ts, self.median_template_good, '-k', lw=2)
        #
        # plt.plot(self.template_rpeak_sp*1/self.fs*1000-250, 1, '.b', ms=12)
        #
        # plt.plot(self.q_times_sp*1/self.fs*1000-250, self.q_amps, '.r')
        # plt.plot(self.q_time_sp*1/self.fs*1000-250, self.q_amp, '.b', ms=12)
        #
        # plt.plot(self.p_times_sp*1/self.fs*1000-250, self.p_amps, '.r')
        # plt.plot(self.p_time_sp*1/self.fs*1000-250, self.p_amp, '.b', ms=12)
        #
        # plt.plot(self.s_times_sp*1/self.fs*1000-250, self.s_amps, '.r')
        # plt.plot(self.s_time_sp*1/self.fs*1000-250, self.s_amp, '.b', ms=12)
        #
        # plt.plot(self.t_times_sp*1/self.fs*1000-250, self.t_amps, '.r')
        # plt.plot(self.t_time_sp*1/self.fs*1000-250, self.t_amp, '.b', ms=12)

        # plt.plot([self.p_time_sp-10, self.p_time_sp+10], [0, 0], '-g')
        # plt.plot([self.t_time_sp-10, self.t_time_sp+10], [0, 0], '-g')

        # plt.xlim([-250, 400])
        # plt.ylim([-0.75, 1.5])
        # plt.xlabel('Time, ms', fontsize=16)
        # plt.ylabel('Amplitude, mV', fontsize=16)
        #
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)

        # plt.show()
        #
        # if self.templates_good.shape[1] > 1:
        #
        #     # Get start point
        #     start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual
        #
        #     # Calculate correlation matrix
        #     correlation_matrix = np.corrcoef(np.transpose(self.templates_good[0:start_sp, :]))
        #
        #     upper_triangle = np.triu(correlation_matrix, k=1)
        #
        #     # Setup figure
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #
        #     # Plot correlations
        #     ms = ax.matshow(np.abs(correlation_matrix), cmap='plasma')
        #
        #     # Rotate x-axis labels
        #     locs, labels = plt.xticks()
        #     plt.setp(labels, rotation=90)
        #
        #     # Format colorbar
        #     divider = make_axes_locatable(ax)
        #     cax = divider.append_axes("right", size="5%", pad=0.2)
        #     cbar = plt.colorbar(ms, cax=cax)
        #     cbar.ax.set_ylabel('Correlation Coefficient', fontsize=20)
        #     cbar.ax.set_ylim([0, 1])
        #
        #     ax.tick_params(labelsize=12)
        #
        #     ax.set_ylabel('Template ID', fontsize=20)
        #
        #     plt.show()

    """
    Feature Methods
    """
    @staticmethod
    def safe_check(value):

        try:
            if np.isfinite(value):
                return value
            else:
                return np.nan()

        except Exception:
            return np.nan

    def calculate_p_wave_statistics(self):

        # Empty dictionary
        p_wave_statistics = dict()

        # Get P-Wave energy bounds
        p_eng_start = self.p_time_sp - 10
        if p_eng_start < 0:
            p_eng_start = 0
        p_eng_end = self.p_time_sp + 10

        # Get end points
        start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

        # Calculate p-wave statistics
        if self.templates_good.shape[1] > 0:
            p_wave_statistics['p_wave_time'] = self.p_time_sp * 1 / self.fs
            p_wave_statistics['p_wave_time_std'] = np.std(self.p_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            p_wave_statistics['p_wave_amp'] = self.p_amp
            p_wave_statistics['p_wave_amp_std'] = np.std(self.p_amps, ddof=1)
            p_wave_statistics['p_wave_eng'] = np.sum(np.power(self.median_template_good[p_eng_start:p_eng_end], 2))
        else:
            p_wave_statistics['p_wave_time'] = np.nan
            p_wave_statistics['p_wave_time_std'] = np.nan
            p_wave_statistics['p_wave_amp'] = np.nan
            p_wave_statistics['p_wave_amp_std'] = np.nan
            p_wave_statistics['p_wave_eng'] = np.nan

        """
        Calculate non-linear statistics
        """
        approximate_entropy = [
            pyeeg.ap_entropy(self.templates_good[0:start_sp, col], M=2, R=0.1*np.std(self.templates_good[0:start_sp, col]))
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_approximate_entropy_median'] = np.median(approximate_entropy)
        p_wave_statistics['p_wave_approximate_entropy_std'] = np.std(approximate_entropy, ddof=1)

        sample_entropy = [
            self.safe_check(
                ent.sample_entropy(
                    self.templates_good[0:start_sp, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates_good[0:start_sp, col])
                )[0]
            )
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_sample_entropy_median'] = np.median(sample_entropy)
        p_wave_statistics['p_wave_sample_entropy_std'] = np.std(sample_entropy, ddof=1)

        multiscale_entropy = [
            self.safe_check(
                ent.multiscale_entropy(
                    self.templates_good[0:start_sp, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates_good[0:start_sp, col])
                )[0]
            )
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_multiscale_entropy_median'] = np.median(multiscale_entropy)
        p_wave_statistics['p_wave_multiscale_entropy_std'] = np.std(multiscale_entropy, ddof=1)

        permutation_entropy = [
            self.safe_check(
                ent.permutation_entropy(
                    self.templates_good[0:start_sp, col], delay=1
                )
            )
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_permutation_entropy_median'] = np.median(permutation_entropy)
        p_wave_statistics['p_wave_permutation_entropy_std'] = np.std(permutation_entropy, ddof=1)

        multiscale_permutation_entropy = [
            self.safe_check(
                ent.multiscale_permutation_entropy(
                    self.templates_good[0:start_sp, col],
                    m=2, delay=1, scale=1
                )[0]
            )
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_multiscale_permutation_entropy_median'] = np.median(multiscale_permutation_entropy)
        p_wave_statistics['p_wave_multiscale_permutation_entropy_std'] = np.std(multiscale_permutation_entropy, ddof=1)

        fisher_information = [
            fisher_info(self.templates_good[0:start_sp, col], tau=1, de=2)
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_fisher_info_median'] = np.median(fisher_information)
        p_wave_statistics['p_wave_fisher_info_std'] = np.std(fisher_information, ddof=1)

        higuchi_fractal = [
            hfd(self.templates_good[0:start_sp, col], k_max=10) for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_higuchi_fractal_median'] = np.median(higuchi_fractal)
        p_wave_statistics['p_wave_higuchi_fractal_std'] = np.std(higuchi_fractal, ddof=1)

        hurst_exponent = [
            pfd(self.templates_good[0:start_sp, col]) for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_hurst_exponent_median'] = np.median(hurst_exponent)
        p_wave_statistics['p_wave_hurst_exponent_std'] = np.std(hurst_exponent, ddof=1)

        svd_entr = [
            svd_entropy(self.templates_good[0:start_sp, col], tau=2, de=2)
            for col in range(self.templates_good.shape[1])
        ]
        p_wave_statistics['p_wave_svd_entropy_median'] = np.median(svd_entr)
        p_wave_statistics['p_wave_svd_entropy_std'] = np.std(svd_entr, ddof=1)

        return p_wave_statistics

    def calculate_q_wave_statistics(self):

        # Empty dictionary
        q_wave_statistics = dict()

        # Calculate p-wave statistics
        if self.templates_good.shape[1] > 0:
            q_wave_statistics['q_wave_time'] = self.q_time_sp * 1 / self.fs
            q_wave_statistics['q_wave_time_std'] = np.std(self.q_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            q_wave_statistics['q_wave_amp'] = self.q_amp
            q_wave_statistics['q_wave_amp_std'] = np.std(self.q_amps, ddof=1)
        else:
            q_wave_statistics['q_wave_time'] = np.nan
            q_wave_statistics['q_wave_time_std'] = np.nan
            q_wave_statistics['q_wave_amp'] = np.nan
            q_wave_statistics['q_wave_amp_std'] = np.nan

        return q_wave_statistics

    def calculate_s_wave_statistics(self):

        # Empty dictionary
        s_wave_statistics = dict()

        # Calculate p-wave statistics
        if self.templates_good.shape[1] > 0:
            s_wave_statistics['s_wave_time'] = self.s_time_sp * 1 / self.fs
            s_wave_statistics['s_wave_time_std'] = np.std(self.s_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            s_wave_statistics['s_wave_amp'] = self.s_amp
            s_wave_statistics['s_wave_amp_std'] = np.std(self.s_amps, ddof=1)
        else:
            s_wave_statistics['s_wave_time'] = np.nan
            s_wave_statistics['s_wave_time_std'] = np.nan
            s_wave_statistics['s_wave_amp'] = np.nan
            s_wave_statistics['s_wave_amp_std'] = np.nan

        return s_wave_statistics

    def calculate_t_wave_statistics(self):

        # Empty dictionary
        t_wave_statistics = dict()

        # Get T-Wave energy bounds
        t_eng_start = self.t_time_sp - 10
        t_eng_end = self.t_time_sp + 10
        if t_eng_end > self.templates_good.shape[0] - 1:
            t_eng_end = self.templates_good.shape[0] - 1

        # Get end points
        end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

        # Calculate p-wave statistics
        if self.templates_good.shape[1] > 0:
            t_wave_statistics['t_wave_time'] = self.t_time_sp * 1 / self.fs
            t_wave_statistics['t_wave_time_std'] = np.std(self.t_times_sp * 1 / self.fs, ddof=1) * 1 / self.fs
            t_wave_statistics['t_wave_amp'] = self.t_amp
            t_wave_statistics['t_wave_amp_std'] = np.std(self.t_amps, ddof=1)
            t_wave_statistics['t_wave_eng'] = np.sum(np.power(self.median_template_good[t_eng_start:t_eng_end], 2))
        else:
            t_wave_statistics['t_wave_time'] = np.nan
            t_wave_statistics['t_wave_time_std'] = np.nan
            t_wave_statistics['t_wave_amp'] = np.nan
            t_wave_statistics['t_wave_amp_std'] = np.nan
            t_wave_statistics['t_wave_eng'] = np.nan

        """
        Calculate non-linear statistics
        """
        approximate_entropy = [
            pyeeg.ap_entropy(self.templates_good[end_sp:, col], M=2, R=0.1*np.std(self.templates_good[end_sp:, col]))
            for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_approximate_entropy_median'] = np.median(approximate_entropy)
        t_wave_statistics['t_wave_approximate_entropy_std'] = np.std(approximate_entropy, ddof=1)

        sample_entropy = [
            self.safe_check(
                ent.sample_entropy(
                    self.templates_good[end_sp:, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates_good[end_sp:, col])
                )[0]
            )
            for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_sample_entropy_median'] = np.median(sample_entropy)
        t_wave_statistics['t_wave_sample_entropy_std'] = np.std(sample_entropy, ddof=1)

        multiscale_entropy = [
            self.safe_check(
                ent.multiscale_entropy(
                    self.templates_good[end_sp:, col],
                    sample_length=2,
                    tolerance=0.1*np.std(self.templates_good[end_sp:, col])
                )[0]
            )
            for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_multiscale_entropy_median'] = np.median(multiscale_entropy)
        t_wave_statistics['t_wave_multiscale_entropy_std'] = np.std(multiscale_entropy, ddof=1)

        permutation_entropy = [
            self.safe_check(
                ent.permutation_entropy(
                    self.templates_good[end_sp:, col],delay=1
                )
            )
            for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_permutation_entropy_median'] = np.median(permutation_entropy)
        t_wave_statistics['t_wave_permutation_entropy_std'] = np.std(permutation_entropy, ddof=1)

        multiscale_permutation_entropy = [
            self.safe_check(
                ent.multiscale_permutation_entropy(
                    self.templates_good[end_sp:, col],
                    m=2, delay=1, scale=1
                )[0]
            )
            for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_multiscale_permutation_entropy_median'] = np.median(multiscale_permutation_entropy)
        t_wave_statistics['t_wave_multiscale_permutation_entropy_std'] = np.std(multiscale_permutation_entropy, ddof=1)

        fisher_information = [
            fisher_info(self.templates_good[end_sp:, col], tau=1, de=2) for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_fisher_info_median'] = np.median(fisher_information)
        t_wave_statistics['t_wave_fisher_info_std'] = np.std(fisher_information, ddof=1)

        higuchi_fractal = [
            hfd(self.templates_good[end_sp:, col], k_max=10) for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_higuchi_fractal_median'] = np.median(higuchi_fractal)
        t_wave_statistics['t_wave_higuchi_fractal_std'] = np.std(higuchi_fractal, ddof=1)

        hurst_exponent = [
            pfd(self.templates_good[end_sp:, col]) for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_hurst_exponent_median'] = np.median(hurst_exponent)
        t_wave_statistics['t_wave_hurst_exponent_std'] = np.std(hurst_exponent, ddof=1)

        svd_entr = [
            svd_entropy(self.templates_good[end_sp:, col], tau=2, de=2) for col in range(self.templates_good.shape[1])
        ]
        t_wave_statistics['t_wave_svd_entropy_median'] = np.median(svd_entr)
        t_wave_statistics['t_wave_svd_entropy_std'] = np.std(svd_entr, ddof=1)

        return t_wave_statistics

    def calculate_pqrst_wave_statistics(self):

        # Empty dictionary
        pqrst_wave_statistics = dict()

        if self.templates_good.shape[1] > 0:

            # PQ time
            pqi = (self.q_times_sp - self.p_times_sp) * 1 / self.fs
            pqrst_wave_statistics['pq_time'] = (self.q_time_sp - self.p_time_sp) * 1 / self.fs
            pqrst_wave_statistics['pq_time_std'] = np.std(pqi, ddof=1)

            # PR time
            pri = (self.template_rpeak_sp - self.p_times_sp) * 1 / self.fs
            pqrst_wave_statistics['pr_time'] = (self.template_rpeak_sp - self.p_time_sp) * 1 / self.fs
            pqrst_wave_statistics['pr_time_std'] = np.std(pri, ddof=1)
            if self.templates_good.shape[1] > 1:
                pqrst_wave_statistics['pri_approximate_entropy'] = \
                    self.safe_check(pyeeg.ap_entropy(pri, M=2, R=0.1*np.std(pri)))
                pqrst_wave_statistics['pri_higuchi_fractal_dimension'] = hfd(pri, k_max=10)
            else:
                pqrst_wave_statistics['pri_approximate_entropy'] = np.nan
                pqrst_wave_statistics['pri_higuchi_fractal_dimension'] = np.nan

            # QR time
            qri = (self.template_rpeak_sp - self.q_times_sp) * 1 / self.fs
            pqrst_wave_statistics['qr_time'] = (self.template_rpeak_sp - self.q_time_sp) * 1 / self.fs
            pqrst_wave_statistics['qr_time_std'] = np.std(qri, ddof=1)

            # RS time
            rsi = (self.s_times_sp - self.template_rpeak_sp) * 1 / self.fs
            pqrst_wave_statistics['rs_time'] = (self.s_time_sp - self.template_rpeak_sp) * 1 / self.fs
            pqrst_wave_statistics['rs_time_std'] = np.std(rsi, ddof=1)

            # QS time
            qsi = (self.s_times_sp - self.q_times_sp) * 1 / self.fs
            pqrst_wave_statistics['qs_time'] = (self.s_time_sp - self.q_time_sp) * 1 / self.fs
            pqrst_wave_statistics['qs_time_std'] = np.std(qsi, ddof=1)
            if self.templates_good.shape[1] > 1:
                pqrst_wave_statistics['qsi_approximate_entropy'] = \
                    self.safe_check(pyeeg.ap_entropy(pri, M=2, R=0.1*np.std(qsi)))
                pqrst_wave_statistics['qsi_higuchi_fractal_dimension'] = hfd(qsi, k_max=10)
                pqrst_wave_statistics['qsi_pearson_coeff'] = np.nan
                pqrst_wave_statistics['qsi_pearson_p_value'] = np.nan
            else:
                pqrst_wave_statistics['qsi_approximate_entropy'] = np.nan
                pqrst_wave_statistics['qsi_higuchi_fractal_dimension'] = np.nan
                pqrst_wave_statistics['qsi_pearson_coeff'] = np.nan
                pqrst_wave_statistics['qsi_pearson_p_value'] = np.nan

            # ST time
            sti = (self.t_times_sp - self.s_times_sp) * 1 / self.fs
            pqrst_wave_statistics['st_time'] = (self.t_time_sp - self.s_time_sp) * 1 / self.fs
            pqrst_wave_statistics['st_time_std'] = np.std(sti, ddof=1)

            # RT time
            rti = (self.t_times_sp - self.template_rpeak_sp) * 1 / self.fs
            pqrst_wave_statistics['rt_time'] = (self.t_time_sp - self.template_rpeak_sp) * 1 / self.fs
            pqrst_wave_statistics['rt_time_std'] = np.std(rti, ddof=1)

            # QT time
            qti = (self.t_times_sp - self.q_times_sp) * 1 / self.fs
            pqrst_wave_statistics['qt_time'] = (self.t_time_sp - self.q_time_sp) * 1 / self.fs
            pqrst_wave_statistics['qt_time_std'] = np.std(qti, ddof=1)

            # PT time
            pti = (self.t_times_sp - self.p_times_sp) * 1 / self.fs
            pqrst_wave_statistics['pt_time'] = (self.t_time_sp - self.p_time_sp) * 1 / self.fs
            pqrst_wave_statistics['pt_time_std'] = np.std(pti, ddof=1)

            # QRS energy
            pqrst_wave_statistics['qrs_energy'] = np.sum(
                np.power(self.median_template_good[self.q_time_sp:self.s_time_sp], 2)
            )
            qrs_eng = np.array([
                np.sum(np.power(self.templates_good[self.q_time_sp:self.s_time_sp, col], 2))
                for col in range(self.templates_good.shape[1])
            ])
            pqrst_wave_statistics['qrs_energy_std'] = np.std(qrs_eng, ddof=1)
            if self.templates_good.shape[1] > 1:
                pqrst_wave_statistics['qrs_energy_approximate_entropy'] = \
                    self.safe_check(pyeeg.ap_entropy(qrs_eng, M=2, R=0.1*np.std(qrs_eng)))
                pqrst_wave_statistics['qrs_energy_higuchi_fractal_dimension'] = hfd(qrs_eng, k_max=10)
                pqrst_wave_statistics['qrs_energy_pearson_coeff'] = np.nan
                pqrst_wave_statistics['qrs_energy_pearson_p_value'] = np.nan
            else:
                pqrst_wave_statistics['qrs_energy_approximate_entropy'] = np.nan
                pqrst_wave_statistics['qrs_energy_higuchi_fractal_dimension'] = np.nan
                pqrst_wave_statistics['qrs_energy_pearson_coeff'] = np.nan
                pqrst_wave_statistics['qrs_energy_pearson_p_value'] = np.nan

        else:
            pqrst_wave_statistics['pq_time'] = np.nan
            pqrst_wave_statistics['pq_time_std'] = np.nan
            pqrst_wave_statistics['pr_time'] = np.nan
            pqrst_wave_statistics['pr_time_std'] = np.nan
            pqrst_wave_statistics['pri_approximate_entropy'] = np.nan
            pqrst_wave_statistics['pri_higuchi_fractal_dimension'] = np.nan
            pqrst_wave_statistics['qr_time'] = np.nan
            pqrst_wave_statistics['qr_time_std'] = np.nan
            pqrst_wave_statistics['rs_time'] = np.nan
            pqrst_wave_statistics['rs_time_std'] = np.nan
            pqrst_wave_statistics['qs_time'] = np.nan
            pqrst_wave_statistics['qs_time_std'] = np.nan
            pqrst_wave_statistics['qsi_approximate_entropy'] = np.nan
            pqrst_wave_statistics['qsi_higuchi_fractal_dimension'] = np.nan
            pqrst_wave_statistics['qsi_pearson_coeff'] = np.nan
            pqrst_wave_statistics['qsi_pearson_p_value'] = np.nan
            pqrst_wave_statistics['st_time'] = np.nan
            pqrst_wave_statistics['st_time_std'] = np.nan
            pqrst_wave_statistics['rt_time'] = np.nan
            pqrst_wave_statistics['rt_time_std'] = np.nan
            pqrst_wave_statistics['qt_time'] = np.nan
            pqrst_wave_statistics['qt_time_std'] = np.nan
            pqrst_wave_statistics['pt_time'] = np.nan
            pqrst_wave_statistics['pt_time_std'] = np.nan
            pqrst_wave_statistics['qrs_energy'] = np.nan
            pqrst_wave_statistics['qrs_energy_std'] = np.nan
            pqrst_wave_statistics['qrs_energy_approximate_entropy'] = np.nan
            pqrst_wave_statistics['qrs_energy_higuchi_fractal_dimension'] = np.nan
            pqrst_wave_statistics['qrs_energy_pearson_coeff'] = np.nan
            pqrst_wave_statistics['qrs_energy_pearson_p_value'] = np.nan

        return pqrst_wave_statistics

    def calculate_r_peak_polarity_statistics(self):

        # Empty dictionary
        r_peak_polarity_statistics = dict()

        # Get positive R-Peak amplitudes
        r_peak_positive = self.templates_good[self.template_rpeak_sp, :] > 0

        # Get negative R-Peak amplitudes
        r_peak_negative = self.templates_good[self.template_rpeak_sp, :] < 0

        # Calculate polarity statistics
        r_peak_polarity_statistics['positive_r_peaks'] = \
            np.sum(r_peak_positive) / self.templates_good.shape[1]
        r_peak_polarity_statistics['negative_r_peaks'] = \
            np.sum(r_peak_negative) / self.templates_good.shape[1]

        return r_peak_polarity_statistics

    def calculate_r_peak_amplitude_statistics(self):

        r_peak_amplitude_statistics = dict()

        if len(self.rpeaks_good) > 1:

            rpeak_indices = self.rpeaks_good.astype(int)
            rpeak_amplitudes = self.signal_filtered[rpeak_indices]

            # Basic statistics
            r_peak_amplitude_statistics['rpeak_min'] = np.min(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_max'] = np.max(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_mean'] = np.mean(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_median'] = np.median(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_std'] = np.std(rpeak_amplitudes, ddof=1)
            r_peak_amplitude_statistics['rpeak_skew'] = sp.stats.skew(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_kurtosis'] = sp.stats.kurtosis(rpeak_amplitudes)

            # Non-linear statistics
            r_peak_amplitude_statistics['rpeak_approximate_entropy'] = \
                self.safe_check(pyeeg.ap_entropy(rpeak_amplitudes, M=2, R=0.1*np.std(rpeak_amplitudes)))
            r_peak_amplitude_statistics['rpeak_sample_entropy'] = \
                self.safe_check(
                    ent.sample_entropy(rpeak_amplitudes, sample_length=2, tolerance=0.1*np.std(rpeak_amplitudes))[0]
                )
            r_peak_amplitude_statistics['rpeak_multiscale_entropy'] = \
                self.safe_check(
                    ent.multiscale_entropy(rpeak_amplitudes, sample_length=2, tolerance=0.1*np.std(rpeak_amplitudes))[0]
                )
            r_peak_amplitude_statistics['rpeak_permutation_entropy'] = \
                self.safe_check(ent.permutation_entropy(rpeak_amplitudes, delay=1))
            r_peak_amplitude_statistics['rpeak_multiscale_permutation_entropy'] = \
                self.safe_check(ent.multiscale_permutation_entropy(rpeak_amplitudes, m=2, delay=1, scale=1)[0])
            r_peak_amplitude_statistics['rpeak_fisher_info'] = fisher_info(rpeak_amplitudes, tau=1, de=2)
            r_peak_amplitude_statistics['rpeak_higuchi_fractal_dimension'] = hfd(rpeak_amplitudes, k_max=10)
            hjorth_parameters = hjorth(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_activity'] = hjorth_parameters[0]
            r_peak_amplitude_statistics['rpeak_complexity'] = hjorth_parameters[1]
            r_peak_amplitude_statistics['rpeak_morbidity'] = hjorth_parameters[2]
            r_peak_amplitude_statistics['rpeak_hurst_exponent'] = pfd(rpeak_amplitudes)
            r_peak_amplitude_statistics['rpeak_svd_entropy'] = svd_entropy(rpeak_amplitudes, tau=2, de=2)
            r_peak_amplitude_statistics['rpeak_petrosian_fractal_dimension'] = pyeeg.pfd(rpeak_amplitudes)

        else:
            r_peak_amplitude_statistics['rpeak_min'] = np.nan
            r_peak_amplitude_statistics['rpeak_max'] = np.nan
            r_peak_amplitude_statistics['rpeak_mean'] = np.nan
            r_peak_amplitude_statistics['rpeak_median'] = np.nan
            r_peak_amplitude_statistics['rpeak_std'] = np.nan
            r_peak_amplitude_statistics['rpeak_skew'] = np.nan
            r_peak_amplitude_statistics['rpeak_kurtosis'] = np.nan
            r_peak_amplitude_statistics['rpeak_approximate_entropy'] = np.nan
            r_peak_amplitude_statistics['rpeak_sample_entropy'] = np.nan
            r_peak_amplitude_statistics['rpeak_multiscale_entropy'] = np.nan
            r_peak_amplitude_statistics['rpeak_permutation_entropy'] = np.nan
            r_peak_amplitude_statistics['rpeak_multiscale_permutation_entropy'] = np.nan
            r_peak_amplitude_statistics['rpeak_fisher_info'] = np.nan
            r_peak_amplitude_statistics['rpeak_higuchi_fractal_dimension'] = np.nan
            r_peak_amplitude_statistics['rpeak_activity'] = np.nan
            r_peak_amplitude_statistics['rpeak_complexity'] = np.nan
            r_peak_amplitude_statistics['rpeak_morbidity'] = np.nan
            r_peak_amplitude_statistics['rpeak_hurst_exponent'] = np.nan
            r_peak_amplitude_statistics['rpeak_svd_entropy'] = np.nan
            r_peak_amplitude_statistics['rpeak_petrosian_fractal_dimension'] = np.nan

        return r_peak_amplitude_statistics

    def calculate_template_correlation_statistics(self):

        # Empty dictionary
        template_correlation_statistics = dict()

        if self.templates_good.shape[1] > 1:

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(np.transpose(self.templates_good))

            # Get upper triangle
            upper_triangle = np.triu(correlation_matrix, k=1).flatten()

            # Get upper triangle index where values are not zero
            upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

            # Get upper triangle values where values are not zero
            upper_triangle = upper_triangle[upper_triangle_index]

            # Calculate correlation matrix statistics
            template_correlation_statistics['template_corr_coeff_median'] = np.median(upper_triangle)
            template_correlation_statistics['template_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        else:
            template_correlation_statistics['template_corr_coeff_median'] = np.nan
            template_correlation_statistics['template_corr_coeff_std'] = np.nan

        return template_correlation_statistics

    def calculate_p_wave_correlation_statistics(self):

        # Empty dictionary
        p_wave_correlation_statistics = dict()

        if self.templates_good.shape[1] > 1:

            # Get start point
            start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(np.transpose(self.templates_good[0:start_sp, :]))

            # Get upper triangle
            upper_triangle = np.triu(correlation_matrix, k=1).flatten()

            # Get upper triangle index where values are not zero
            upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

            # Get upper triangle values where values are not zero
            upper_triangle = upper_triangle[upper_triangle_index]

            # Calculate correlation matrix statistics
            p_wave_correlation_statistics['p_wave_corr_coeff_median'] = np.median(upper_triangle)
            p_wave_correlation_statistics['p_wave_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        else:
            p_wave_correlation_statistics['p_wave_corr_coeff_median'] = np.nan
            p_wave_correlation_statistics['p_wave_corr_coeff_std'] = np.nan

        return p_wave_correlation_statistics

    def calculate_qrs_correlation_statistics(self):

        # Empty dictionary
        qrs_correlation_statistics = dict()

        if self.templates_good.shape[1] > 1:

            # Get start and end points
            start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual
            end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(np.transpose(self.templates_good[start_sp:end_sp, :]))

            # Get upper triangle
            upper_triangle = np.triu(correlation_matrix, k=1).flatten()

            # Get upper triangle index where values are not zero
            upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

            # Get upper triangle values where values are not zero
            upper_triangle = upper_triangle[upper_triangle_index]

            # Calculate correlation matrix statistics
            qrs_correlation_statistics['qrs_corr_coeff_median'] = np.median(upper_triangle)
            qrs_correlation_statistics['qrs_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        else:
            qrs_correlation_statistics['qrs_corr_coeff_median'] = np.nan
            qrs_correlation_statistics['qrs_corr_coeff_std'] = np.nan

        return qrs_correlation_statistics

    def calculate_t_wave_correlation_statistics(self):

        # Empty dictionary
        t_wave_correlation_statistics = dict()

        if self.templates_good.shape[1] > 1:

            # Get end point
            end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(np.transpose(self.templates_good[end_sp:, :]))

            # Get upper triangle
            upper_triangle = np.triu(correlation_matrix, k=1).flatten()

            # Get upper triangle index where values are not zero
            upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

            # Get upper triangle values where values are not zero
            upper_triangle = upper_triangle[upper_triangle_index]

            # Calculate correlation matrix statistics
            t_wave_correlation_statistics['t_wave_corr_coeff_median'] = np.median(upper_triangle)
            t_wave_correlation_statistics['t_wave_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        else:
            t_wave_correlation_statistics['t_wave_corr_coeff_median'] = np.nan
            t_wave_correlation_statistics['t_wave_corr_coeff_std'] = np.nan

        return t_wave_correlation_statistics

    def calculate_bad_template_correlation_statistics(self):

        # Empty dictionary
        bad_template_correlation_statistics = dict()

        if self.templates_bad is not None and self.templates_bad.shape[1] > 1:

            # Get start and end points
            start_sp = self.template_rpeak_sp - self.qrs_start_sp_manual
            end_sp = self.template_rpeak_sp + self.qrs_end_sp_manual

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(np.transpose(self.templates_bad[start_sp:end_sp, :]))

            # Get upper triangle
            upper_triangle = np.triu(correlation_matrix, k=1).flatten()

            # Get upper triangle index where values are not zero
            upper_triangle_index = np.triu(correlation_matrix, k=1).flatten().nonzero()[0]

            # Get upper triangle values where values are not zero
            upper_triangle = upper_triangle[upper_triangle_index]

            # Calculate correlation matrix statistics
            bad_template_correlation_statistics['bad_template_corr_coeff_median'] = np.median(upper_triangle)
            bad_template_correlation_statistics['bad_template_corr_coeff_std'] = np.std(upper_triangle, ddof=1)

        else:
            bad_template_correlation_statistics['bad_template_corr_coeff_median'] = 0
            bad_template_correlation_statistics['bad_template_corr_coeff_std'] = 0

        return bad_template_correlation_statistics

    @staticmethod
    def calculate_decomposition_level(waveform_length, level):

        # Set starting multiplication factor
        factor = 0

        # Set updated waveform length variable
        waveform_length_updated = None

        # If waveform is not the correct length for proposed decomposition level
        if waveform_length % 2**level != 0:

            # Calculate remainder
            remainder = waveform_length % 2**level

            # Loop through multiplication factors until minimum factor found
            while remainder != 0:

                # Update multiplication factor
                factor += 1

                # Update waveform length
                waveform_length_updated = factor * waveform_length

                # Calculate updated remainder
                remainder = waveform_length_updated % 2**level

            return waveform_length_updated

        # If waveform is the correct length for proposed decomposition level
        else:
            return waveform_length

    @staticmethod
    def add_padding(waveform, waveform_length_updated):

        # Calculate required padding
        pad_count = np.abs(len(waveform) - waveform_length_updated)

        # Calculate before waveform padding
        pad_before = int(np.floor(pad_count / 2.0))

        # Calculate after waveform padding
        pad_after = pad_count - pad_before

        # Add padding to waveform
        waveform_padded = np.append(np.zeros(pad_before), np.append(waveform, np.zeros(pad_after)))

        return waveform_padded, pad_before, pad_after

    def stationary_wavelet_transform(self, waveform, wavelet, level):

        # Calculate waveform length
        waveform_length = len(waveform)

        # Calculate minimum waveform length for SWT of certain decomposition level
        waveform_length_updated = self.calculate_decomposition_level(waveform_length, level)

        # Add necessary padding to waveform
        waveform_padded, pad_before, pad_after = self.add_padding(waveform, waveform_length_updated)

        # Compute stationary wavelet transform
        swt = pywt.swtn(waveform_padded, wavelet=wavelet, level=level, start_level=0)

        # Loop through decomposition levels and remove padding
        for lev in range(len(swt)):

            # Approximation
            swt[lev]['a'] = swt[lev]['a'][pad_before:len(waveform_padded)-pad_after]

            # Detail
            swt[lev]['d'] = swt[lev]['d'][pad_before:len(waveform_padded)-pad_after]

        return swt

    def calculate_stationary_wavelet_transform_statistics(self):

        # Empty dictionary
        stationary_wavelet_transform_features = dict()

        # Decomposition level
        decomp_level = 4

        # Stationary wavelet transform
        swt = self.stationary_wavelet_transform(self.median_template, wavelet='db4', level=decomp_level)

        # Set frequency band
        freq_band_low = (3, 10)
        freq_band_med = (10, 30)
        freq_band_high = (30, 45)

        """
        Frequency Domain
        """
        for level in range(len(swt)):

            """
            Detail
            """
            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=swt[level]['d'], fs=self.fs)

            # Get frequency band
            freq_band_low_index = np.logical_and(fxx >= freq_band_low[0], fxx < freq_band_low[1])
            freq_band_med_index = np.logical_and(fxx >= freq_band_med[0], fxx < freq_band_med[1])
            freq_band_high_index = np.logical_and(fxx >= freq_band_high[0], fxx < freq_band_high[1])

            # Calculate maximum power
            max_power_low = np.max(pxx[freq_band_low_index])
            max_power_med = np.max(pxx[freq_band_med_index])
            max_power_high = np.max(pxx[freq_band_high_index])

            # Calculate average power
            mean_power_low = np.trapz(y=pxx[freq_band_low_index], x=fxx[freq_band_low_index])
            mean_power_med = np.trapz(y=pxx[freq_band_med_index], x=fxx[freq_band_med_index])
            mean_power_high = np.trapz(y=pxx[freq_band_high_index], x=fxx[freq_band_high_index])

            # Calculate max/mean power ratio
            stationary_wavelet_transform_features['template_swt_d_' + str(level+1) + '_low_power_ratio'] = \
                max_power_low / mean_power_low
            stationary_wavelet_transform_features['template_swt_d_' + str(level+1) + '_med_power_ratio'] = \
                max_power_med / mean_power_med
            stationary_wavelet_transform_features['template_swt_d_' + str(level+1) + '_high_power_ratio'] = \
                max_power_high / mean_power_high

            """
            Approximation
            """
            # Compute Welch periodogram
            fxx, pxx = signal.welch(x=swt[level]['a'], fs=self.fs)

            # Get frequency band
            freq_band_low_index = np.logical_and(fxx >= freq_band_low[0], fxx < freq_band_low[1])
            freq_band_med_index = np.logical_and(fxx >= freq_band_med[0], fxx < freq_band_med[1])
            freq_band_high_index = np.logical_and(fxx >= freq_band_high[0], fxx < freq_band_high[1])

            # Calculate maximum power
            max_power_low = np.max(pxx[freq_band_low_index])
            max_power_med = np.max(pxx[freq_band_med_index])
            max_power_high = np.max(pxx[freq_band_high_index])

            # Calculate average power
            mean_power_low = np.trapz(y=pxx[freq_band_low_index], x=fxx[freq_band_low_index])
            mean_power_med = np.trapz(y=pxx[freq_band_med_index], x=fxx[freq_band_med_index])
            mean_power_high = np.trapz(y=pxx[freq_band_high_index], x=fxx[freq_band_high_index])

            # Calculate max/mean power ratio
            stationary_wavelet_transform_features['template_swt_a_' + str(level+1) + '_low_power_ratio'] = \
                max_power_low / mean_power_low
            stationary_wavelet_transform_features['template_swt_a_' + str(level+1) + '_med_power_ratio'] = \
                max_power_med / mean_power_med
            stationary_wavelet_transform_features['template_swt_a_' + str(level+1) + '_high_power_ratio'] = \
                max_power_high / mean_power_high

        return stationary_wavelet_transform_features
