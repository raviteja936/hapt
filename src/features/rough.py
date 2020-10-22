from scipy.fftpack import fft
import numpy as np
from detecta import detect_peaks
from scipy.signal import welch

percentile = 5
denominator = 10

def get_first_n_peaks(x, y, no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0] * missing_no_peaks, y_ + [0] * missing_no_peaks


def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y


def get_fft_values(y_values, T=0.0078125, N=128):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0:N // 2])
    signal_min = np.nanpercentile(y_values, percentile)
    signal_max = np.nanpercentile(y_values, 100 - percentile)
    mph = signal_min + (signal_max - signal_min) / denominator
    return get_features(f_values, fft_values, mph)


def get_psd_values(y_values, f_s=50):
    f_values, psd_values = welch(y_values, fs=f_s)
    signal_min = np.nanpercentile(y_values, percentile)
    signal_max = np.nanpercentile(y_values, 100 - percentile)
    mph = signal_min + (signal_max - signal_min) / denominator
    return get_features(f_values, psd_values, mph)


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T=0.0078125, N=128):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    signal_min = np.nanpercentile(y_values, percentile)
    signal_max = np.nanpercentile(y_values, 100 - percentile)
    mph = signal_min + (signal_max - signal_min) / denominator
    return get_features(x_values, autocorr_values, mph)
