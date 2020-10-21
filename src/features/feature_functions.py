import numpy as np
# from src.features.rough import get_features, get_fft_values, get_psd_values, get_autocorr_values
from tsfresh.feature_extraction import feature_calculators

N = 50
f_s = 50
t_n = 1
T = t_n / N
percentile = 5
denominator = 10


class Features:
    def __init__(self):
        self.functions = [np.mean, np.std, np.amax, np.amin, np.quantile, feature_calculators.abs_energy, \
                          feature_calculators.absolute_sum_of_changes, feature_calculators.autocorrelation, \
                          feature_calculators.kurtosis, feature_calculators.skewness, \
                          feature_calculators.longest_strike_above_mean, feature_calculators.longest_strike_below_mean]
                          # feature_calculators.sample_entropy]

        self.function_args = {np.quantile: [[0.25, 0.75]], feature_calculators.autocorrelation: [5]}
        self.function_kwargs = {}

    def get_signal_features(self, signal):

        features = []

            # signal_min = np.nanpercentile(signal[i, :], percentile)
            # signal_max = np.nanpercentile(signal[i, :], 100 - percentile)
            # mph = signal_min + (signal_max - signal_min) / denominator
            # # print (signal[i, :].shape)
            # features += get_features(*get_fft_values(signal[i, :]), mph)
            # features += get_features(*get_psd_values(signal[i, :]), mph)
            # features += get_features(*get_autocorr_values(signal[i, :]), mph)
        return features

    def get_features(self, segments):
        features = np.empty((segments.shape[0], 0), float)
        for i, f in enumerate(self.functions):
            args = []
            kwargs = {}
            if f in self.function_args:
                args = self.function_args[f]
            if f in self.function_kwargs:
                kwargs = self.function_kwargs[f]
            curr_feat = np.reshape(np.apply_along_axis(f, 2, segments, *args, **kwargs), (segments.shape[0], -1))
            # print("F", i, curr_feat[:2])
            features = np.concatenate((features, curr_feat), axis=1)
        return features