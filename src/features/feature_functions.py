import numpy as np
from src.features.rough import get_fft_values, get_psd_values, get_autocorr_values
from tsfresh.feature_extraction import feature_calculators


percentile = 5
denominator = 10


class Features:
    def __init__(self):
        self.functions = [np.mean, np.std, np.amax, np.amin, np.quantile, feature_calculators.abs_energy,
                          feature_calculators.absolute_sum_of_changes, feature_calculators.autocorrelation,
                          feature_calculators.kurtosis, feature_calculators.skewness,
                          feature_calculators.longest_strike_above_mean,
                          feature_calculators.longest_strike_below_mean, feature_calculators.sample_entropy]

        self.function_args = {np.quantile: [[0.25, 0.75]], feature_calculators.autocorrelation: [5]}
        self.function_kwargs = {}
        self.freq_functions = [get_fft_values, get_psd_values, get_autocorr_values]

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
            curr_feat[curr_feat == np.inf] = 0
            features = np.concatenate((features, curr_feat), axis=1)

        for j, f in enumerate(self.freq_functions):
            curr_feat = np.reshape(np.apply_along_axis(f, 2, segments), (segments.shape[0], -1))
            features = np.concatenate((features, curr_feat), axis=1)
            # curr_feat = (curr_feat - curr_feat.mean(axis=0)) / (0.001 + curr_feat.std(axis=0))
        return features