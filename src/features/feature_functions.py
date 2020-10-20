import numpy as np
from src.features.rough import get_features, get_fft_values, get_psd_values, get_autocorr_values
from tqdm import tqdm
from tsfresh.feature_extraction import feature_calculators

N = 50
f_s = 50
t_n = 1
T = t_n / N
percentile = 5
denominator = 10


class Features:
    def __init__(self):
        self.functions = [np.mean, np.std, np.amax, np.amin, np.quantile]

        self.function_args = {np.quantile: [[0.25, 0.75]]}
        self.function_kwargs = {np.mean: {'axis': 1}, np.std: {'axis': 1}, np.amax: {'axis': 1}, np.amin: {'axis': 1},
                           np.quantile: {'axis': 1}}

        self.tsf_functions = [feature_calculators.abs_energy, feature_calculators.absolute_sum_of_changes, \
                          feature_calculators.autocorrelation, feature_calculators.kurtosis,
                          feature_calculators.skewness, \
                          feature_calculators.longest_strike_above_mean, feature_calculators.longest_strike_below_mean, \
                          feature_calculators.sample_entropy]

        self.tsf_function_args = {feature_calculators.autocorrelation: [5]}
        self.tsf_function_kwargs = {}

    def get_signal_features(self, signal):

        features = []
        for f in self.functions:
            args = [signal]
            kwargs = {}
            if f in self.function_args:
                args += self.function_args[f]
            if f in self.function_kwargs:
                kwargs = self.function_kwargs[f]
            features += list(f(*args, **kwargs).flatten())

        for i in range(signal.shape[0]):
            for f in self.tsf_functions:
                args = [signal[i, :]]
                kwargs = {}
                if f in self.tsf_function_args:
                    args += self.tsf_function_args[f]
                if f in self.tsf_function_kwargs:
                    kwargs = self.tsf_function_kwargs[f]
                features += list(f(*args, **kwargs).flatten())


            # signal_min = np.nanpercentile(signal[i, :], percentile)
            # signal_max = np.nanpercentile(signal[i, :], 100 - percentile)
            # mph = signal_min + (signal_max - signal_min) / denominator
            # # print (signal[i, :].shape)
            # features += get_features(*get_fft_values(signal[i, :]), mph)
            # features += get_features(*get_psd_values(signal[i, :]), mph)
            # features += get_features(*get_autocorr_values(signal[i, :]), mph)
        return features

    def get_features(self, data_df):
        n_rows = data_df.shape[0]
        features = []
        for i in tqdm(range(n_rows)):
            signal = data_df[i,:,:]
            features.append(self.get_signal_features(signal))
        return np.array(features)