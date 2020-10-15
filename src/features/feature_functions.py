import numpy as np
from src.features.rough import get_features, get_fft_values, get_psd_values, get_autocorr_values
from tqdm import tqdm


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
        self.function_kwargs = {np.mean: {'axis': 0}, np.std: {'axis': 0}, np.amax: {'axis': 0}, np.amin: {'axis': 0},
                           np.quantile: {'axis': 0}}

    def get_signal_features(self, signal):
        functions = [np.mean, np.std, np.amax, np.amin, np.quantile]
        function_args = {np.quantile: [[0.25, 0.75]]}
        function_kwargs = {np.mean: {'axis': 1}, np.std: {'axis': 1}, np.amax: {'axis': 1}, np.amin: {'axis': 1},
                           np.quantile: {'axis': 1}}
        features = []
        for f in functions:
            args = []
            args += function_args.get(f, [])
            kwargs = function_kwargs.get(f, {})
            features += list(f(signal, *args, **kwargs).flatten())

        for i in range(signal.shape[0]):
            signal_min = np.nanpercentile(signal[i,:], percentile)
            signal_max = np.nanpercentile(signal[i,:], 100 - percentile)
            mph = signal_min + (signal_max - signal_min) / denominator
            # print (signal[i,:].shape)
            features += get_features(*get_fft_values(signal[i,:]), mph)
            features += get_features(*get_psd_values(signal[i,:]), mph)
            features += get_features(*get_autocorr_values(signal[i,:]), mph)
        return features

    def get_features(self, data_df):
        n_rows = data_df.shape[0]
        features = []
        for i in tqdm(range(n_rows)):
            signal = data_df[i,:,:]
            features.append(self.get_signal_features(signal))
        return np.array(features)