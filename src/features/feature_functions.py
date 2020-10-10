import numpy as np

class Features():
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
        return features

    def get_features(self, data_df):
        n_rows = data_df.shape[0]
        features = []
        for i in range(n_rows):
            signal = data_df[i,:,:]
            features.append(self.get_signal_features(signal))
        return np.array(features)