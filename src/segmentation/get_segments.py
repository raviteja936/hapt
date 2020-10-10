import os
import numpy as np
import pandas as pd

window = 50
n_signals = 6
labels_file = './data/RawData/labels.txt'
names = ['experiment_id', 'user_id', 'activity_id', 'label_start', 'label_end']
path = './data/RawData/'

class SegmentFiles:
    def __init__(self):
        self.df = pd.read_csv(labels_file, header=None, delim_whitespace=True, names=names)

    def get_sample_label(self, user, expt, idx):
        activity_id = self.df[(self.df['user_id'] == user) & (self.df['experiment_id'] == expt) & \
                            (idx >= self.df['label_start']) & (idx <= self.df['label_end'])].activity_id

        if activity_id is None:
            return -1
        else:
            return activity_id

    def get_segment_label(self, user, expt, start, end):
        label = self.get_sample_label(user, expt, start)
        for i in range(start+1, end):
            if self.get_sample_label(user, expt, i) != label or label == -1:
                return -1
        return label

    def get_segments_from_file(self, f):
        if isinstance(f, str):
            df_sensors = pd.read_csv(os.path.join(path, f), header=None, delim_whitespace=True)
        else:
            df_sensors = pd.DataFrame()
            for file in f:
                # assert df_acc.shape[0] == df_gyro.shape[0]
                df_sensors = pd.concat([df_sensors, pd.read_csv(os.path.join(path, file), header=None, delim_whitespace=True)], axis=1)

        start = 0
        end = start + window
        df_out = np.empty((0, window, 6), float)
        labels = []
        while (end < df_sensors.shape[0]):
            curr_window = np.reshape(np.array(df_sensors.iloc[list(range(start, end)), :]), (1, window, n_signals))
            label = self.get_segment_label(user, expt, start, end)
            if label == -1:
                start += 25
                end += 25
                continue
            labels.append(label)
            df_out = np.append(df_out, curr_window, axis=0)

            start += 25
            end += 25
        return df_out, np.array(labels)

    def get_segments(self, files):
        df_out = np.empty((0, window, n_signals), float)
        df_labels = np.empty(0, int)

        for f in files:
            file_segments, file_labels = get_segments_from_file(f)
            df_out = np.append(df_out, file_segments)
            df_labels = np.append(df_labels, file_labels)
        return df_out, df_labels