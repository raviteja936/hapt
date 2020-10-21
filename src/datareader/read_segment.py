import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

signals_means = [0.8258365140465357, -0.011189948135622126, 0.07350957819992998, 0.021580320059638523, -0.009909851091173486, -0.011808718360273666]
signals_stds = [0.38593920553146555, 0.38144292588805123, 0.35258332424300165, 0.5734890112106984, 0.44280906155490224, 0.34660938960079046]
sensors = ['a1', 'a2', 'a3', 'g1', 'g2', 'g3']
signal_names = ['experiment_id', 'user_id', 'activity_id'] + sensors
label_names = ['experiment_id', 'user_id', 'activity_id', 'label_start', 'label_end']
labels_to_use = [1, 2, 3, 4, 5, 6]
label_file = 'labels.txt'
path = './data/RawData/'


class ReadSegment:
    def __init__(self, users):
        self.users = users
        self.labels = pd.read_csv(path + label_file, header=None, delim_whitespace=True, names=label_names)
        self.signals = pd.DataFrame(columns=signal_names)

    def get_labels(self, n, expt, user):
        labels = -1 * np.ones(n)
        label_idxs = self.labels[(self.labels['user_id'] == user) & (self.labels['experiment_id'] == expt)].index
        for idx in label_idxs:
            labels[self.labels.iloc[idx].label_start: self.labels.iloc[idx].label_end] = self.labels.iloc[idx].activity_id
        return labels

    def read_files(self):
        for user in self.users:
            experiments = self.labels.loc[self.labels['user_id'] == user].experiment_id.unique()
            for expt in experiments:
                expt_prefix = user_prefix = ''
                if expt < 10:
                    expt_prefix = '0'
                if user < 10:
                    user_prefix = '0'
                acc_file = (path + 'acc_exp' + expt_prefix + '%d_user' % (expt) + user_prefix + '%d.txt' % (user))
                gyro_file = (path + 'gyro_exp' + expt_prefix + '%d_user' % (expt) + user_prefix + '%d.txt' % (user))
                signals_acc = pd.read_csv(acc_file, header=None, delim_whitespace=True, names=['a1', 'a2', 'a3'])
                signals_gyro = pd.read_csv(gyro_file, header=None, delim_whitespace=True, names=['g1', 'g2', 'g3'])
                assert signals_acc.shape[0] == signals_gyro.shape[0]
                signals_expt = pd.concat([signals_acc, signals_gyro], axis=1)
                signals_expt = 1.0 * (signals_expt - signals_means) / signals_stds
                signals_expt['experiment_id'] = expt
                signals_expt['user_id'] = user
                signals_expt['activity_id'] = self.get_labels(signals_expt.shape[0], expt, user)
                self.signals = pd.concat([self.signals, signals_expt], axis=0)
        print(self.signals.shape)

    def get_stats(self):
        return self.signals[sensors].mean(axis=0), self.signals[sensors].std(axis=0)

    def get_segment_label(self, start, end):
        return int(stats.mode(self.signals.iloc[start:end][['activity_id']])[0][0][0])

    def segment(self, window, stride):
        self.read_files()
        if window == 1:
            segments = self.signals[self.signals['activity_id'].isin(labels_to_use)][sensors]
            labels = self.signals[self.signals['activity_id'].isin(labels_to_use)]['activity_id']
            return np.reshape(np.array(segments), (-1, len(sensors), 1)), np.array(labels)

        else:
            segments = np.empty((0, len(sensors), window), float)
            labels = []

            for start in tqdm(range(0, self.signals.shape[0] - stride, stride)):
                end = start + window
                label = self.get_segment_label(start, end)
                if (label not in labels_to_use) or (self.signals.iloc[start]['user_id'] != self.signals.iloc[end]['user_id']):
                    continue
                labels.append(label - 1)
                curr_segment = self.signals.iloc[list(range(start, end)), :][sensors]
                segments = np.append(segments, np.reshape(np.array(curr_segment), (1, len(sensors), window)), axis=0)
            return segments, np.array(labels)


if __name__ == "__main__":
    reader = ReadSegment([18, 6, 4, 25, 23, 2])
    x, y = reader.segment(50, 25)
    print(x.shape, y.shape)
