import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


window = 50
stride = 25
sensors = ['a1', 'a2', 'a3', 'g1', 'g2', 'g3']
labels_file = 'labels.txt'
names = ['experiment_id', 'user_id', 'activity_id', 'label_start', 'label_end']
path = './data/RawData/'


class ReadSegment():
    def __init__(self, users):
        self.users = users
        self.df_labels = pd.read_csv(path + labels_file, header=None, delim_whitespace=True, names=names)

    def get_labels(self, n, expt, user):
        labels = -1 * np.ones(n)
        label_idxs = self.df_labels[(self.df_labels['user_id'] == user) & (self.df_labels['experiment_id'] == expt)].index
        for idx in label_idxs:
            labels[self.df_labels.iloc[idx].label_start : self.df_labels.iloc[idx].label_end] = self.df_labels.iloc[idx].activity_id
        return labels

    def get_segment_label(self, labels):
        return labels.mode().item()

    def read_signal_files(self):
        self.df_signals = pd.DataFrame(
            columns=['experiment_id', 'user_id', 'activity_id', 'a1', 'a2', 'a3', 'g1', 'g2', 'g3'])
        for user in self.users:
            experiments = self.df_labels.loc[self.df_labels['user_id'] == user].experiment_id.unique()
            for expt in experiments:
                expt_prefix = user_prefix = ''
                if expt < 10:
                    expt_prefix = '0'
                if user < 10:
                    user_prefix = '0'
                acc_file = (path + 'acc_exp' + expt_prefix + '%d_user' % (expt) + user_prefix + '%d.txt' % (user))
                gyro_file = (path + 'gyro_exp' + expt_prefix + '%d_user' % (expt) + user_prefix + '%d.txt' % (user))
                df_acc = pd.read_csv(acc_file, header=None, delim_whitespace=True, names=['a1', 'a2', 'a3'])
                df_gyro = pd.read_csv(gyro_file, header=None, delim_whitespace=True, names=['g1', 'g2', 'g3'])
                assert df_acc.shape[0] == df_gyro.shape[0]
                df_out = pd.concat([df_acc, df_gyro], axis=1)
                df_out['experiment_id'] = expt
                df_out['user_id'] = user
                df_out['activity_id'] = self.get_labels(df_out.shape[0], expt, user)
                self.df_signals = pd.concat([self.df_signals, df_out], axis=0)
        # print (self.df_signals.shape)

    def segment(self):
        self.read_signal_files()
        df_out = np.empty((0, len(sensors), window), float)
        labels = []

        print("LOADING DATA#########################################################")
        for start in tqdm(range(0, self.df_signals.shape[0]-stride, stride)):
            end = start + window

            # label = self.get_segment_label(self.df_signals.iloc[start:end]['activity_id'])
            label = (stats.mode(self.df_signals.iloc[start:end][['activity_id']])[0][0][0])
            if (label == -1) or (self.df_signals.iloc[start]['user_id'] != self.df_signals.iloc[end]['user_id']):
                continue

            labels.append(label-1)
            curr_segment = self.df_signals.iloc[list(range(start, end)), :][sensors]
            df_out = np.append(df_out, np.reshape(np.array(curr_segment), (1, len(sensors), window)), axis=0)

        return df_out, np.array(labels)

if __name__ == "__main__":
    reader = ReadSegment([18, 6, 4, 25, 23, 2])
    x, y = reader.segment()
    print (x.shape, y.shape)