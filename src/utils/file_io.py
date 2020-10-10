import pandas as pd
import math
from sklearn.utils import shuffle

test_pc = 0.2
val_pc = 0.2
labels_file = './data/RawData/labels.txt'
# acc_file_prefix = './data/RawData/acc_exp'
# gyro_file_prefix = './data/RawData/gyro_exp'

activities_file = "./data/activity_labels.txt"
names = ['experiment_id', 'user_id', 'activity_id', 'label_start', 'label_end']

# labels = pd.read_csv(activities_file, header=None, delim_whitespace=True, names=['activity_id', 'activity'])

def get_user_splits():
    df = pd.read_csv(labels_file, header=None, delim_whitespace=True, names=names)
    total_users = df['user_id'].unique().shape[0]
    test_ct = math.floor(test_pc * total_users)
    val_ct = math.floor(val_pc * (total_users - test_ct))
    train_ct = total_users - test_ct - val_ct

    users_list = list(shuffle(df['user_id'].unique()))

    train_users = users_list[:train_ct]
    val_users = users_list[train_ct:train_ct+val_ct]
    test_users = users_list[train_ct+val_ct:]
    return (train_users, val_users, test_users)

def get_signal_files(users):
    acc_files = []
    gyro_files = []
    df = pd.read_csv(labels_file, header=None, delim_whitespace=True, names=names)
    for user in users:
        experiments = df.loc[df['user_id'] == user].experiment_id.unique()
        for expt in experiments:
            pre_expt = pre_user = ''
            if expt < 10:
                pre_expt = '0'
            if user < 10:
                pre_user = '0'
            acc_files.append('acc_exp' + pre_expt + '%d_user' % (expt) + pre_user + '%d.txt' % (user))
            gyro_files.append('gyro_exp' + pre_expt + '%d_user' % (expt) + pre_user + '%d.txt' % (user))
    return ([acc_files, gyro_files])

if __name__ == "__main__":
    (train_users, val_users, test_users) = get_user_splits()
    print(len(train_users), len(val_users), len(test_users))
    print (get_signal_files(val_users))