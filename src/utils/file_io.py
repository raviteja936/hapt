import pandas as pd
import math
from sklearn.utils import shuffle

random_state = 1
test_pc = 0.2
val_pc = 0.2
labels_file = './data/RawData/labels.txt'
names = ['experiment_id', 'user_id', 'activity_id', 'label_start', 'label_end']
activities_file = './data/activity_labels.txt'

def get_activity_names():
    activities = pd.read_csv(activities_file, header=None, delim_whitespace=True, names=['activity_id', 'activity'])
    return list(activities.sort_values(by='activity_id')['activity'])

def get_user_splits():
    df = pd.read_csv(labels_file, header=None, delim_whitespace=True, names=names)
    total_users = df['user_id'].unique().shape[0]
    test_ct = math.floor(test_pc * total_users)
    val_ct = math.floor(val_pc * (total_users - test_ct))
    train_ct = total_users - test_ct - val_ct

    users_list = list(shuffle(df['user_id'].unique(), random_state=random_state))

    train_users = users_list[:train_ct]
    val_users = users_list[train_ct:train_ct+val_ct]
    test_users = users_list[train_ct+val_ct:]
    return (train_users, val_users, test_users)


if __name__ == "__main__":
    (train_users, val_users, test_users) = get_user_splits()
    print((train_users), len(val_users), len(test_users))
    print (get_signal_files(val_users))