from src.utils.file_io import get_user_splits
from src.preprocessing.dataset import HAPTDataset
from src.models.ffnn import Net

(train_users, val_users, test_users) = get_user_splits()
# print(len(train_users), len(val_users), len(test_users))

hapt_dataset = HAPTDataset([train_users[0]])
sample = hapt_dataset[1]
print (sample['x'], sample['y'])