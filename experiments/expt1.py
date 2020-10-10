import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.utils.file_io import get_user_splits
from src.preprocessing.dataset import HAPTDataset
from src.models.ffnn import Net
from src.training.train import Train

(train_users, val_users, test_users) = get_user_splits()
# print(len(train_users), len(val_users), len(test_users))

hapt_dataset = HAPTDataset(train_users)
sample = hapt_dataset[1]

trainloader = DataLoader(hapt_dataset, batch_size=4, shuffle=True, num_workers=0)
model = Net(sample['x'].shape[0], 12, n_layers=3, n_units=(128, 128, 128))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
writer = SummaryWriter('experiments/runs/experiment_1')

train = Train(model, trainloader, optimizer, loss_fn, writer)
train.fit(20)