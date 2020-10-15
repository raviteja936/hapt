import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.utils.file_io import get_user_splits, get_activity_names
from src.datasets.torch_dataset import HAPTDataset
from src.models.ffnn import Net
from src.train.trainloop import TrainLoop

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

train_users, val_users, test_users = get_user_splits()
# print(len(train_users), len(val_users), len(test_users))
activity_names = get_activity_names()
train_dataset = HAPTDataset(train_users)
val_dataset = HAPTDataset(val_users)

sample = train_dataset[1]

trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)

model = Net(sample['x'].shape[0], 12, n_layers=3, n_units=(512, 1024, 512))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter('experiments/runs/experiment_1')

train = TrainLoop(model, trainloader, optimizer, loss_fn, writer, valloader=valloader, print_every=1000)
train.fit(20)
