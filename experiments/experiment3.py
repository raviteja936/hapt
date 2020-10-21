import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.utils.file_io import get_user_splits, get_activity_names, get_stats
from src.dataloader.torch_dataloader import CustomDataLoader
from src.models.ffnn import Net
from src.train.trainloop import TrainLoop

'''
Fit the model to preprocessed sensor data with window = 50, stride = 25, basic statistical features + freq domain features
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
# device = torch.device("cuda:0" if use_cuda else "cpu")

batch_size = 256
n_layers = 5
n_units = (512, 1024, 1024, 1024, 512)
lr = 0.001
momentum = 0.9
max_epochs = 20
writer = SummaryWriter("experiments/tensorboard/experiment_3")

activity_names = get_activity_names()
train_users, val_users, test_users = get_user_splits()
# Mean, Std = get_stats(train_users)
# print("Means: ", list(Mean), "Std Devs: ", list(Std))
# print(len(train_users), len(val_users), len(test_users))

# train_dataset = CustomDataLoader(train_users)
# val_dataset = CustomDataLoader(val_users)
# test_dataset = CustomDataLoader(test_users)

# torch.save(train_dataset, './data/preprocessed/train_dataset.pt')
# torch.save(val_dataset, './data/preprocessed/val_dataset.pt')
# torch.save(test_dataset, './data/preprocessed/test_dataset.pt')

train_dataset = torch.load('./data/preprocessed/train_dataset.pt', map_location=device)
val_dataset = torch.load('./data/preprocessed/val_dataset.pt', map_location=device)
# test_dataset = torch.load('./data/preprocessed/test_dataset.pt', map_location=device)

sample = train_dataset[1]
trainloader = CustomDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valloader = CustomDataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

net = Net(sample['x'].shape[0], 12, n_layers=n_layers, n_units=n_units)
net.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

print_every = int(len(train_dataset)/(10 * batch_size))
train = TrainLoop(net, trainloader, optimizer, loss_fn, device, writer, val_loader=valloader, print_every=print_every)
train.fit(max_epochs)
