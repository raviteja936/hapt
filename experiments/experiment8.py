import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.utils.file_io import get_user_splits, get_activity_names, get_stats
from src.dataloader.torch_dataloader import CustomDataset
from src.models.deepconvlstm import Net
from src.train.trainloop import TrainLoop

'''
Fit the model to raw sensor data as a baseline  
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")
# device = torch.device("cuda:0" if use_cuda else "cpu")

batch_size = 10
lr = 0.001
momentum = 0.9
max_epochs = 40
writer = SummaryWriter("experiments/tensorboard/experiment_7")
window = 128
stride = 64

activity_names = get_activity_names()
train_users, val_users, test_users = get_user_splits()
# Mean, Std = get_stats(train_users)
# print("Means: ", list(Mean), "Std Devs: ", list(Std))
# print(len(train_users), len(val_users), len(test_users))

train_dataset = CustomDataset(train_users[:5], stride, window=window, extract_feat=False)
# val_dataset = CustomDataset(val_users, stride, window=window, extract_feat=False)
# test_dataset = CustomDataset(test_users, window, stride, extract_feat=False)

# torch.save(train_dataset, './data/preprocessed/train_dataset_experiment8.pt')
# torch.save(val_dataset, './data/preprocessed/val_dataset_experiment8.pt')
# torch.save(test_dataset, './data/preprocessed/test_dataset_experiment8.pt')

# train_dataset = torch.load('./data/preprocessed/train_dataset_experiment8.pt', map_location=device)
# val_dataset = torch.load('./data/preprocessed/val_dataset_experiment8.pt', map_location=device)
# test_dataset = torch.load('./data/preprocessed/test_dataset_experiment8.pt', map_location=device)

# sample = train_dataset[1]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net = Net(6)
net.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

# print_every = int(len(train_dataset)/(10 * batch_size))
print_every = 1
train = TrainLoop(net, train_loader, optimizer, loss_fn, device, writer, val_loader=val_loader, print_every=print_every)
train.fit(max_epochs)
