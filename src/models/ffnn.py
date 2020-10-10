import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, n_units=(128, 128, 128)):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(in_dim, n_units[0]))
        curr_dim = n_units[0]
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(curr_dim, n_units[i]))
            curr_dim = n_units[i]
        self.final_layer = nn.Linear(curr_dim, out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    net = Net(36, 4, n_layers=3, n_units=(128, 128, 128))
    print (list(net.parameters()))