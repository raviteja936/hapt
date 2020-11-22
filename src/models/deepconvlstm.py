import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_classes, n_conv=64, kernel_size=(5, 1), n_lstm=128, dropout=0.5):
        super().__init__()

        self.n_classes = n_classes
        self.n_conv = n_conv
        self.n_lstm = n_lstm

        self.conv1 = nn.Conv2d(1, n_conv, kernel_size)
        self.conv2 = nn.Conv2d(n_conv, n_conv, kernel_size)
        self.conv3 = nn.Conv2d(n_conv, n_conv, kernel_size)
        self.conv4 = nn.Conv2d(n_conv, n_conv, kernel_size)

        self.lstm1 = nn.LSTM(n_conv * 128, n_lstm)
        self.lstm2 = nn.LSTM(n_lstm, n_lstm)

        self.fc = nn.Linear(n_lstm, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        weight = next(self.parameters()).data

        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(2), x.size(0), -1)
        print(x.shape)
        hidden = (weight.new(1, batch_size, self.n_lstm).zero_(), weight.new(1, batch_size, self.n_lstm).zero_())
        print(hidden[0].shape, hidden[1].shape)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)

        x = x.view(-1, self.n_lstm)
        x = x.contiguous()
        x = self.dropout(x)
        x = self.fc(x)

        out = x.view(batch_size, -1, self.n_classes)[:, -1, :]
        return out, hidden


if __name__ == "__main__":
    net = Net(6)
    print(list(net.parameters()))
