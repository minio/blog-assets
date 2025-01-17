from typing import List

from torch import nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNISTModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()

        self.lin1 = nn.Linear(input_size, hidden_sizes[0])
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.lin4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.lin5 = nn.Linear(hidden_sizes[3], output_size)
        #self.activation = F.relu()
        #self.output_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = F.relu(self.lin4(out))
        out = F.log_softmax(self.lin5(out), dim=1)
        return out
