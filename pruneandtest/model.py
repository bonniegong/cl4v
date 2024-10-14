import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, codeemb, label, data=None, targets=None):
        self.emb = codeemb
        self.label = label
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class MyNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


