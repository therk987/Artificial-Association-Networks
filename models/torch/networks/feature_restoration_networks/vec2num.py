import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from config.option import device


class Vec2Num(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, batch_tree):
        batch_h = batch_tree.get('dh')
        # print(batch_h)
        dh = torch.stack(batch_h)
        h = self.fc1(dh)
        h = F.normalize(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        Wh = self.fc3(h)
        return Wh

    def __repr__(self):
        return self.__class__.__name__
