import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from config.option import device


class Num2Vec(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, out_features):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, out_features)
        self.min = 0
        self.max = 1


    def minmax(self, v):
        v_p = (v - self.min) / (self.max - self.min) * (self.new_max - self.new_min) + self.new_min
        return v_p

    def forward(self, batch_tree):
        batch_x = batch_tree.get('x')
        x = torch.Tensor(batch_x).unsqueeze(1).to(device)
        clamped_batch = x.clamp(self.min - 1, self.max + 1)
        # x = self.minmax(clamped_batch)
        x = clamped_batch

        batch_tree.set('x', x)

        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        Wh = self.fc3(h)
        h_prime = F.normalize(Wh)
        return F.relu(h_prime)
