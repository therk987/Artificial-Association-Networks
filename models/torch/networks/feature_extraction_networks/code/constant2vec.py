import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Num2Vec(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, out_features):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, h):

        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        Wh = self.fc3(h)
        h_prime = F.normalize(Wh)
        return F.relu(h_prime)

