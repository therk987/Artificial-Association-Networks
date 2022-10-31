import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device

class MultiLayerPerceptron(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 3)



    def forward(self, h):

        # h = torch.stack(h)
        h = self.layer1(h)
        h = F.normalize(h)
        h = F.relu(h)
        h = self.layer2(h)
        h = F.normalize(h)
        h = F.relu(h)
        h = self.layer3(h)
        h = F.log_softmax(h, dim = -1)
        return h

