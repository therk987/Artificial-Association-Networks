from torch.nn import functional as F
import torch.nn as nn
import torch
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from config.option import device

class GraphEdge2Vec(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.layer = nn.Linear(7, out_features)

    def forward(self, batch_tree: BatchNeuroTree):
        batch_x = batch_tree.get('x')
        x = torch.stack(batch_x).to(device)
        Wh = self.layer(x)
        h_prime = F.normalize(Wh)
        return F.relu(h_prime)

