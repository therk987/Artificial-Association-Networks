import torch
from torch import nn
import torch.nn.functional as F

from aan.data_structures.batch_neurotree import BatchNeuroTree


class Graph2Vec(nn.Module):
    """Graph-node feature extractor psi_graph (paper, Experiment 5; UPFD uses 10 inputs)."""

    def __init__(self, input_dim=10, out_features=128):
        super().__init__()
        self.out_features = out_features
        self.layer = nn.Linear(input_dim, out_features)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.layer.weight.device
        x = torch.stack(batch_tree.getX()).to(device, dtype=torch.float)
        h = self.layer(x)
        h = F.normalize(h)
        return F.relu(h)


class GraphEdge2Vec(nn.Module):
    """Graph-edge feature extractor for datasets with edge attributes."""

    def __init__(self, input_dim=7, out_features=128):
        super().__init__()
        self.out_features = out_features
        self.layer = nn.Linear(input_dim, out_features)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.layer.weight.device
        x = torch.stack(batch_tree.getX()).to(device, dtype=torch.float)
        h = self.layer(x)
        h = F.normalize(h)
        return F.relu(h)
