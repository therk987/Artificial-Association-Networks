import torch
from torch import nn
import torch.nn.functional as F

from aan.data_structures.batch_neurotree import BatchNeuroTree


class Tabular2Vec(nn.Module):
    """Tabular feature extractor psi_tabular (paper, Experiment 5; Iris uses 4 inputs)."""

    def __init__(self, input_dim=4, output_size=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_size)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.layer1.weight.device
        x = torch.stack(batch_tree.getX()).to(device, dtype=torch.float)
        h = self.layer1(x)
        h = F.normalize(h)
        return F.relu(h)
