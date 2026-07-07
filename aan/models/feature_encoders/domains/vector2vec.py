"""Feature encoder for nodes whose ``x`` is already a vector.

Used by the episodic-memory trees: stored root vectors (domain 'memory')
and query vectors (domain 'query') live in hidden space already, so the
encoder is a single linear projection into the connector's feature size.
"""
import torch
from torch import nn


class VectorEncoder(nn.Module):

    def __init__(self, vector_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(vector_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)

    def forward(self, batch_tree):
        device = self.fc.weight.device
        x = torch.stack([node.x for node in batch_tree.nodes], dim=0)
        return self.fc(x.to(device=device, dtype=self.fc.weight.dtype))
