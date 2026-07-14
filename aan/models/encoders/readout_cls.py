import torch
import torch.nn as nn


class CLSReadout(nn.Module):
    """tau1 readout: the aggregation returns [CLS; children] with the
    encoder layer already applied; the parent state IS the CLS row.

    Family leaf semantics are preserved: a node with zero real children
    aggregates to the zero vector (exactly what the max-pool readout's
    zero-padding rule yields for the other cells), so a leaf grouped with
    non-leaf siblings behaves identically in both engines instead of
    receiving the CLS-of-nothing constant.
    """

    def forward(self, hidden, child_counts):
        out = hidden[:, :1, :]
        counts = torch.as_tensor(child_counts, device=hidden.device)
        return out * (counts > 0).view(-1, 1, 1), None
