import torch.nn as nn


class CLSReadout(nn.Module):
    """tau1 readout: the aggregation returns [self; children] with the
    encoder layer already applied; the node's new state IS the self row.
    Leaves went through the same layer (single self token), so no
    zero-child special case exists."""

    def forward(self, hidden, child_counts):
        return hidden[:, :1, :], None
