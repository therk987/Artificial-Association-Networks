import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedRecurrentUnit(nn.Module):
    """GRU cell combining the node feature with the aggregated child hidden
    state (paper, Section III-J, eq. 34-38)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.X2H = nn.Linear(input_dim, 3 * hidden_dim)
        self.H2H = nn.Linear(hidden_dim, 3 * hidden_dim)

        nn.init.xavier_uniform_(self.X2H.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.H2H.weight.data, gain=1.414)

    def forward(self, x, hidden):
        gate_x = self.X2H(x)
        gate_h = self.H2H(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)

        irh = i_n + (resetgate * h_n)
        irh = F.layer_norm(irh, irh.shape[2:])
        newgate = torch.tanh(irh)

        # equivalent to (1 - inputgate) * newgate + inputgate * hidden
        return newgate + inputgate * (hidden - newgate)
