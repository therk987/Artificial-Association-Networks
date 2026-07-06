import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentNeuralNetwork(nn.Module):
    """RNN-style cell: h' = act(LN(W [x, h])) (paper, eq. 26)."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(0.02)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, x, h):
        x1 = torch.cat([x, h], dim=-1)
        x2 = self.fc1(x1)
        x3 = F.layer_norm(x2, x2.shape[2:])
        return self.activation(x3)
