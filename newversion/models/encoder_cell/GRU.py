import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config.option import device

class GatedRecurrentUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)
        # self.dropout = nn.Dropout(0.25)

#         self.X2H = nn.Parameter(torch.empty(size=(input_dim, 3 * hidden_dim)))
        self.X2H = nn.Linear(input_dim, 3 * hidden_dim)
        nn.init.xavier_uniform_(self.X2H.weight.data, gain=1.414)

        self.H2H = nn.Linear(hidden_dim, 3 * hidden_dim)
        nn.init.xavier_uniform_(self.H2H.weight.data, gain=1.414)

        self.reset_parameters()


    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

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
        # newgate = self.dropout(newgate)

        # newgate = F.tanh(i_n + (resetgate * h_n))
        # (1 - inputgate) * newgate + inputgate * hidden
        # newgate + inputgate * hidden - inputgate * newgate
        # newgate + inputgate * (hidden - newgate)

        hy = newgate + inputgate * (hidden - newgate)

        return hy






#
# class GatedRecurrentUnit(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
#
#         self.parameter_init()
#
#     def parameter_init(self):
#         for param in self.parameters():
#             if len(param.shape) > 1:
#                 nn.init.xavier_uniform_(param, gain=1.414)
#
#     def forward(self, x, h):
#         h = h.transpose(0, 1)
#         a_h, h = self.rnn(x, h)
#         return h.transpose(0, 1)
