import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config.option import device

class EdgeRecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, nonlinearity = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.edge_dim = edge_dim
        # self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True, nonlinearity = nonlinearity)
        self.fc1 = nn.Linear(input_dim + hidden_dim * edge_dim, hidden_dim, bias=False)

        self.activation = nn.LeakyReLU(0.02)
        self.parameter_init()
        # self.dropout = nn.Dropout(0.25)

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, x, h):
        # h = self.dropout(h)
        x1 = torch.cat([x, h], dim = -1)
        x2 = self.fc1(x1)
        # x2 = x2.view(x2.shape[0], x2.shape[1], x2.shape[2]//self.edge_dim, self.edge_dim).sum(-1) / self.edge_dim
        x3 = F.layer_norm(x2, x2.shape[2:])
        return self.activation(x3)

        # h = h.transpose(0, 1)
        # a_h, h = self.rnn(x, h)
        # return h.transpose(0, 1)

