
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DeconvRecurrentNeuralNetwork(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.empty(size=(hidden_dim, self.input_dim + self.hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.alpha = 0.2

        self.activation = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


    def forward(self, hidden):
        xh = torch.matmul(hidden, self.W)
        xh = self.activation(xh)
        x, h = xh.split([self.input_dim, self.hidden_dim], dim = -1)
        return x, h

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.hidden_dim) + ')'


