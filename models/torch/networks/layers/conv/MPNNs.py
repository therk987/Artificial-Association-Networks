import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device
import numpy as np

class MPNN(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, edge_features):
        super().__init__()
        self.in_features = in_features
        self.edge_features = edge_features

        self.INW = nn.Linear(in_features, in_features)
        nn.init.xavier_uniform_(self.INW.weight.data, gain=1.414)
        self.OUTW = nn.Linear(in_features, in_features)
        nn.init.xavier_uniform_(self.OUTW.weight.data, gain=1.414)
        self.EW = nn.Linear(edge_features, in_features)
        nn.init.xavier_uniform_(self.EW.weight.data, gain=1.414)
        self.a = nn.Linear(in_features, 1)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)

        self.alpha = 0.02
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, edge_idx, Wh):
        Wh1 = torch.matmul(Wh, self.a.weight.T[self.in_features * edge_idx:self.in_features * (edge_idx + 1), :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.in_features * (edge_idx + 1): self.in_features * (edge_idx + 2), :])
        e = Wh1 + Wh2.transpose(2 ,1)
        return self.leaky_relu(e)


    def weight(self, Wh):
        INWh = self.INW(Wh)
        OUTWh = self.OUTW(Wh)
        return INWh, OUTWh


    def relation(self, Wh, INWh, OUTWh, batch_edge_list):

        outputs = []
        for i, E in enumerate(batch_edge_list):
            if E is None:
                EWh = Wh[i] + INWh[i]
                attention = self.a(EWh)
                attention = self.leaky_relu(attention)
                attention = F.softmax(attention, dim=0)
                EWh = EWh * attention

            else:
                E_idx, E_values = E

                if E_values is not None:
                    EWh = self.EW(E_values.to(device)) + INWh[i, E_idx[0]] + OUTWh[i, E_idx[1]] + Wh[i, E_idx[0]]
                else:
                    EWh = INWh[i, E_idx[0]] + OUTWh[i, E_idx[1]] + Wh[i, E_idx[0]]

                attention = self.a(EWh)
                attention = self.leaky_relu(attention)
                attention = F.softmax(attention, dim=0)
                EWh = EWh * attention
            EWh = F.layer_norm(EWh, EWh.shape[1:])
            EWh = self.leaky_relu(EWh)

            IEOWh = EWh.sum(dim=0)
            # print(IEOWh.shape, EWh.shape)
            outputs.append(IEOWh)

        return torch.stack(outputs, dim = 0)


    def forward(self, Ec, Wh):

        INWh, OUTWh = self.weight(Wh)
        EWh = self.relation(Wh, INWh, OUTWh, Ec)
        # print(EWh.shape)
        return EWh.unsqueeze(1)
