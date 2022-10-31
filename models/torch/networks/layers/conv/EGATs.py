import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device
import numpy as np

class EdgeGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, edge_features):
        super().__init__()
        self.in_features = in_features
        self.edge_features = edge_features

        self.W = nn.Linear(edge_features * in_features, in_features)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        self.a = nn.Linear(2 * in_features * edge_features, 1)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)

        self.alpha = 0.02
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def _prepare_attentional_mechanism_input(self, edge_idx, Wh):
        Wh1 = torch.matmul(Wh, self.a.weight.T[self.in_features * edge_idx:self.in_features * (edge_idx + 1), :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.in_features * (edge_idx + 1): self.in_features * (edge_idx + 2), :])
        e = Wh1 + Wh2.transpose(2 ,1)
        return self.leaky_relu(e)

    def stack_edge_graphs(self, batch_edge_list, h_size, e_size):
        A = torch.zeros(len(batch_edge_list), e_size, h_size, h_size)
        for i, E in enumerate(batch_edge_list):
            if E is None:
                A[i] = torch.eye(h_size).unsqueeze(0).repeat(e_size, 1, 1)
            else:
                E_idx, E_values = E

                if E_values is not None:
                    edge_size = E_values.shape[1]
                    # print(E_values.shape)
                    # print(A.shape)
                    A[i, :edge_size, E_idx[0, :], E_idx[1, :]] = E_values.T
                else:
                    A[i, :, E_idx[0, :], E_idx[1, :]] = 1

        return A


    def forward(self, Ec, Wh):
        adj = self.stack_edge_graphs(Ec, Wh.shape[1], self.edge_features).to(device)

        edge_attention = []
        for i in range(self.edge_features):
            attention = self._prepare_attentional_mechanism_input(i, Wh)
            edge_attention.append(attention)

        edge_attention = torch.stack(edge_attention, dim = 1)
        # e = adj * attention.unsqueeze(1)
        e = adj * edge_attention
        edge_attention = F.softmax(e, dim=3)
        h_prime = torch.matmul(edge_attention, Wh.unsqueeze(1))
        h_prime = h_prime.reshape(edge_attention.shape[0], edge_attention.shape[2], -1)
        h_prime = self.W(h_prime) + Wh
        h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        h_prime = self.leaky_relu(h_prime)

        return h_prime
