
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device
import numpy as np

class EdgeGraphConvolutionalLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, edge_features):
        super().__init__()
        self.in_features = in_features
        self.edge_features = edge_features
        self.alpha = 0.02
        self.W = nn.Linear(edge_features * in_features, in_features)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        self.activation = nn.LeakyReLU(self.alpha)

    def prepare_spectral_graph_inputs(self, E):
        E2 = E + torch.eye(E.shape[2]).unsqueeze(0).unsqueeze(1).repeat(E.shape[0], E.shape[1], 1, 1)
        d_graph = torch.sum(E2, 3).unsqueeze(-1) * torch.eye(E2.shape[2])
        r_graph = torch.inverse(d_graph ** 0.5)
        return torch.matmul(torch.matmul(r_graph, E2), r_graph).to(device)

    def stack_edge_graphs(self, batch_edge_list, h_size, e_size):
        A = torch.zeros(len(batch_edge_list), e_size, h_size, h_size)
        for i, E in enumerate(batch_edge_list):
            if E is None:
                A[i] = torch.eye(h_size).unsqueeze(0).repeat(e_size, 1, 1)
            else:
                E_idx, E_values = E

                if E_values is not None:
                    edge_size = E_values.shape[1]
                    A[i, :edge_size, E_idx[0, :], E_idx[1, :]] = E_values.T
                else:
                    A[i, :, E_idx[0, :], E_idx[1, :]] = 1

        return A

    def forward(self, adj, Wh):
        adj = self.stack_edge_graphs(adj, Wh.shape[1], self.edge_features)
        graphs = self.prepare_spectral_graph_inputs(adj)
        h_prime = torch.matmul(graphs, Wh.unsqueeze(1)).sum(dim = 1) / self.edge_features
            # .reshape(graphs.shape[0], graphs.shape[2], -1)
        # h_prime = h_prime.view(h_prime.shape[0], h_prime.shape[1], h_prime.shape[2]//self.edge_features, self.edge_features).sum(-1) / self.edge_features
        # h_prime = self.W(h_prime)
        # h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        # h_prime = self.activation(h_prime)

        return h_prime
