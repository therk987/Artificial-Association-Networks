
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
        self.W = nn.Linear(self.in_features * edge_features, self.in_features)
        self.alpha = 0.02
        self.activation = nn.LeakyReLU(self.alpha)

    def prepare_spectral_graph_inputs(self, E):
        E2 = E + torch.eye(E.shape[2]).unsqueeze(0).unsqueeze(1).repeat(E.shape[0], E.shape[1], 1, 1)
        d_graph = torch.sum(E2, 3).unsqueeze(-1) * torch.eye(E2.shape[2])
        r_graph = torch.inverse(d_graph ** 0.5)
        return torch.matmul(torch.matmul(r_graph, E2), r_graph).to(device)

    def stack_edge_graphs(self, batch_edge_list, h_size, e_size):
        tmp_adj = torch.zeros(len(batch_edge_list), e_size, h_size, h_size)
        for i, edge_list in enumerate(batch_edge_list):
            if edge_list is None:
                tmp_adj[i] = torch.eye(h_size).unsqueeze(0).repeat(e_size, 1, 1)
            else:
                for edge in edge_list:
                    tmp_adj[i, : , edge[0], edge[1]] = edge[2]
        return tmp_adj


    def forward(self, adj, Wh):
        adj = self.stack_edge_graphs(adj, Wh.shape[1], self.edge_features)
        graphs = self.prepare_spectral_graph_inputs(adj)
        h_prime = torch.matmul(graphs, Wh.unsqueeze(1)).view(graphs.shape[0], graphs.shape[2], -1)
        # print(h_prime.shape)
        h_prime = self.W(h_prime)
        h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        return self.activation(h_prime)
