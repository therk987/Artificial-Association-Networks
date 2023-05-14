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

        # self.a = nn.Parameter(torch.empty(size=( 2 * in_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.W = nn.Linear(2 * in_features + edge_features, in_features)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        self.a = nn.Linear(2 * in_features + edge_features, 1)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)

        self.alpha = 0.02
        self.leaky_relu = nn.LeakyReLU(self.alpha)



    def stack_edge_graphs(self, batch_edge_list, h_size, e_size):
        tmp_adj = torch.zeros(len(batch_edge_list), e_size, h_size, h_size)
        for i, edge_list in enumerate(batch_edge_list):
            if edge_list is None:
                tmp_adj[i] = torch.eye(h_size).unsqueeze(0).repeat(e_size, 1, 1)
            else:
                for edge in edge_list:
                    tmp_adj[i, : , edge[0], edge[1]] = edge[2]
        return tmp_adj

    def forward(self, E, Wh):
        edge_features = self.stack_edge_graphs(E, Wh.shape[1], self.edge_features)

        e = self._prepare_attentional_mechanism_input(Wh)

        edge_attention = edge_features * e.unsqueeze(1)

        zero_vec = -9e15 *torch.ones_like(edge_attention)


        attention = torch.where(edge_features != 0, edge_attention, zero_vec)
        attention = F.softmax(attention, dim=3)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a.weight.T[:self.in_features, :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.in_features:, :])
        e = Wh1 + Wh2.transpose(2 ,1)
        return self.leaky_relu(e)


    # def _prepare_attentional_mechanism_input(self, Wh, E):
    #
    #     outputs = []
    #
    #     tmp_adj = torch.zeros(Wh.shape[0], Wh.shape[1], Wh.shape[1]).to(device)
    #     zero_vec = -9e15 *torch.ones_like(tmp_adj)
    #
    #
    #
    #     for edge in E:
    #         Wh1 = torch.matmul(Wh, self.a.weight.T[:self.in_features, :])
    #         Wh2 = torch.matmul(Wh, self.a.weight.T[self.in_features:, :])
    #         # Wh1 = torch.matmul(Wh, self.a[:self.in_features, :])
    #         # Wh2 = torch.matmul(Wh, self.a[self.in_features:, :])
    #         e = Wh1 + Wh2.transpose(2, 1)
    #         e = edge * self.leaky_relu(e)
    #         attention = torch.where(edge != 0, e, zero_vec)
    #         attention = F.softmax(attention, dim=2)
    #
    #         i = edge[0]
    #         j = edge[1]
    #         edge_feature = edge[2]
    #         input_features = torch.cat([Wh[:,i], Wh[:,j] , edge_feature], dim = 2)
    #         output = self.W(input_features)
    #         attention = self.a(input_features)
    #
    #         outputs.append(attention * output)
    #
    #
    #     e = Wh1 + Wh2.transpose(2 ,1)
    #
    #
    #     return self.leaky_relu(e)
