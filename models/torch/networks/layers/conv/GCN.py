
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device
import numpy as np

class GraphConvolutionalLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features



    #
    # def forward(self, adj, hiddens):
    #
    #     outputs = []
    #     indices = []
    #     for a, h in zip(adj, hiddens):
    #         h_size = h.shape[0]
    #         # print(h_size)
    #
    #         if a is None:
    #             tmp_a = torch.eye(h_size)
    #             tmp_h = h
    #         else:
    #             if type(a) == np.ndarray:
    #                 a = torch.from_numpy(a)
    #             tmp_a = a
    #             a_size = a.shape[1]
    #             tmp_h = h[:a_size]
    #             # tmp_a = torch.zeros(h_size, h_size)
    #             # # print(tmp_adj.shape, a.shape)
    #             # tmp_a[:a.shape[1], :a.shape[1]] += a
    #
    #         graph = tmp_a + torch.eye(tmp_a.shape[1])
    #         d_graph = torch.sum(graph, 1) * torch.eye(graph.shape[1])
    #         r_graph = torch.inverse(d_graph ** 0.5)
    #         adj_dr = torch.matmul(torch.matmul(r_graph, graph), r_graph).to(device)
    #
    #         h_prime = torch.matmul(adj_dr, tmp_h).unsqueeze(0)
    #         hidden, idx = F.max_pool2d(h_prime, kernel_size=(adj_dr.shape[1], 1), return_indices=True)
    #         outputs.append(hidden)
    #         indices.append(idx)
    #
    #     outputs = torch.cat(outputs, dim=0)
    #     indices = torch.cat(indices, dim=0)
    #     return outputs, indices



    def prepare_spectral_graph_inputs(self, graphs):
        graphs = graphs + torch.eye(graphs.shape[1]).unsqueeze(0).repeat(graphs.shape[0], 1, 1)
        d_graph = torch.unsqueeze(torch.sum(graphs, 2), 1) * torch.eye(graphs.shape[1])
        r_graph = torch.inverse(d_graph ** 0.5)
        adj = torch.matmul(torch.matmul(r_graph, graphs), r_graph)
        return adj.to(device)

    def stack_graphs(self, adj_list, h_size):
        adj = []
        for i, a in enumerate(adj_list):
            if a is None:
                tmp_adj = torch.eye(h_size)
            else:
                tmp_adj = torch.zeros(h_size, h_size)
                # print(tmp_adj.shape, a.shape)
                tmp_adj[:a.shape[1], :a.shape[1]] += a
            adj.append(tmp_adj)

        graphs = torch.stack(adj, dim=0)
        return graphs

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
        adj = self.stack_graphs(adj, Wh.shape[1])
        graphs = self.prepare_spectral_graph_inputs(adj)
        h_prime = torch.matmul(graphs, Wh)
        return h_prime
        # h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        # return self.activation(h_prime)
