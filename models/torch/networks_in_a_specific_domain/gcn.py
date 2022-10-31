from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch import nn
import torch.nn.functional as F
import torch
from config.option import device

#
#
# class EdgeGraphConvolutionalLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, edge_features):
#         super().__init__()
#         self.in_features = in_features
#         self.edge_features = edge_features
#         self.alpha = 0.02
#         self.W = nn.Linear(edge_features * in_features, in_features)
#         nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
#         self.activation = nn.LeakyReLU(self.alpha)
#
#     def prepare_spectral_graph_inputs(self, E):
#         E2 = E + torch.eye(E.shape[2]).unsqueeze(0).unsqueeze(1).repeat(E.shape[0], E.shape[1], 1, 1)
#         d_graph = torch.sum(E2, 3).unsqueeze(-1) * torch.eye(E2.shape[2])
#         r_graph = torch.inverse(d_graph ** 0.5)
#         return torch.matmul(torch.matmul(r_graph, E2), r_graph).to(device)
#
#     def stack_edge_graphs(self, batch_edge_list, h_size, e_size):
#         A = torch.zeros(len(batch_edge_list), e_size, h_size, h_size)
#         for i, E in enumerate(batch_edge_list):
#             if E is None:
#                 A[i] = torch.eye(h_size).unsqueeze(0).repeat(e_size, 1, 1)
#             else:
#                 E_idx, E_values = E
#
#                 if E_values is not None:
#                     edge_size = E_values.shape[1]
#                     A[i, :edge_size, E_idx[0, :], E_idx[1, :]] = E_values.T
#                 else:
#                     A[i, :, E_idx[0, :], E_idx[1, :]] = 1
#
#         return A
#
#     def forward(self, adj, Wh):
#         adj = self.stack_edge_graphs(adj, Wh.shape[1], self.edge_features)
#         graphs = self.prepare_spectral_graph_inputs(adj)
#         h_prime = torch.matmul(graphs, Wh.unsqueeze(1)).sum(dim = 1) / self.edge_features
#             # .reshape(graphs.shape[0], graphs.shape[2], -1)
#         # h_prime = h_prime.view(h_prime.shape[0], h_prime.shape[1], h_prime.shape[2]//self.edge_features, self.edge_features).sum(-1) / self.edge_features
#         # h_prime = self.W(h_prime)
#         # h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
#         # h_prime = self.activation(h_prime)
#
#         return h_prime

#
# class SingleGraphConvolutionalLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features=10, out_features=2):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.input_dim = in_features
#
#         self.fc_layer1 = nn.Linear(in_features, 128)
#         self.fc_layer2 = nn.Linear(128, 128)
#
#         # self.W = nn.Parameter(torch.empty(size=(128, 128)))
#         # nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(0.02)
#         self.fc_layer3 = torch.nn.Linear(128, out_features)
#
#
#
#     def prepare_spectral_graph_inputs(self, E):
#         E2 = E + torch.eye(E.shape[2]).unsqueeze(0).unsqueeze(1).repeat(E.shape[0], E.shape[1], 1, 1)
#         d_graph = torch.sum(E2, 3).unsqueeze(-1) * torch.eye(E2.shape[2])
#         r_graph = torch.inverse(d_graph ** 0.5)
#         return torch.matmul(torch.matmul(r_graph, E2), r_graph).to(device)
#
#     def stack_edge_graphs(self, batch_edge_list, h_size, e_size):
#         A = torch.zeros(len(batch_edge_list), e_size, h_size, h_size)
#         for i, E in enumerate(batch_edge_list):
#             if E is None:
#                 A[i] = torch.eye(h_size).unsqueeze(0).repeat(e_size, 1, 1)
#             else:
#                 E_idx, E_values = E
#
#                 if E_values is not None:
#                     edge_size = E_values.shape[1]
#                     A[i, :edge_size, E_idx[0, :], E_idx[1, :]] = E_values.T
#                 else:
#                     A[i, :, E_idx[0, :], E_idx[1, :]] = 1
#
#         return A
#
#
#     def forward(self, data):
#         maximum_size = max([_[0].shape[0] for _ in data])
#         hidden = torch.zeros(len(data), maximum_size, self.input_dim)
#         adj_list = []
#         for idx, hg in enumerate(data):
#             h, adj = hg
#             if adj is None:
#                 tmp_adj = torch.eye(hidden.shape[1])
#             else:
#                 tmp_adj = torch.zeros(hidden.shape[1], hidden.shape[1])
#                 tmp_adj[:adj.shape[1], :adj.shape[1]] += adj
#
#             adj_list.append(tmp_adj)
#             hidden[idx, :h.shape[0], :h.shape[1]] += h
#
#         stacked_adj = torch.stack(adj_list, dim=0)
#         stacked_adj = self.prepare_spectral_graph_inputs(stacked_adj)
#         hidden = hidden.to(device)
#         hidden = self.fc_layer1(hidden)
#         hidden = self.leakyrelu(F.layer_norm(hidden, hidden.shape[2:]))
#         h_prime = torch.matmul(stacked_adj, hidden)
#         # h_prime = torch.matmul(stacked_adj, h_prime)
#         h_prime = self.fc_layer2(h_prime)
#         # h_prime = F.relu(F.layer_norm(h_prime, h_prime.shape[2:]))
#         h_prime = self.leakyrelu(F.layer_norm(h_prime, h_prime.shape[2:]))
#
#         h_prime = h_prime.max(1)[0]
#         out = self.fc_layer3(h_prime)
#
#         return F.log_softmax(out, dim=-1)


class SingleGraphConvolutionalLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features=10, out_features=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.input_dim = in_features

        self.fc_layer1 = nn.Linear(in_features, 128)
        self.fc_layer2 = nn.Linear(128, 128)
        self.edge_features = 1

        # self.W = nn.Parameter(torch.empty(size=(128, 128)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.02)
        self.fc_layer3 = torch.nn.Linear(128, out_features)

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


    def forward(self, data):
        maximum_size = max([_[0].shape[0] for _ in data])
        hidden = torch.zeros(len(data), maximum_size, self.input_dim)
        adj_list = []
        for idx, hg in enumerate(data):
            h, adj = hg
            adj_list.append(adj)
            hidden[idx, :h.shape[0], :h.shape[1]] += h

        adjs = self.stack_edge_graphs(adj_list, hidden.shape[1], self.edge_features)
        graphs = self.prepare_spectral_graph_inputs(adjs)

        hidden = self.fc_layer1(hidden.to(device))
        hidden = self.leakyrelu(F.layer_norm(hidden, hidden.shape[2:]))
        hidden = torch.matmul(graphs, hidden.unsqueeze(1)).sum(dim = 1) / self.edge_features
        h_prime = self.fc_layer2(hidden)
        h_prime = self.leakyrelu(F.layer_norm(h_prime, h_prime.shape[2:]))
        h_prime = h_prime.max(1)[0]
        out = self.fc_layer3(h_prime)
        return F.log_softmax(out, dim=-1)



    #
    # def forward(self, data):
    #
    #     h = [_[0] for _ in data]
    #     adj = [_[1] for _ in data]
    #
    #     maximum_size = max([_.shape[0] for _ in h])
    #     hidden = torch.zeros(len(h), maximum_size, self.in_features)
    #     for idx, hid in enumerate(h):
    #         hidden[idx, :hid.shape[0], :hid.shape[1]] += hid
    #
    #     hidden = hidden.to(device)
    #     hidden = self.fc_layer1(hidden)
    #
    #     hidden = F.relu(
    #         F.layer_norm(hidden, hidden.shape[2:]))  # h.shape: (N, in_features), Wh.shape: (N, out_features)
    #     Wh = torch.matmul(hidden, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
    #
    #     adj_list = []
    #     for a in adj:
    #         if a is None:
    #             tmp_adj = torch.eye(Wh.shape[1])
    #         else:
    #             tmp_adj = torch.zeros(Wh.shape[1], Wh.shape[1])
    #             tmp_adj[:a.shape[1], :a.shape[1]] += a.squeeze(0)
    #         adj_list.append(tmp_adj)
    #     adj = torch.stack(adj_list, dim=0).to(device)
    #
    #     h_prime = torch.matmul(adj, Wh)
    #     h_prime = F.relu(F.layer_norm(h_prime, h_prime.shape[2:]))
    #     h_prime = h_prime.max(1)[0]
    #     out = self.fc_layer2(h_prime)
    #
    #     return F.log_softmax(out, dim=-1)


