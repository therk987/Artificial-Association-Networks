from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch import nn
import torch.nn.functional as F
import torch
from config.option import device


class GraphAttentionNetworks(nn.Module):
    def __init__(self):
        super().__init__()

        drop_out = 0

        self.input_dim = 10
        self.output_dim = 2
        self.edge_features = 1
        self.alpha = 0.02

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.gats1 = GAT(128, 128, drop_out, self.alpha)
        # self.gats2 = GAT(128, 128, drop_out, self.alpha)

        self.fc_layer1 = nn.Linear(10, 128)
        self.fc_layer2 = torch.nn.Linear(128, 2)

    #     def dict2matrix(self, batch_dict):

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

        adjs = self.stack_edge_graphs(adj_list, hidden.shape[1], self.edge_features).squeeze(1)

        # graphs = self.prepare_spectral_graph_inputs(adj)

        hidden = self.fc_layer1(hidden.to(device))
        hidden = self.leakyrelu(F.layer_norm(hidden, hidden.shape[2:]))
        hidden = self.gats1(hidden, adjs.to(device))

        node_count = hidden.shape[1]
        hidden = hidden.unsqueeze(1)
        out, indices = F.max_pool2d(hidden, kernel_size=(node_count, 1), return_indices=True)
        out = out.squeeze(1)
        out = out.squeeze(1)
        out = self.fc_layer2(out)

        return F.log_softmax(out, dim=-1)



#
# class GraphAttentionNetworks(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         drop_out = 0
#
#         self.input_dim = 10
#         self.output_dim = 2
#         self.alpha = 0.02
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.gats1 = GAT(128, 128, drop_out, self.alpha)
#         # self.gats2 = GAT(128, 128, drop_out, self.alpha)
#
#         self.fc_layer1 = nn.Linear(10, 128)
#         self.fc_layer2 = torch.nn.Linear(128, 2)
#
#     #     def dict2matrix(self, batch_dict):
#
#     def forward(self, data):
#
#         maximum_size = max([_[0].shape[0] for _ in data])
#         hidden = torch.zeros(len(data), maximum_size, self.input_dim)
#         adj_list = []
#         for idx, hg in enumerate(data):
#             h, adj = hg
#             # if adj is None:
#             #     tmp_adj = torch.eye(hidden.shape[1])
#             # else:
#             #     tmp_adj = torch.zeros(hidden.shape[1], hidden.shape[1])
#             #     tmp_adj[:adj.shape[1], :adj.shape[1]] += adj
#             #
#             adj_list.append(adj)
#             hidden[idx, :h.shape[0], :h.shape[1]] += h
#
#         stacked_adj = torch.stack(adj_list, dim=0).to(device)
#         hidden = hidden.to(device)
#         hidden = self.fc_layer1(hidden)
#         hidden = F.layer_norm(hidden, hidden.shape[2:])
#         hidden = self.leakyrelu(hidden)
#         hidden = self.gats1(hidden, stacked_adj)
#
#         # hidden = self.gats2(hidden, stacked_adj)
#         #         hidden = F.relu(F.layer_norm(hidden, hidden.shape[2:]))  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         #         out = self.W(hidden)
#
#         node_count = hidden.shape[1]
#         hidden = hidden.unsqueeze(1)
#         out, indices = F.max_pool2d(hidden, kernel_size=(node_count, 1), return_indices=True)
#         out = out.squeeze(1)
#         out = out.squeeze(1)
#         out = self.fc_layer2(out)
#
#         return F.log_softmax(out, dim=-1)

#
# class GraphAttentionNetworks(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         drop_out = 0
#
#         self.input_dim = 10
#         self.output_dim = 2
#         self.alpha = 0.02
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.gats1 = GAT(128, 128, drop_out, self.alpha)
#         # self.gats2 = GAT(128, 128, drop_out, self.alpha)
#
#         self.fc_layer1 = nn.Linear(10, 128)
#         self.fc_layer2 = torch.nn.Linear(128, 2)
#
#     #     def dict2matrix(self, batch_dict):
#
#     def forward(self, data):
#
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
#         stacked_adj = torch.stack(adj_list, dim=0).to(device)
#         hidden = hidden.to(device)
#         hidden = self.fc_layer1(hidden)
#         hidden = F.layer_norm(hidden, hidden.shape[2:])
#         hidden = self.leakyrelu(hidden)
#         hidden = self.gats1(hidden, stacked_adj)
#
#         # hidden = self.gats2(hidden, stacked_adj)
#         #         hidden = F.relu(F.layer_norm(hidden, hidden.shape[2:]))  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         #         out = self.W(hidden)
#
#         node_count = hidden.shape[1]
#         hidden = hidden.unsqueeze(1)
#         out, indices = F.max_pool2d(hidden, kernel_size=(node_count, 1), return_indices=True)
#         out = out.squeeze(1)
#         out = out.squeeze(1)
#         out = self.fc_layer2(out)
#
#         return F.log_softmax(out, dim=-1)



class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        self.a = nn.Linear(2 * out_features, 1)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h)
        # Wh = torch.matmul(h, self.W.weight.T)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj != 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        #         attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        #         h_prime = torch.matmul(adj, Wh)
        h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        return self.leakyrelu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a.weight.T[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 1)

        return self.leakyrelu(e)
