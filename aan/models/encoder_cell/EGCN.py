import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGraphConvolutionalLayer(nn.Module):
    """GCN variant with multi-dimensional edge features E_c (paper, Section III-B).

    Each edge channel is normalized independently, aggregated, and the
    per-channel results are mixed by a linear layer.
    """

    def __init__(self, in_features, edge_features):
        super().__init__()
        self.in_features = in_features
        self.edge_features = edge_features
        self.W = nn.Linear(self.in_features * edge_features, self.in_features)
        self.alpha = 0.02
        self.activation = nn.LeakyReLU(self.alpha)

    def stack_edge_graphs(self, batch_edge_list, h_size, e_size, device, dtype):
        graphs = torch.zeros(len(batch_edge_list), e_size, h_size, h_size, device=device, dtype=dtype)
        for i, edge_list in enumerate(batch_edge_list):
            if edge_list is None:
                graphs[i] = torch.eye(h_size, device=device, dtype=dtype).unsqueeze(0).repeat(e_size, 1, 1)
            else:
                for edge in edge_list:
                    graphs[i, :, edge[0], edge[1]] = edge[2]
        return graphs

    def prepare_spectral_graph_inputs(self, E):
        """Return D^-1/2 (E + I) D^-1/2 per edge channel."""
        h_size = E.shape[2]
        eye = torch.eye(h_size, device=E.device, dtype=E.dtype).view(1, 1, h_size, h_size)
        e_tilde = E + eye
        deg = e_tilde.sum(dim=3)
        d_inv_sqrt = deg.pow(-0.5)
        return e_tilde * d_inv_sqrt.unsqueeze(3) * d_inv_sqrt.unsqueeze(2)

    def forward(self, adj, Wh):
        graphs = self.stack_edge_graphs(adj, Wh.shape[1], self.edge_features, Wh.device, Wh.dtype)
        graphs = self.prepare_spectral_graph_inputs(graphs)
        h_prime = torch.matmul(graphs, Wh.unsqueeze(1))  # (B, E, N, H)
        # gather each node's per-channel results contiguously: (B, N, E*H)
        h_prime = h_prime.permute(0, 2, 1, 3).reshape(graphs.shape[0], graphs.shape[2], -1)
        h_prime = self.W(h_prime)
        h_prime = F.layer_norm(h_prime, h_prime.shape[2:])
        return self.activation(h_prime)
