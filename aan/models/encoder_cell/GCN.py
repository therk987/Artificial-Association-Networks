import torch
import torch.nn as nn
import numpy as np


class GraphConvolutionalLayer(nn.Module):
    """Spectral graph convolution with renormalization (Kipf & Welling 2016).

    Aggregates the children's hidden states with the normalized child
    adjacency matrix: D^-1/2 (A + I) D^-1/2 h  (paper, eq. 25).
    """

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def stack_graphs(self, adj_list, h_size, device, dtype):
        """Pad per-node adjacency matrices into one (B, N, N) tensor.

        ``None`` means "no relationships among children": the identity is used
        so each child only aggregates itself.
        """
        graphs = torch.zeros(len(adj_list), h_size, h_size, device=device, dtype=dtype)
        for i, a in enumerate(adj_list):
            if a is None:
                graphs[i] = torch.eye(h_size, device=device, dtype=dtype)
            else:
                if isinstance(a, np.ndarray):
                    a = torch.from_numpy(a)
                a = a.to(device=device, dtype=dtype)
                n = a.shape[-1]
                graphs[i, :n, :n] += a.reshape(n, n)
        return graphs

    def prepare_spectral_graph_inputs(self, graphs):
        """Return D^-1/2 (A + I) D^-1/2 without materializing dense inverses."""
        eye = torch.eye(graphs.shape[1], device=graphs.device, dtype=graphs.dtype).unsqueeze(0)
        a_tilde = graphs + eye
        deg = a_tilde.sum(dim=2)
        d_inv_sqrt = deg.pow(-0.5)
        return a_tilde * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    def forward(self, adj, Wh):
        # Fast path: A=None for every node means identity aggregation
        # (A~ = 2I normalizes to I), so the output equals the input.
        if all(a is None for a in adj):
            return Wh
        graphs = self.stack_graphs(adj, Wh.shape[1], Wh.device, Wh.dtype)
        norm_adj = self.prepare_spectral_graph_inputs(graphs)
        return torch.matmul(norm_adj, Wh)
