import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """Graph attention over children (Velickovic et al. 2017; paper eq. 30-31).

    Attention scores are computed for connected pairs only; disconnected
    entries are masked out before the softmax.
    """

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.a = nn.Linear(2 * in_features, 1)
        nn.init.xavier_uniform_(self.a.weight.data, gain=1.414)

        self.alpha = 0.02
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def stack_graphs(self, adj_list, h_size, device, dtype):
        """Pad per-node adjacency matrices into one (B, N, N) tensor.

        ``None`` means "no relationships among children": the identity is used
        so each child only attends to itself.
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

    def forward(self, adj, Wh):
        # Fast path: A=None for every node means each child only attends to
        # itself; the softmax over a single logit is 1, so output == input
        # (and the attention parameters receive no gradient either way).
        if all(a is None for a in adj):
            return Wh

        e = self._prepare_attentional_mechanism_input(Wh)

        graphs = self.stack_graphs(adj, Wh.shape[1], Wh.device, Wh.dtype)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(graphs != 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        return torch.matmul(attention, Wh)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a.weight.T[:self.in_features, :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.in_features:, :])
        e = Wh1 + Wh2.transpose(2, 1)
        return self.leaky_relu(e)
