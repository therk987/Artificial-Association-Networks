"""Attention-pool readout (Set Transformer PMA, minimal form).

A learned seed query attends over the (padded) child dimension and returns
the weighted sum. Replaces the max-pool readout for the 'tau2' cell family:
max-pool forces retrieval-style circuits to fight through an elementwise
max (the M1 bottleneck), whereas attention pooling reads out selectively.
Returns no unpooling indices; engines skip DFD index bookkeeping.
"""
import torch
import torch.nn as nn


class AttentionPoolReadout(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.key = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden, child_counts):
        # hidden: (n, C, H); mask padded child rows before the softmax
        n, c, h = hidden.shape
        counts = torch.as_tensor(child_counts, device=hidden.device).clamp(min=1)
        mask = torch.arange(c, device=hidden.device).unsqueeze(0) < counts.unsqueeze(1)
        scores = (self.key(hidden) @ self.seed) / (h ** 0.5)      # (n, C)
        scores = scores.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)        # (n, C, 1)
        pooled = (alpha * hidden).sum(dim=1, keepdim=True)        # (n, 1, H)
        return pooled, None
