"""TAU (Transformer Association Unit): transformer-style cell pair.

Combine (rnn slot) — ``TransformerAssociationUnit``: a weight-tied residual
block applied along the depth direction, i.e. the Universal Transformer
recurrence adapted to the neurotree. The node feature is injected into the
residual stream carried by the aggregated child hidden state, followed by a
pre-LN feed-forward sublayer and a per-step output norm (the RNN/GAU cells
also normalize every step, keeping the family comparable):

    u  = h + W_x x          (input injection into the residual stream)
    u  = u + FFN(LN(u))     (pre-LN feed-forward sublayer)
    h' = LN(u)

Aggregation (gnn slot) — ``TransformerChildAttention``: masked multi-head
self-attention over the children with A~ = A + I as the attention mask plus
a feed-forward sublayer — the transformer generalization of the GAT
aggregation. ``None`` adjacency keeps the family semantics (no sibling
relationships -> no mixing -> identity), which also preserves both engines'
all-None fast path and their output equivalence.
"""
import numpy as np
import torch
import torch.nn as nn


def _feed_forward(hidden_dim, ff_mult):
    return nn.Sequential(
        nn.Linear(hidden_dim, ff_mult * hidden_dim),
        nn.GELU(),
        nn.Linear(ff_mult * hidden_dim, hidden_dim),
    )


class TransformerAssociationUnit(nn.Module):

    def __init__(self, input_dim, hidden_dim, ff_mult=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.inject = nn.Linear(input_dim, hidden_dim)
        self.ln_ff = nn.LayerNorm(hidden_dim)
        self.ff = _feed_forward(hidden_dim, ff_mult)
        self.ln_out = nn.LayerNorm(hidden_dim)
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, x, h):
        u = h + self.inject(x)
        u = u + self.ff(self.ln_ff(u))
        return self.ln_out(u)


class TransformerChildAttention(nn.Module):

    def __init__(self, hidden_dim, heads=4, ff_mult=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(hidden_dim)
        self.ff = _feed_forward(hidden_dim, ff_mult)

    def blocked_mask(self, adj_list, n_children, device):
        """(B, C, C) bool mask, True = attention blocked.

        Real children attend within A~ = A + I. Padded rows keep their
        self-loop so no softmax row is fully masked (a fully-masked row
        yields NaN, which poisons gradients even though the readout later
        discards padded rows); their finite outputs are masked at readout.
        """
        eye = torch.eye(n_children, dtype=torch.bool, device=device)
        allowed = eye.unsqueeze(0).repeat(len(adj_list), 1, 1)
        for i, a in enumerate(adj_list):
            if a is None:
                continue
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a)
            a = a.to(device=device)
            k = a.shape[-1]
            allowed[i, :k, :k] |= a.reshape(k, k) != 0
        return ~allowed

    def forward(self, adj, Wh):
        # Family semantics (same as GCN/GAT): A=None means "no sibling
        # relationships", so that node's children pass through unmixed.
        # This must hold PER NODE, not per batch group — the recursive and
        # flat engines group the same nodes differently, so any group-level
        # shortcut would make their outputs diverge.
        if all(a is None for a in adj):
            return Wh
        blocked = self.blocked_mask(adj, Wh.shape[1], Wh.device)
        mask = blocked.repeat_interleave(self.heads, dim=0)
        q = self.ln_attn(Wh)
        attn_out, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        out = Wh + attn_out
        out = out + self.ff(self.ln_ff(out))
        none_rows = torch.tensor([a is None for a in adj], device=Wh.device)
        if bool(none_rows.any()):
            out = torch.where(none_rows.view(-1, 1, 1), Wh, out)
        return out
