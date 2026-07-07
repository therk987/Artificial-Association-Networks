"""TAU decoder pair — the DFD (deconvolution) counterparts of the TAU cell.

``TAUDecoderUnit`` inverts the combine step: the parent's residual-stream
state is refined by a pre-LN feed-forward sublayer and split into (a) the
reconstructed input injection x_hat and (b) the stream handed down to the
children (paper, Algorithm 5's RNN^-1 slot, transformer form).

``TAUChildDistribution`` inverts the aggregation: child slots (position
embeddings) CROSS-ATTEND to the parent state to claim their share of the
stream — the attention analogue of unpooling, masked by A~ = A + I when a
sibling adjacency is present.
"""
import torch
import torch.nn as nn


def _feed_forward(dim, ff_mult=2):
    return nn.Sequential(
        nn.Linear(dim, ff_mult * dim),
        nn.GELU(),
        nn.Linear(ff_mult * dim, dim),
    )


class TAUDecoderUnit(nn.Module):
    """h_parent -> (x_hat, h_child_stream), residual pre-LN style."""

    def __init__(self, out_dim, hidden_dim, ff_mult=2):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.ln_ff = nn.LayerNorm(hidden_dim)
        self.ff = _feed_forward(hidden_dim, ff_mult)
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.to_x = nn.Linear(hidden_dim, out_dim)
        self.to_child = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        u = h + self.ff(self.ln_ff(h))
        u = self.ln_out(u)
        return self.to_x(u), self.to_child(u)


class TAUChildDistribution(nn.Module):
    """Distribute a parent stream to ``n`` child slots via cross-attention.

    Child queries are position embeddings (up to ``max_children``); the
    parent state is the single key/value. Returns (B, n, H).
    """

    def __init__(self, hidden_dim, heads=4, max_children=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_embedding = nn.Embedding(max_children, hidden_dim)
        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(hidden_dim)
        self.ff = _feed_forward(hidden_dim)

    def forward(self, parent_state, n_children):
        b = parent_state.shape[0]
        slots = self.slot_embedding(
            torch.arange(n_children, device=parent_state.device))
        q = self.ln_q(slots.unsqueeze(0).expand(b, -1, -1))
        kv = self.ln_kv(parent_state.view(b, 1, -1))
        mixed, _ = self.attn(q, kv, kv, need_weights=False)
        out = q + mixed
        return out + self.ff(self.ln_ff(out))
