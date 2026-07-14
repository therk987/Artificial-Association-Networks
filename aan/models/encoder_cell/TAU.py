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

Depth variants (E3 deep-chain regime, depth 81): the per-step output norm
makes the stack post-LN along depth — the identity path is renormalized at
every level, the known post-LN failure mode at depth. Two fixes:

    variant='pre'   (ptau) h' = u — pure pre-LN residual stream; the
                    identity path crosses all levels unnormalized.
    variant='gated' (gtau) h' = (1-z) h + z LN(u), z = sigmoid(W_z[h, LN(u)]
                    + b_z), b_z = -2 — GTrXL-style convex gate biased to
                    carry; preserves h without renormalization the way the
                    GAU family does.

Aggregation (gnn slot) — ``TransformerChildAttention``: masked multi-head
self-attention over the children with A~ = A + I as the attention mask plus
a feed-forward sublayer — the transformer generalization of the GAT
aggregation. ``None`` adjacency keeps the family semantics (no sibling
relationships -> no mixing -> identity), which also preserves both engines'
all-None fast path and their output equivalence.

CAVEAT (semantic difference vs GCN/GAT): for those cells an EXPLICIT
identity adjacency coincides mathematically with ``None`` (GCN: A=I
normalizes to I; GAT: a single allowed position softmaxes to weight 1).
For TAU they differ — an explicit A=I still applies the value/output
projections and the FF sublayer to each child. Datasets that mean "no
sibling relationships" must pass ``None`` (the repository loaders do),
not an identity matrix.
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

    def __init__(self, input_dim, hidden_dim, ff_mult=2, variant='post'):
        super().__init__()
        if variant not in ('post', 'pre', 'gated', 'inject', 'identity'):
            raise ValueError('unknown TAU variant: %r' % (variant,))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.variant = variant
        self.inject = nn.Linear(input_dim, hidden_dim)
        if variant != 'inject':
            self.ln_ff = nn.LayerNorm(hidden_dim)
            self.ff = _feed_forward(hidden_dim, ff_mult)
            self.ln_out = nn.LayerNorm(hidden_dim)
        if variant == 'gated':
            self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)
        if variant == 'gated':
            nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x, h):
        if self.variant == 'identity':
            # tau1: the node's whole update (self token + children, one
            # encoder layer) happened in the aggregation; h IS the state.
            return h
        u = h + self.inject(x)
        if self.variant == 'inject':
            # tau1: the whole encoder layer lives in the CLS aggregation;
            # the combine only injects the node input into the residual
            # stream (u = h_CLS + W x), keeping one cell = one layer.
            return u
        u = u + self.ff(self.ln_ff(u))
        if self.variant == 'pre':
            return u
        out = self.ln_out(u)
        if self.variant == 'post':
            return out
        z = torch.sigmoid(self.gate(torch.cat([h, out], dim=-1)))
        return (1 - z) * h + z * out


class ParentChildAttention(nn.Module):
    """tau1 aggregation: ONE pre-LN transformer encoder layer over
    [self; children] --- the parent's own embedded input is the query row
    and its output row IS the node's new state. Tree-native: no synthetic
    CLS token; the tree already provides the aggregator, the node itself.
    Leaves follow the same rule with zero children (one layer over the
    single self token), so every node applies exactly one encoder layer.

    Restores the family's cell-for-cell design law (GAU cell = one GRU
    step, RAN = one RNN step, graph slot = one GCN/GAT round) for the
    transformer, and stays inside the DFC contract: the cell reads only
    the node's own input and its children's finished hidden states.

    Mask semantics: the self row attends every real child and itself, and
    every real child attends the self row (the parent collects even when
    A=None); child<->child attention follows A~ = A + I exactly as in
    TransformerChildAttention; padded rows keep a self-loop only.
    """

    needs_counts = True
    needs_x = True

    def __init__(self, hidden_dim, inject, heads=4, ff_mult=2,
                 sibling_pe=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.sibling_pe = sibling_pe
        self.inject = inject           # shared with the combine (embedding)
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ln_ff = nn.LayerNorm(hidden_dim)
        self.ff = _feed_forward(hidden_dim, ff_mult)

    def blocked_mask(self, adj_list, counts, n_children, device):
        """(B, C+1, C+1) bool mask over [self; children], True = blocked."""
        B = len(adj_list)
        allowed = torch.zeros(B, n_children + 1, n_children + 1,
                              dtype=torch.bool, device=device)
        allowed[:, torch.arange(n_children + 1),
                torch.arange(n_children + 1)] = True  # self-loops
        for i, a in enumerate(adj_list):
            k = int(counts[i]) if counts is not None else n_children
            if k > 0:
                allowed[i, 0, 1:k + 1] = True   # self -> real children
                allowed[i, 1:k + 1, 0] = True   # real children -> self
            if a is None:
                continue                        # siblings do not mix
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a)
            a = a.to(device=device)
            ka = a.shape[-1]
            allowed[i, 1:ka + 1, 1:ka + 1] |= a.reshape(ka, ka) != 0
        return ~allowed

    def forward(self, adj, Wh, child_counts=None, x=None):
        B, C, H = Wh.shape
        if self.sibling_pe and C > 0:
            Wh = Wh + sinusoidal_pe(C, H, device=Wh.device, dtype=Wh.dtype)
        parent = self.inject(x) if x is not None else Wh.new_zeros(B, 1, H)
        if parent.dim() == 2:
            parent = parent.unsqueeze(1)
        tokens = torch.cat([parent, Wh], dim=1)
        blocked = self.blocked_mask(adj, child_counts, C, Wh.device)
        mask = blocked.repeat_interleave(self.heads, dim=0)
        q = self.ln_attn(tokens)
        attn_out, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        out = tokens + attn_out
        out = out + self.ff(self.ln_ff(out))
        return out


def sinusoidal_pe(n, dim, device=None, dtype=None):
    """(n, dim) sinusoidal position table (Vaswani et al., 2017, eq. PE)."""
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    div = torch.exp(idx * (-np.log(10000.0) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype) if dtype is not None else pe


class TransformerChildAttention(nn.Module):

    def __init__(self, hidden_dim, heads=4, ff_mult=2, sibling_pe=False):
        """``sibling_pe=True`` adds a sinusoidal encoding of each child's
        position (list order) to its state before attention, restoring
        sibling-order sensitivity (by default the cell family is
        permutation-invariant over siblings given the adjacency). Off by
        default: none of the paper's datasets encode order in sibling
        position, and the engines' A=None identity semantics must hold."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.sibling_pe = sibling_pe
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
        none_flags = [a is None for a in adj]
        if all(none_flags):
            return Wh
        unmixed = Wh
        if self.sibling_pe:
            Wh = Wh + sinusoidal_pe(Wh.shape[1], self.hidden_dim,
                                    device=Wh.device, dtype=Wh.dtype)
        blocked = self.blocked_mask(adj, Wh.shape[1], Wh.device)
        mask = blocked.repeat_interleave(self.heads, dim=0)
        q = self.ln_attn(Wh)
        attn_out, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        out = Wh + attn_out
        out = out + self.ff(self.ln_ff(out))
        if any(none_flags):
            none_rows = torch.tensor(none_flags, device=Wh.device).view(-1, 1, 1)
            out = torch.where(none_rows, unmixed, out)
        return out
