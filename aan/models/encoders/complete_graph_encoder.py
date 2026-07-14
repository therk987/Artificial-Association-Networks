"""Complete-graph association encoder — the recursive AAN cell run as a
graph transformer, with structure GIVEN, SOFT, or INDUCED.

Motivation (2026-07-11 discussion). AAN composes one shared association cell
ALONG a provided neurotree (DFC). A transformer is the same idea on a
COMPLETE graph: self-attention IS message passing where every node is
connected to every other node. This module bridges the two so that the
structure "may or may not be given":

    structure='given'   attention is masked to each node's DESCENDANTS (+self)
                         — a node mixes only with its own subtree, i.e. the
                         AAN "compute along the provided structure" behaviour
                         expressed with a complete-graph transformer cell.
                         Equivalent role to mask_engine.AncestorMaskTransformer
                         (structured=True).

    structure='induce'  the edges are DROPPED. Each sample is a complete graph
                         and the self-attention weights ARE a learned soft
                         adjacency — the model must DISCOVER which nodes
                         associate. This is the structure-induction arm the v3
                         verdict left open (V3_FRAMING.md): a falsifiable test
                         of "can self-attention find the tree when it is not
                         provided?".

    structure='soft'    complete graph PLUS a learned positive bias on the
                         provided edges (softplus(beta) added to descendant
                         scores). Uses structure when present, but is free to
                         attend elsewhere — the "tree may or may not be given"
                         middle ground.

The cell is WEIGHT-SHARED across depth by default (Universal-Transformer /
AAN shared-cell recurrence: one block applied ``depth`` times), so compute
scales with the recursion depth, not with a parameter count. A pre-LN
transformer block (masked MHA + FFN) plays both AAN roles at once — the MHA
is the child aggregation (gnn slot), the residual+FFN is the combine (rnn /
TAU slot).

Batching: a batch of neurotrees is packed into one token sequence with
mask_engine.compile_ancestor_masks; a per-sample block mask keeps attention
inside each tree, so packing gives isolation for free (graph-transformer
batching, same as AncestorMaskTransformer).

Induced structure is inspectable: ``forward(..., return_attention=True)``
also returns the last block's per-sample soft adjacency, and
``induced_adjacency`` / ``structure_alignment`` compare it against the gold
descendant mask (the induction metric).
"""
import torch
from torch import nn

from aan.models.encoders.mask_engine import compile_ancestor_masks
from aan.data_structures.batch_neurotree import BatchNeuroTree


class CompleteGraphAssociationEncoder(nn.Module):
    """Recursive complete-graph association cell (given / soft / induced)."""

    STRUCTURES = ('given', 'soft', 'induce')

    def __init__(self, input_dim, hidden_dim, feature_extraction_networks,
                 depth=4, heads=4, ff_mult=2, max_levels=64,
                 structure='induce', weight_shared=True, inject_level=None,
                 inject_position=None):
        super().__init__()
        if structure not in self.STRUCTURES:
            raise ValueError('structure must be one of %r, got %r'
                             % (self.STRUCTURES, structure))
        if hidden_dim % heads:
            raise ValueError('hidden_dim %d not divisible by heads %d'
                             % (hidden_dim, heads))

        self.feature_extraction_networks = feature_extraction_networks
        self.type_count = len(feature_extraction_networks.type_keys)
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.depth = depth
        self.structure = structure
        self.weight_shared = weight_shared
        self.max_levels = max_levels

        # A depth (level) embedding leaks how deep the gold tree is; that is
        # fair when the structure is provided but a giveaway during induction.
        # Default: on unless we are inducing. A within-sample POSITION
        # embedding (canonical packed order) is the induction-safe way to let
        # the model see node order, so default it on when inducing.
        self.inject_level = (structure != 'induce') if inject_level is None else inject_level
        self.inject_position = (structure == 'induce') if inject_position is None else inject_position

        self.proj = nn.Linear(input_dim + self.type_count, hidden_dim)
        if self.inject_level:
            self.level_embedding = nn.Embedding(max_levels, hidden_dim)
        if self.inject_position:
            self.position_embedding = nn.Embedding(max_levels, hidden_dim)
        if structure == 'soft':
            # softplus(beta) >= 0: a bias, never a hard block, so the graph
            # stays complete while the provided edges are up-weighted.
            self.edge_bias = nn.Parameter(torch.zeros(()))

        def block():
            return nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, heads, batch_first=True),
                'ln2': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(nn.Linear(hidden_dim, ff_mult * hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(ff_mult * hidden_dim, hidden_dim)),
            })

        n_blocks = 1 if weight_shared else depth
        self.blocks = nn.ModuleList([block() for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------ masks

    def _positions(self, tree_ids):
        """Within-sample index of each node in packed order (0,1,2,... per tree)."""
        counters = {}
        pos = torch.empty_like(tree_ids)
        for i, t in enumerate(tree_ids.tolist()):
            pos[i] = counters.get(t, 0)
            counters[t] = pos[i].item() + 1
        return pos.clamp(max=self.max_levels - 1)

    def _attn_mask(self, allowed, tree_ids, device):
        """Float additive attention mask (N, N) for the chosen structure.

        same_tree gates isolation in every mode (no cross-sample leakage).
        given:  -inf outside descendants(+self)  -> collapse onto the DAG
        induce: -inf outside the sample           -> complete graph per tree
        soft:   0 inside the sample + softplus(beta) on descendant edges
        """
        n = allowed.shape[0]
        same_tree = (tree_ids.unsqueeze(0) == tree_ids.unsqueeze(1)).to(device)
        neg_inf = torch.finfo(torch.float32).min

        if self.structure == 'given':
            keep = allowed.to(device) & same_tree
            return torch.where(keep, torch.zeros((), device=device),
                               torch.full((), neg_inf, device=device))
        # complete graph within each sample
        base = torch.where(same_tree, torch.zeros((), device=device),
                           torch.full((), neg_inf, device=device))
        if self.structure == 'induce':
            return base
        # soft: add a learned positive bias on the provided edges
        bias = torch.zeros(n, n, device=device)
        edges = allowed.to(device) & same_tree
        bias = bias.masked_fill(edges, 1.0) * nn.functional.softplus(self.edge_bias)
        return base + bias

    # --------------------------------------------------------------- forward

    def _embed(self, nodes, level_ids, tree_ids, device):
        features = self.feature_extraction_networks(BatchNeuroTree(nodes)).squeeze(1)
        x = self.proj(features.to(device))
        if self.inject_level:
            x = x + self.level_embedding(level_ids.clamp(max=self.max_levels - 1).to(device))
        if self.inject_position:
            x = x + self.position_embedding(self._positions(tree_ids).to(device))
        return x

    def encode(self, batch_tree, return_attention=False, depth=None):
        """Run the recursion; return (root embeddings, [soft adjacency]).

        ``depth`` overrides the number of recurrence steps for this call (used
        by the data-driven-depth control: depth = batch tree height). Only
        valid when weight_shared, since a stacked model has fixed blocks.
        """
        nodes, allowed, level_ids, roots_pos, tree_ids = \
            compile_ancestor_masks(batch_tree.nodes)
        device = self.proj.weight.device
        n_steps = self.depth if depth is None else depth
        if depth is not None and not self.weight_shared and depth != self.depth:
            raise ValueError('depth override needs weight_shared=True')

        x = self._embed(nodes, level_ids, tree_ids, device)
        mask = self._attn_mask(allowed, tree_ids, device)
        x = x.unsqueeze(0)  # one packed sequence, batch dim = 1

        last_attn = None
        for step in range(n_steps):
            blk = self.blocks[0 if self.weight_shared else step]
            q = blk['ln1'](x)
            want = return_attention and step == n_steps - 1
            # need_weights=True returns weights averaged over heads (the torch
            # default on every version this repo runs), i.e. the (N, N) soft
            # adjacency; average_attn_weights is intentionally not passed so
            # the call works on torch 1.9 (remote box) through 2.x alike.
            attn_out, attn_w = blk['attn'](q, q, q, attn_mask=mask,
                                           need_weights=want)
            x = x + attn_out
            x = x + blk['ff'](blk['ln2'](x))
            if want:
                last_attn = attn_w.squeeze(0)  # (N, N) averaged over heads

        x = self.ln_out(x).squeeze(0)
        roots = x[torch.as_tensor(roots_pos, dtype=torch.long, device=device)]
        if return_attention:
            return roots, last_attn, tree_ids
        return roots

    def forward(self, batch_tree, return_attention=False, depth=None):
        return self.encode(batch_tree, return_attention=return_attention, depth=depth)

    @staticmethod
    def batch_height(batch_tree):
        """Height of the tallest tree in the batch (for data-driven depth)."""
        _n, _a, level_ids, _r, _t = compile_ancestor_masks(batch_tree.nodes)
        return int(level_ids.max().item()) + 1

    # ------------------------------------------------------- induced structure

    @torch.no_grad()
    def structure_alignment(self, batch_tree):
        """How well the induced soft adjacency matches the gold DAG.

        Returns mean attention mass a node places on its true descendants
        (excluding self), averaged over internal nodes and samples — 1.0 means
        attention lands exactly on the subtree, chance is (#descendants /
        #other-nodes-in-sample). The induction signal the v3 track-3 negative
        was measuring, read straight off the attention map.
        """
        nodes, allowed, _lvl, _roots, tree_ids = compile_ancestor_masks(batch_tree.nodes)
        _roots_emb, attn, tree_ids = self.encode(batch_tree, return_attention=True)
        if attn is None:
            return float('nan')
        device = attn.device
        allowed = allowed.to(device)
        eye = torch.eye(allowed.shape[0], dtype=torch.bool, device=device)
        descendants = allowed & ~eye                       # strict descendants
        same_tree = (tree_ids.unsqueeze(0) == tree_ids.unsqueeze(1)).to(device)
        has_desc = descendants.any(dim=1) & (same_tree.sum(dim=1) > 1)
        if not bool(has_desc.any()):
            return float('nan')
        mass = (attn * descendants.float()).sum(dim=1)     # attention on subtree
        return float(mass[has_desc].mean())
