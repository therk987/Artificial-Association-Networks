"""S0 — neurotree as attention masks (the scale-evidence substrate).

Two executors over one compiled representation:

(a) ``MaskedDFCEngine`` (engine='mask', tau only) — DFC-faithful: the exact
    tau-cell math per height level, but with the sibling attention masks
    PRECOMPILED per batch instead of rebuilt inside the cell. Output- and
    gradient-equivalent to the flat/recursive engines (tests enforce this);
    the compiled-mask path is what standard attention infra can consume.

(b) ``AncestorMaskTransformer`` — the structured model for the S1 scaling
    study: a standard pre-LN transformer stack over ALL nodes packed into
    one sequence, where a token may attend only to its DESCENDANTS (+self)
    and depth is encoded by a level embedding. Identical block code to a
    vanilla transformer — only the mask differs — which is exactly the
    controlled comparison S1 needs. NOT equivalent to DFC (one-shot
    ancestor attention vs level-wise composition); that difference is an
    S1 ablation, not a bug.

Trees in a batch are packed into one token sequence; the masks have no
cross-tree edges, so isolation comes for free (graph-transformer batching).
"""
import torch
from torch import nn

from aan.models.encoders.recursive_encoder import build_cells
from aan.models.encoders.flat_recursive_encoder import (CompiledBatch,
                                                        FlatRecursiveAssociationNeuralNetworks)
from aan.data_structures.batch_neurotree import BatchNeuroTree


def compile_level_masks(level, device):
    """Precompile one level's sibling-attention structure from A_c.

    Returns (none_rows bool (n,1,1), blocked bool (n, maxC, maxC)) or
    (None, None) when the level has no adjacency at all (family semantics:
    nothing is mixed). Same mask semantics as TransformerChildAttention:
    A~ = A + I inside each node's real children, self-loops on padded rows
    (a fully masked softmax row NaN-poisons gradients).
    """
    adj_list = level['A_c']
    none_flags = [a is None for a in adj_list]
    if all(none_flags):
        return None, None
    n, max_c = len(adj_list), level['max_children']
    eye = torch.eye(max_c, dtype=torch.bool, device=device)
    allowed = eye.unsqueeze(0).repeat(n, 1, 1)
    for i, a in enumerate(adj_list):
        if a is None:
            continue
        if not torch.is_tensor(a):
            a = torch.as_tensor(a)
        a = a.to(device=device)
        k = a.shape[-1]
        allowed[i, :k, :k] |= a.reshape(k, k) != 0
    none_rows = torch.tensor(none_flags, device=device).view(-1, 1, 1)
    return none_rows, ~allowed


class MaskedDFCEngine(FlatRecursiveAssociationNeuralNetworks):
    """Level-batched DFC where sibling attention runs on precompiled masks.

    Reuses the tau cell's modules (same parameters, same math) — only the
    Python-side mask construction moves out of the cell into the compiler.
    """

    def __init__(self, input_dim, hidden_dim, feature_extraction_networks,
                 task_networks, version='tau'):
        if version not in ('tau', 'tau2'):
            raise ValueError("engine='mask' supports the tau family only; "
                             'got %r' % version)
        super().__init__(input_dim, hidden_dim, feature_extraction_networks,
                         task_networks, version=version)

    def masked_aggregate(self, level, Wh):
        """TransformerChildAttention.forward with compiler-built masks."""
        none_rows, blocked = compile_level_masks(level, Wh.device)
        if blocked is None:
            return Wh
        gnn = self.gnn
        mask = blocked.repeat_interleave(gnn.heads, dim=0)
        q = gnn.ln_attn(Wh)
        attn_out, _ = gnn.attn(q, q, q, attn_mask=mask, need_weights=False)
        out = Wh + attn_out
        out = out + gnn.ff(gnn.ln_ff(out))
        if bool(none_rows.any()):
            out = torch.where(none_rows, Wh, out)
        return out

    def propagation(self, batch_tree: BatchNeuroTree, node_level=False):
        compiled = CompiledBatch(batch_tree.nodes)
        device = self.zero_hiddens.device
        level_outputs = []

        for level in compiled.levels:
            n = len(level['nodes'])
            node_features = self.feature_extraction_networks(level['batch_tree'])

            if level['max_children'] == 0:
                hiddens = self.zero_hiddens.repeat(n, 1, 1)
            else:
                if level['prev_contiguous']:
                    child_hiddens = level_outputs[-1].unsqueeze(1)
                else:
                    buffer = torch.cat([self.zero_hiddens.unsqueeze(0)] + level_outputs, dim=0)
                    idx = torch.tensor(level['padded_index'], dtype=torch.long, device=device)
                    child_hiddens = buffer[idx]

                hiddens = self.masked_aggregate(level, child_hiddens)
                hiddens, indices = self.readout(hiddens, level['child_counts'])
                if indices is not None:
                    level['batch_tree'].setIndices(indices)

            hiddens = self.rnn(node_features, hiddens)
            if node_level:
                self.task_networks(hiddens, level['batch_tree'])
            level_outputs.append(hiddens.squeeze(1))

        if compiled.roots_are_last_level:
            return level_outputs[-1]
        buffer = torch.cat([self.zero_hiddens.unsqueeze(0)] + level_outputs, dim=0)
        root_idx = torch.as_tensor(compiled.root_positions, dtype=torch.long, device=device)
        return buffer[root_idx]


def compile_ancestor_masks(roots):
    """Pack a batch of neurotrees into one token sequence.

    Returns (nodes in topological order, allowed bool (N, N) where
    allowed[i][j] = j is a descendant of i or j == i, level ids (N,),
    root positions). No cross-tree edges: batching by packing.
    """
    compiled = CompiledBatch(roots)
    nodes, level_ids, position = [], [], {}
    for lv, level in enumerate(compiled.levels):
        for node in level['nodes']:
            position[id(node)] = len(nodes)
            nodes.append(node)
            level_ids.append(lv)

    n = len(nodes)
    allowed = torch.eye(n, dtype=torch.bool)
    # ascending height: a child's descendant set is complete before its parents
    for i, node in enumerate(nodes):
        for child in node.C:
            j = position[id(child)]
            allowed[i] |= allowed[j]
    # CompiledBatch reserves position 0 for its padding row; recompute roots
    roots_pos = [position[id(root)] for root in roots]
    return nodes, allowed, torch.tensor(level_ids, dtype=torch.long), roots_pos


class AncestorMaskTransformer(nn.Module):
    """Structured transformer for S1: vanilla blocks, data-defined mask.

    ``weight_shared=True`` applies ONE block ``n_layers`` times (the
    Universal-Transformer variant: compute scales with data depth);
    ``False`` uses a standard stack. Removing the mask (``structured=False``)
    yields the flat-transformer control arm with identical code.
    """

    def __init__(self, input_dim, hidden_dim, feature_extraction_networks,
                 n_layers=4, heads=4, ff_mult=2, max_levels=64,
                 weight_shared=False, structured=True):
        super().__init__()
        self.feature_extraction_networks = feature_extraction_networks
        self.type_count = len(feature_extraction_networks.type_keys)
        self.proj = nn.Linear(input_dim + self.type_count, hidden_dim)
        self.level_embedding = nn.Embedding(max_levels, hidden_dim)
        self.max_levels = max_levels
        self.heads = heads
        self.n_layers = n_layers
        self.weight_shared = weight_shared
        self.structured = structured

        def block():
            return nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, heads, batch_first=True),
                'ln2': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(nn.Linear(hidden_dim, ff_mult * hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(ff_mult * hidden_dim, hidden_dim)),
            })

        self.blocks = nn.ModuleList([block() for _ in range(1 if weight_shared else n_layers)])
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, batch_tree: BatchNeuroTree):
        nodes, allowed, level_ids, roots_pos = compile_ancestor_masks(batch_tree.nodes)
        device = self.proj.weight.device
        features = self.feature_extraction_networks(BatchNeuroTree(nodes)).squeeze(1)
        x = self.proj(features.to(device))
        x = x + self.level_embedding(level_ids.clamp(max=self.max_levels - 1).to(device))
        x = x.unsqueeze(0)  # one packed sequence

        blocked = (~allowed).to(device) if self.structured else None
        for step in range(self.n_layers):
            blk = self.blocks[0 if self.weight_shared else step]
            q = blk['ln1'](x)
            attn_out, _ = blk['attn'](q, q, q, attn_mask=blocked, need_weights=False)
            x = x + attn_out
            x = x + blk['ff'](blk['ln2'](x))
        x = self.ln_out(x).squeeze(0)
        return x[torch.as_tensor(roots_pos, dtype=torch.long, device=device)]
