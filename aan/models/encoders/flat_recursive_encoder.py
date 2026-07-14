"""Flat (level-batched) execution engine for the RAN-series encoder.

This computes EXACTLY the same model as
``recursive_encoder.RecursiveAssociationNeuralNetworks`` — the same shared
cell, the same per-node equations (paper eq. 22-27), each node computed once
from its children's finished hidden states (Algorithm 4). Only the execution
*schedule* differs: instead of recursing node-group by node-group, nodes are
bucketed by their height (leaves first), and every bucket is computed with a
handful of large batched tensor ops.

The bucket structure is compiled from the data for every batch, so the
data-driven property is preserved: the number of executed levels equals the
height of the trees in that batch, samples of different depths can be mixed,
multiple parents reuse a node's row by index, and the weights remain one
shared cell (level-dependent weights of the LAN series are NOT supported
here; use the recursive engine for LAN).

Why: the recursive engine stores hidden states as per-node attributes, which
splits every level's tensor into rows and re-assembles them later; the tiny
select/stack ops dominate backward time on GPU. Here hidden states stay as
per-level tensors: chain-shaped levels pass the previous level's output
through directly, and only levels whose children span multiple earlier
levels materialize a concatenated buffer for one padded gather.
"""
from collections import defaultdict

import torch
import torch.nn as nn

from aan.models.encoders.recursive_encoder import build_cells, build_readout
from aan.data_structures.batch_neurotree import BatchNeuroTree


class CompiledBatch(object):
    """Height-bucketed view of a batch of neurotrees (pure Python, no tensors)."""

    __slots__ = ('levels', 'root_positions', 'node_count', 'roots_are_last_level')

    def __init__(self, roots):
        height = {}
        node_of = {}

        def height_of(node):
            nid = id(node)
            if nid in height:
                return height[nid]
            node_of[nid] = node
            if node.C:
                h = 1 + max(height_of(c) for c in node.C)
            else:
                h = 0
            height[nid] = h
            return h

        for root in roots:
            height_of(root)

        buckets = defaultdict(list)
        for nid, h in height.items():
            buckets[h].append(node_of[nid])

        # global position of each node in the hidden buffer;
        # position 0 is reserved for the zero (padding) row
        position = {}
        next_pos = 1
        self.levels = []
        for h in sorted(buckets):
            nodes = buckets[h]
            level_start = next_pos
            child_index = []
            max_children = 0
            for node in nodes:
                idxs = [position[id(c)] for c in node.C]
                child_index.append(idxs)
                max_children = max(max_children, len(idxs))
            # pre-pad in Python so the engine uploads ONE index tensor per level
            padded_index = [idxs + [0] * (max_children - len(idxs)) for idxs in child_index]
            # chain pattern: every node's single child is the previous level's
            # node at the same row — the gather (and the growing global buffer)
            # can be skipped entirely
            prev_start = self.levels[-1]['start'] if self.levels else None
            prev_len = len(self.levels[-1]['nodes']) if self.levels else None
            prev_contiguous = (
                max_children == 1
                and prev_len == len(nodes)
                and all(idxs == [prev_start + i] for i, idxs in enumerate(padded_index))
            )
            for node in nodes:
                position[id(node)] = next_pos
                next_pos += 1
            self.levels.append({
                'nodes': nodes,
                'start': level_start,
                'batch_tree': BatchNeuroTree(nodes),
                'padded_index': padded_index,
                'max_children': max_children,
                'prev_contiguous': prev_contiguous,
                'child_counts': [len(node.C) for node in nodes],
                'A_c': [node.A_c for node in nodes],
            })
        self.root_positions = [position[id(root)] for root in roots]
        self.node_count = next_pos
        # roots == the last level in batch order (uniform-depth batches)
        last = self.levels[-1]
        self.roots_are_last_level = (
            len(roots) == len(last['nodes'])
            and self.root_positions == list(range(last['start'], last['start'] + len(roots)))
        )


class FlatRecursiveAssociationNeuralNetworks(nn.Module):
    """Level-batched executor for the RAN series (same math, fewer kernels)."""

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 feature_extraction_networks,
                 task_networks,
                 version='gaau'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.feature_extraction_networks = feature_extraction_networks
        self.task_networks = task_networks

        self.type_count = len(self.feature_extraction_networks.type_keys)
        self.input_dim_with_bias = self.input_dim + self.type_count

        self.register_buffer('zero_hiddens', torch.zeros(self.hidden_dim))

        self.rnn, self.gnn = build_cells(version, self.input_dim_with_bias, hidden_dim)
        self.readout = build_readout(version, hidden_dim)

    def propagation(self, batch_tree: BatchNeuroTree, node_level=False):
        compiled = CompiledBatch(batch_tree.nodes)
        device = self.zero_hiddens.device

        # per-level outputs; the global (node_count, H) buffer is only
        # materialized for levels whose children span multiple levels
        level_outputs = []

        for level in compiled.levels:
            n = len(level['nodes'])
            node_features = self.feature_extraction_networks(level['batch_tree'])

            if level['max_children'] == 0:  # a level of pure leaves
                if getattr(self.gnn, 'needs_x', False):
                    empty = self.zero_hiddens.repeat(n, 1, 1)[:, :0, :]
                    hiddens = self.gnn([None] * n, empty, [0] * n,
                                       node_features)
                    hiddens, _ = self.readout(hiddens, [0] * n)
                else:
                    hiddens = self.zero_hiddens.repeat(n, 1, 1)
            else:
                if level['prev_contiguous']:
                    # chain pattern: children ARE the previous level, in order
                    child_hiddens = level_outputs[-1].unsqueeze(1)
                else:
                    # one padded gather from the concatenated earlier levels
                    buffer = torch.cat([self.zero_hiddens.unsqueeze(0)] + level_outputs, dim=0)
                    idx = torch.tensor(level['padded_index'], dtype=torch.long, device=device)
                    child_hiddens = buffer[idx]  # (n, maxC, H)

                if getattr(self.gnn, 'needs_x', False):
                    hiddens = self.gnn(level['A_c'], child_hiddens,
                                       level['child_counts'], node_features)
                elif getattr(self.gnn, 'needs_counts', False):
                    hiddens = self.gnn(level['A_c'], child_hiddens,
                                       level['child_counts'])
                else:
                    hiddens = self.gnn(level['A_c'], child_hiddens)
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

    def forward(self, batch_tree: BatchNeuroTree, node_level=False):
        return self.propagation(batch_tree, node_level)
