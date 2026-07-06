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
select/stack ops dominate backward time on GPU. Here hidden states live in
one growing (N, H) buffer and children are fetched with a single gather.
"""
from collections import defaultdict

import torch
import torch.nn as nn

from aan.models.encoders.recursive_encoder import build_cells
from aan.models.encoders.readout_max import MaxpoolReadoutLayer
from aan.data_structures.batch_neurotree import BatchNeuroTree


class CompiledBatch(object):
    """Height-bucketed view of a batch of neurotrees (pure Python, no tensors)."""

    __slots__ = ('levels', 'root_positions', 'node_count')

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
            child_index = []
            max_children = 0
            for node in nodes:
                idxs = [position[id(c)] for c in node.C]
                child_index.append(idxs)
                max_children = max(max_children, len(idxs))
            for node in nodes:
                position[id(node)] = next_pos
                next_pos += 1
            self.levels.append({
                'nodes': nodes,
                'batch_tree': BatchNeuroTree(nodes),
                'child_index': child_index,
                'max_children': max_children,
                'child_counts': [len(node.C) for node in nodes],
                'A_c': [node.A_c for node in nodes],
            })
        self.root_positions = [position[id(root)] for root in roots]
        self.node_count = next_pos


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
        self.readout = MaxpoolReadoutLayer()

    def propagation(self, batch_tree: BatchNeuroTree, node_level=False):
        compiled = CompiledBatch(batch_tree.nodes)
        device = self.zero_hiddens.device

        # row 0 is the zero padding row; levels append their outputs
        all_hidden = self.zero_hiddens.unsqueeze(0)

        for level in compiled.levels:
            n = len(level['nodes'])
            node_features = self.feature_extraction_networks(level['batch_tree'])

            if level['max_children'] == 0:  # a level of pure leaves
                hiddens = self.zero_hiddens.repeat(n, 1, 1)
            else:
                # one padded gather instead of per-node stacks
                idx = torch.zeros(n, level['max_children'], dtype=torch.long, device=device)
                for i, child_idxs in enumerate(level['child_index']):
                    if child_idxs:
                        idx[i, :len(child_idxs)] = torch.as_tensor(
                            child_idxs, dtype=torch.long, device=device)
                child_hiddens = all_hidden[idx]  # (n, maxC, H)

                hiddens = self.gnn(level['A_c'], child_hiddens)
                hiddens, indices = self.readout(hiddens, level['child_counts'])
                level['batch_tree'].setIndices(indices)

            hiddens = self.rnn(node_features, hiddens)

            if node_level:
                self.task_networks(hiddens, level['batch_tree'])

            all_hidden = torch.cat([all_hidden, hiddens.squeeze(1)], dim=0)

        root_idx = torch.as_tensor(compiled.root_positions, dtype=torch.long, device=device)
        return all_hidden[root_idx]

    def forward(self, batch_tree: BatchNeuroTree, node_level=False):
        return self.propagation(batch_tree, node_level)
