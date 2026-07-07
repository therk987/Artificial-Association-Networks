"""Episodic memory as a neurotree structure — not a separate network module.

The unification claim (MAN, recast): recall is just DFC over a tree whose
children are memories. A recall tree is

    root (empty association node)
      A_c = star graph: query <-> every memory slot
      C   = [ query node (domain 'query', x = query vector),
              memory nodes (domain 'memory', x = stored root vectors) ]

so the SAME shared cell that aggregates children in perception performs
retrieval: with the TAU cell the star mask makes this literal key-value
attention (query row attends over all memories); with GCN/GAT cells it is
their respective aggregation. No dedicated memory module exists — only a
store (this file) and a tree builder.

Writing implements the MAN short-term semantics: one bounded FIFO queue per
class, so sampling per class is balanced regardless of how imbalanced the
stream was. Stored vectors are detached; memory is data, not parameters.

Batched recall shares the SAME memory node objects across every query's
recall tree — the multiple-parent structure — so each memory slot is
embedded once per forward pass, not once per query.
"""
from collections import deque

import torch

from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree

QUERY_DOMAIN = 'query'
MEMORY_DOMAIN = 'memory'


def star_adjacency(n_children):
    """A_c for a recall tree: child 0 (the query) connected to every memory,
    memories not connected among themselves. Keeps attention cost O(N)
    in the number of memories instead of O(N^2)."""
    A = torch.zeros(n_children, n_children)
    A[0, 1:] = 1.0
    A[1:, 0] = 1.0
    return A


class EpisodicMemory(object):
    """Per-class bounded FIFO store of detached vectors + recall-tree builder."""

    def __init__(self, capacity_per_class):
        if capacity_per_class < 1:
            raise ValueError('capacity_per_class must be >= 1')
        self.capacity_per_class = capacity_per_class
        self.queues = {}

    def __len__(self):
        return sum(len(q) for q in self.queues.values())

    def classes(self):
        return sorted(self.queues)

    def write(self, vectors, labels):
        """Store a batch of vectors (detached) under their class queues."""
        if len(vectors) != len(labels):
            raise ValueError('vectors and labels must align')
        for vec, label in zip(vectors, labels):
            label = int(label)
            if label not in self.queues:
                self.queues[label] = deque(maxlen=self.capacity_per_class)
            self.queues[label].append(vec.detach().clone())

    def sample_balanced(self, per_class, seed=None):
        """(vectors, labels): up to ``per_class`` most-recent items per class —
        the MAN class-imbalance countermeasure."""
        vectors, labels = [], []
        for label in self.classes():
            queue = self.queues[label]
            take = min(per_class, len(queue))
            for i in range(len(queue) - take, len(queue)):
                vectors.append(queue[i])
                labels.append(label)
        return vectors, labels

    def memory_nodes(self, per_class=None):
        """Fresh NeuroNode leaves for the current memory contents."""
        per_class = per_class or self.capacity_per_class
        vectors, labels = self.sample_balanced(per_class)
        nodes = []
        for vec, label in zip(vectors, labels):
            node = NeuroNode(vec, MEMORY_DOMAIN)
            node.label = label
            nodes.append(node)
        return nodes

    def recall_tree(self, query_vector, memory_nodes=None):
        """One query's recall tree (see module docstring)."""
        if memory_nodes is None:
            memory_nodes = self.memory_nodes()
        if not memory_nodes:
            raise ValueError('memory is empty — write() before recall')
        query = NeuroNode(query_vector, QUERY_DOMAIN)
        children = [query] + list(memory_nodes)
        return NeuroNode(None, None,
                         A_c=star_adjacency(len(children)), C=children)

    def recall_batch(self, query_vectors, per_class=None):
        """BatchNeuroTree of recall trees, one per query, all trees sharing
        the same memory node objects (multiple parents -> each memory slot
        is computed once per forward)."""
        shared = self.memory_nodes(per_class)
        trees = [self.recall_tree(q, memory_nodes=shared) for q in query_vectors]
        for tree in trees:
            tree.reset_state()
        return BatchNeuroTree(trees)
