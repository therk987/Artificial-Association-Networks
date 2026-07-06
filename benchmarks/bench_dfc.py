"""DFC engine throughput benchmark.

Measures trees/sec of the recursive DFC forward pass across tree depths,
branching factors, and batch sizes. Use this to quantify engine changes
(e.g., the planned flattened/level-bucketed rewrite in REVISION_PLAN.md Phase 3).

Run from the repository root:
    python benchmarks/bench_dfc.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from aan.config.option import device
from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks

FEATURE_DIM = 128
HIDDEN_DIM = 128


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, FEATURE_DIM)

    def forward(self, batch_tree):
        x = torch.stack([n.x for n in batch_tree.nodes]).to(self.fc.weight.device)
        return self.fc(x)


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, h, tree):
        if isinstance(h, list):
            h = torch.stack(h)
        return self.fc(h)


def make_tree(depth, branching, with_adjacency):
    """A tree with `depth` levels below the root; leaves carry data."""
    def build(level):
        if level == 0:
            return NeuroNode(torch.randn(16), 'toy')
        children = [build(level - 1) for _ in range(branching)]
        A_c = torch.ones(branching, branching) if with_adjacency else None
        return NeuroNode(None, None, A_c=A_c, C=children)
    return build(depth)


def bench(model, depth, branching, batch_size, with_adjacency, warmup=2, iters=10):
    def new_batch():
        return BatchNeuroTree(
            [make_tree(depth, branching, with_adjacency) for _ in range(batch_size)])

    tasks = ['classification'] * batch_size
    for _ in range(warmup):
        model(new_batch(), tasks)

    batches = [new_batch() for _ in range(iters)]  # exclude build time
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            model(batch, tasks)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return iters * batch_size / elapsed


def main():
    models = {}
    for engine in ('recursive', 'flat'):
        torch.manual_seed(1234)
        models[engine] = ArtificialAssociationNeuralNetworks(
            FEATURE_DIM, HIDDEN_DIM,
            {'toy': ToyEncoder()}, {}, {}, {'classification': ToyHead()},
            version='gaau', engine=engine,
        ).to(device)
        models[engine].eval()

    print('device: {}   (trees/sec, higher is better)'.format(device))
    print('{:>6} {:>10} {:>7} {:>6} {:>12} {:>12} {:>8}'.format(
        'depth', 'branching', 'batch', 'A_c', 'recursive', 'flat', 'speedup'))
    configs = [
        (2, 1, 100), (8, 1, 100), (32, 1, 100),   # chain trees (RNN-like)
        (2, 4, 100), (4, 3, 50),                  # bushy trees (GNN-like)
    ]
    for depth, branching, batch in configs:
        for with_adj in (False, True):
            tps = {engine: bench(models[engine], depth, branching, batch, with_adj)
                   for engine in ('recursive', 'flat')}
            print('{:>6} {:>10} {:>7} {:>6} {:>12.1f} {:>12.1f} {:>7.1f}x'.format(
                depth, branching, batch, 'yes' if with_adj else 'no',
                tps['recursive'], tps['flat'], tps['flat'] / tps['recursive']))


if __name__ == '__main__':
    main()
