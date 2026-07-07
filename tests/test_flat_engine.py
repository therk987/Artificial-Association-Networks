"""Equivalence tests: the flat (level-batched) engine must compute EXACTLY
the same model as the recursive engine — same outputs, same gradients —
on trees with mixed depths, sibling adjacency, and multiple parents."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.encoders.recursive_encoder import RecursiveAssociationNeuralNetworks
from aan.models.encoders.flat_recursive_encoder import FlatRecursiveAssociationNeuralNetworks
from aan.models.feature_encoders.multiencoder_connector import MultiExtractionConnector

FEATURE_DIM = 8
HIDDEN_DIM = 16


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, FEATURE_DIM)

    def forward(self, batch_tree):
        return self.fc(torch.stack([n.x for n in batch_tree.nodes], dim=0))


def make_engines(version):
    torch.manual_seed(0)
    fe = MultiExtractionConnector(FEATURE_DIM, {'toy': ToyEncoder()})
    recursive = RecursiveAssociationNeuralNetworks(FEATURE_DIM, HIDDEN_DIM, fe, None, version=version)
    flat = FlatRecursiveAssociationNeuralNetworks(FEATURE_DIM, HIDDEN_DIM, fe, None, version=version)
    # both engines must run the SAME cells (shared weights)
    flat.rnn = recursive.rnn
    flat.gnn = recursive.gnn
    return recursive, flat


def leaf():
    return NeuroNode(torch.randn(4), 'toy')


def make_mixed_trees():
    """Chains, bushy nodes with A_c, multiple parents, varying depths."""
    torch.manual_seed(7)
    trees = []
    # depth-3 chain (MNIST-like)
    trees.append(NeuroNode(None, None, C=[NeuroNode(None, None, C=[leaf()])]))
    # depth-1: root with 3 children and an adjacency among siblings
    A = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    trees.append(NeuroNode(None, None, A_c=A, C=[leaf(), leaf(), leaf()]))
    # multiple parents: shared leaf under two mid nodes
    shared = leaf()
    mid_a = NeuroNode(torch.randn(4), 'toy', C=[shared])
    mid_b = NeuroNode(torch.randn(4), 'toy', C=[shared])
    trees.append(NeuroNode(None, None, C=[mid_a, mid_b]))
    # deep chain (depth 6)
    node = leaf()
    for _ in range(5):
        node = NeuroNode(None, None, C=[node])
    trees.append(node)
    # data-carrying internal nodes
    trees.append(NeuroNode(torch.randn(4), 'toy', C=[leaf(), leaf()]))
    return trees


def run_engine(engine, trees):
    for t in trees:
        t.reset_state()
    out = engine(BatchNeuroTree(trees))
    if isinstance(out, list):
        out = torch.stack(out, dim=0)
    return out.view(len(trees), -1)


def test_outputs_match():
    for version in ('ran', 'raan', 'gau', 'gaau', 'tau', 'ptau', 'gtau'):
        recursive, flat = make_engines(version)
        trees = make_mixed_trees()
        out_r = run_engine(recursive, trees)
        out_f = run_engine(flat, trees)
        assert torch.allclose(out_r, out_f, atol=1e-5), \
            '{}: max diff {}'.format(version, (out_r - out_f).abs().max().item())


def test_gradients_match():
    for version in ('gaau', 'tau', 'ptau', 'gtau'):
        recursive, flat = make_engines(version)
        trees = make_mixed_trees()

        out_r = run_engine(recursive, trees)
        out_r.sum().backward()
        grads_r = {n: p.grad.clone() for n, p in recursive.named_parameters() if p.grad is not None}
        for p in recursive.parameters():
            p.grad = None

        out_f = run_engine(flat, trees)
        out_f.sum().backward()
        grads_f = {n: p.grad.clone() for n, p in recursive.named_parameters() if p.grad is not None}

        assert set(grads_r) == set(grads_f), version
        for name in grads_r:
            assert torch.allclose(grads_r[name], grads_f[name], atol=1e-5), \
                '{} {}: max diff {}'.format(version, name,
                                            (grads_r[name] - grads_f[name]).abs().max().item())


def test_variable_depth_batch():
    """Roots at different heights within one batch."""
    recursive, flat = make_engines('gaau')
    trees = make_mixed_trees()[:2] + [leaf()]  # includes a bare-leaf root
    out_r = run_engine(recursive, trees)
    out_f = run_engine(flat, trees)
    assert torch.allclose(out_r, out_f, atol=1e-5)


def _run_all():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print('FAIL', name, '->', repr(exc))
    if failures:
        sys.exit(1)
    print('all tests passed')


if __name__ == '__main__':
    _run_all()
