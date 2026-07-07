"""Episodic memory-as-tree tests: store semantics, recall-tree structure,
engine equivalence, and gradient flow — memory is data, the cell retrieves."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.memory.episodic import (EpisodicMemory, star_adjacency,
                                 QUERY_DOMAIN, MEMORY_DOMAIN)
from aan.models.feature_encoders.multiencoder_connector import MultiExtractionConnector
from aan.models.feature_encoders.domains.vector2vec import VectorEncoder
from aan.models.encoders.recursive_encoder import RecursiveAssociationNeuralNetworks
from aan.models.encoders.flat_recursive_encoder import FlatRecursiveAssociationNeuralNetworks

VEC_DIM = 8
FEATURE_DIM = 8
HIDDEN_DIM = 16


def _filled_memory(capacity=4, classes=3, per_class=2, seed=0):
    torch.manual_seed(seed)
    memory = EpisodicMemory(capacity_per_class=capacity)
    for label in range(classes):
        memory.write(torch.randn(per_class, VEC_DIM), [label] * per_class)
    return memory


def test_capacity_eviction_fifo():
    memory = EpisodicMemory(capacity_per_class=2)
    vectors = torch.arange(3, dtype=torch.float).unsqueeze(1).repeat(1, VEC_DIM)
    memory.write(vectors, [0, 0, 0])
    kept, labels = memory.sample_balanced(per_class=5)
    assert len(kept) == 2 and labels == [0, 0]
    assert kept[0][0].item() == 1.0 and kept[1][0].item() == 2.0  # oldest evicted


def test_balanced_sampling_under_imbalance():
    memory = EpisodicMemory(capacity_per_class=10)
    memory.write(torch.randn(9, VEC_DIM), [0] * 9)
    memory.write(torch.randn(1, VEC_DIM), [1])
    _, labels = memory.sample_balanced(per_class=1)
    assert labels == [0, 1]  # one per class despite the 9:1 stream


def test_recall_tree_structure():
    memory = _filled_memory()
    tree = memory.recall_tree(torch.randn(VEC_DIM))
    assert tree.C[0].t_d == QUERY_DOMAIN
    assert all(c.t_d == MEMORY_DOMAIN for c in tree.C[1:])
    n = len(tree.C)
    A = tree.A_c
    assert A.shape == (n, n)
    assert torch.equal(A, star_adjacency(n))
    assert A[1:, 1:].sum() == 0  # memories not connected among themselves


def test_recall_batch_shares_memory_nodes():
    memory = _filled_memory()
    batch = memory.recall_batch(torch.randn(3, VEC_DIM))
    trees = batch.nodes
    assert len(trees) == 3
    shared_ids = [tuple(id(c) for c in t.C[1:]) for t in trees]
    assert shared_ids[0] == shared_ids[1] == shared_ids[2]  # multiple parents
    query_ids = {id(t.C[0]) for t in trees}
    assert len(query_ids) == 3  # queries are per-tree


def test_empty_memory_raises():
    memory = EpisodicMemory(capacity_per_class=2)
    try:
        memory.recall_tree(torch.randn(VEC_DIM))
    except ValueError:
        return
    raise AssertionError('expected ValueError on empty memory')


def _make_engines(version):
    torch.manual_seed(0)
    fe = MultiExtractionConnector(FEATURE_DIM, {
        QUERY_DOMAIN: VectorEncoder(VEC_DIM, FEATURE_DIM),
        MEMORY_DOMAIN: VectorEncoder(VEC_DIM, FEATURE_DIM),
    })
    recursive = RecursiveAssociationNeuralNetworks(FEATURE_DIM, HIDDEN_DIM, fe, None, version=version)
    flat = FlatRecursiveAssociationNeuralNetworks(FEATURE_DIM, HIDDEN_DIM, fe, None, version=version)
    flat.rnn = recursive.rnn
    flat.gnn = recursive.gnn
    return recursive, flat


def _run(engine, memory, queries):
    batch = memory.recall_batch(queries)
    out = engine(batch)
    if isinstance(out, list):
        out = torch.stack(out, dim=0)
    return out.view(len(queries), -1)


def test_recall_forward_engine_equivalence():
    for version in ('gaau', 'tau'):
        recursive, flat = _make_engines(version)
        memory = _filled_memory()
        queries = torch.randn(3, VEC_DIM)
        out_r = _run(recursive, memory, queries)
        out_f = _run(flat, memory, queries)
        assert out_r.shape == (3, HIDDEN_DIM)
        assert torch.allclose(out_r, out_f, atol=1e-5), \
            '{}: max diff {}'.format(version, (out_r - out_f).abs().max().item())


def test_recall_gradients_flow():
    torch.manual_seed(0)
    _, flat = _make_engines('tau')
    memory = _filled_memory()
    out = _run(flat, memory, torch.randn(2, VEC_DIM))
    # NOT out.sum(): TAU ends in a LayerNorm, and LN's input gradient under a
    # CONSTANT upstream gradient is exactly zero (shift invariance) — a plain
    # sum() would show zero grads everywhere upstream by construction.
    (out * torch.randn_like(out)).sum().backward()
    grads = {n for n, p in flat.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any('query' in n for n in grads)
    assert any('memory' in n for n in grads)
    assert any(n.startswith('gnn.') for n in grads)  # attention retrieval trained


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
