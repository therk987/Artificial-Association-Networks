"""Tests for the ported domain feature extractors, dataset loader, and
the GCN/GAT identity fast paths."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.encoder_cell.GATs import GraphAttentionLayer
from aan.models.feature_encoders.domains.image2vec import LeNet_5
from aan.models.feature_encoders.domains.tabular2vec import Tabular2Vec
from aan.models.feature_encoders.domains.graph2vec import Graph2Vec, GraphEdge2Vec
from aan.models.feature_encoders.domains.code2vec import Ast2VectorForPython, Constant2Vec
from aan.datas.code.load import load_compat_pickle, A2Edge, SORT_CFG_PKL


def batch_of(xs, domain='toy'):
    return BatchNeuroTree([NeuroNode(x, domain) for x in xs])


def test_gat_none_adjacency_is_identity():
    gat = GraphAttentionLayer(8)
    h = torch.randn(3, 4, 8)
    out = gat([None, None, None], h)
    assert torch.allclose(out, h)


def test_lenet5_28x28_and_32x32():
    for size, kernel in ((28, 4), (32, 5)):
        net = LeNet_5(output_dim=128, conv3_kernel=kernel)
        batch = batch_of([torch.randn(1, size, size) for _ in range(3)], 'image')
        out = net(batch)
        assert out.shape == (3, 128)
        assert torch.all(out[:, 120:] == 0)  # zero padding


def test_tabular_and_graph_extractors():
    assert Tabular2Vec(4, 128)(batch_of([torch.randn(4)] * 2)).shape == (2, 128)
    assert Graph2Vec(10, 128)(batch_of([torch.randn(10)] * 2)).shape == (2, 128)
    assert GraphEdge2Vec(7, 128)(batch_of([torch.randn(7)] * 2)).shape == (2, 128)


def test_ast2vec_and_constant2vec():
    ast = Ast2VectorForPython(input_dim=25, output_dim=128)
    assert len(ast.keys) > 1  # vocab pkl found and loaded

    nodes = [NeuroNode('Assign', 'class'), NeuroNode('NotARealType', 'class')]
    nodes[0].y = 'Module'
    batch = BatchNeuroTree(nodes)
    out = ast(batch)
    assert out.shape == (2, 128)
    assert nodes[0].label != ast.NULL_INDEX  # known parent type resolved
    assert nodes[1].label == ast.NULL_INDEX  # unknown falls back to <unk>

    const = Constant2Vec(128)
    out = const(batch_of([1.0, torch.tensor([2.0])], 'Num'))
    assert out.shape == (2, 128)


def test_a2edge():
    A = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    edges = A2Edge(A.numpy())
    assert (0, 1, 1.0) in edges and (1, 0, 2.0) in edges
    assert A2Edge(None) is None


def test_sort_pkl_compat_unpickler():
    data = load_compat_pickle(SORT_CFG_PKL)
    assert len(data) > 100
    key, ast_tree, cfg_tree = data[0]
    assert isinstance(cfg_tree, NeuroNode)  # legacy class remapped
    cfg_tree.reset_state()                  # new API works on legacy objects
    assert cfg_tree.count == 0


def test_sort_loader_splits():
    from aan.datas.code.load import SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA
    splits = SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA()
    assert set(splits) >= {'train', 'valid', 'test'}
    n = sum(len(splits[s]['x']) for s in ('train', 'valid', 'test'))
    assert n > 100
    assert len(splits['label_names']) == 6  # six sorting algorithms


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
