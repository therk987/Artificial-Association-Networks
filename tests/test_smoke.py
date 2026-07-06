"""CPU smoke tests for the aan package.

Run from the repository root:
    python -m pytest tests/            (with pytest)
    python tests/test_smoke.py         (without pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.data_structures.neurodataloader import NeuroDataset, createNeuroDataloader
from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks
from aan.models.encoders.recursive_encoder import build_cells
from aan.models.encoder_cell.GCN import GraphConvolutionalLayer
from aan.models.encoder_cell.EGCN import EdgeGraphConvolutionalLayer
from aan.models.encoders.readout_max import MaxpoolReadoutLayer

FEATURE_DIM = 8
HIDDEN_DIM = 16


class CountingToyEncoder(nn.Module):
    """Feature extractor stub: linear over node.x, counts processed nodes."""

    def __init__(self, feature_dim=FEATURE_DIM):
        super().__init__()
        self.fc = nn.Linear(4, feature_dim)
        self.processed_nodes = 0

    def forward(self, batch_tree):
        xs = torch.stack([node.x for node in batch_tree.nodes], dim=0)
        self.processed_nodes += len(batch_tree.nodes)
        return self.fc(xs)


class ToyClassifier(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, class_count=3):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, class_count)

    def forward(self, hiddens, tree):
        if isinstance(hiddens, list):
            hiddens = torch.stack(hiddens, dim=0)
        return self.layer(hiddens)


def make_model(version='gaau', engine='flat'):
    encoder = CountingToyEncoder()
    model = ArtificialAssociationNeuralNetworks(
        FEATURE_DIM, HIDDEN_DIM,
        {'toy': encoder}, {}, {}, {'classification': ToyClassifier()},
        version=version, engine=engine,
    )
    return model, encoder


def make_chain_tree(x=None):
    """leaf(toy) -> mid(empty) -> root(empty), like the MNIST demo tree."""
    if x is None:
        x = torch.randn(4)
    leaf = NeuroNode(x, 'toy')
    mid = NeuroNode(None, None, C=[leaf])
    root = NeuroNode(None, None, C=[mid])
    return root


def make_multiparent_tree():
    """One shared leaf under two mid nodes: DFC must compute it exactly once."""
    shared = NeuroNode(torch.randn(4), 'toy')
    mid_a = NeuroNode(torch.randn(4), 'toy', C=[shared])
    mid_b = NeuroNode(torch.randn(4), 'toy', C=[shared])
    root = NeuroNode(None, None, C=[mid_a, mid_b])
    return root, shared


def test_forward_shapes_and_grad():
    model, encoder = make_model()
    batch = BatchNeuroTree([make_chain_tree() for _ in range(5)])
    tasks = ['classification'] * 5

    outputs, h_root, _ = model(batch, tasks)

    assert h_root.shape == (5, HIDDEN_DIM)
    assert len(outputs) == 5
    assert outputs[0].shape[-1] == 3

    loss = torch.stack(outputs).sum()
    loss.backward()
    assert encoder.fc.weight.grad is not None
    assert model.ran.rnn.X2H.weight.grad is not None


def test_all_versions_forward():
    for version in ('ran', 'raan', 'gau', 'gaau', 'egaau'):
        model, _ = make_model(version)
        batch = BatchNeuroTree([make_chain_tree() for _ in range(2)])
        outputs, h_root, _ = model(batch, ['classification'] * 2)
        assert h_root.shape == (2, HIDDEN_DIM), version


def test_version_validation():
    try:
        build_cells('bogus', 10, 10)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for unknown version')


def test_multiparent_computed_once():
    # both engines must extract/compute a shared node exactly once
    for engine in ('recursive', 'flat'):
        model, encoder = make_model(engine=engine)
        root, shared = make_multiparent_tree()
        batch = BatchNeuroTree([root])

        model(batch, ['classification'])

        # 3 'toy' nodes exist (shared leaf + 2 mids); shared extracted once
        assert encoder.processed_nodes == 3, engine

    # recursive-engine bookkeeping: visit counts and stored hidden
    model, _ = make_model(engine='recursive')
    root, shared = make_multiparent_tree()
    model(BatchNeuroTree([root]), ['classification'])
    assert shared.count == 2  # visited via both parents
    assert shared.h is not None


def test_node_reset_state():
    # per-node state is written by the recursive engine
    root, shared = make_multiparent_tree()
    model, _ = make_model(engine='recursive')
    model(BatchNeuroTree([root]), ['classification'])
    assert root.h is not None

    root.reset_state()
    assert root.h is None and root.count == 0
    assert shared.h is None and shared.count == 0

    # tree is reusable after reset
    _, h_root, _ = model(BatchNeuroTree([root]), ['classification'])
    assert h_root.shape == (1, HIDDEN_DIM)


def test_batchneurotree_is_reiterable():
    nodes = [NeuroNode(None, None) for _ in range(3)]
    batch = BatchNeuroTree(nodes)
    assert len(list(batch)) == 3
    assert len(list(batch)) == 3  # the old iterator broke on the 2nd pass


def test_get_child_hiddens_padding():
    child_a = NeuroNode(None, None)
    child_a.h = torch.ones(HIDDEN_DIM)
    parent_two = NeuroNode(None, None, C=[child_a, NeuroNode(None, None)])
    parent_two.C[1].h = 2 * torch.ones(HIDDEN_DIM)
    parent_one = NeuroNode(None, None, C=[NeuroNode(None, None)])
    parent_one.C[0].h = 3 * torch.ones(HIDDEN_DIM)

    batch = BatchNeuroTree([parent_two, parent_one])
    hiddens = batch.get_child_hiddens(torch.zeros(HIDDEN_DIM))
    assert hiddens.shape == (2, 2, HIDDEN_DIM)
    assert torch.all(hiddens[1, 1] == 0)  # padded position


def test_gcn_matches_reference():
    torch.manual_seed(0)
    gcn = GraphConvolutionalLayer(HIDDEN_DIM)
    N = 4
    A = torch.tensor([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0]], dtype=torch.float)
    h = torch.randn(1, N, HIDDEN_DIM)

    out = gcn([A], h)

    # reference: D^-1/2 (A + I) D^-1/2 h with an explicit inverse
    a_tilde = A + torch.eye(N)
    d = torch.diag(a_tilde.sum(1))
    d_inv_sqrt = torch.inverse(d ** 0.5)
    ref = (d_inv_sqrt @ a_tilde @ d_inv_sqrt) @ h[0]
    assert torch.allclose(out[0], ref, atol=1e-5)


def test_gcn_none_adjacency_is_identity_aggregation():
    gcn = GraphConvolutionalLayer(HIDDEN_DIM)
    h = torch.randn(2, 3, HIDDEN_DIM)
    out = gcn([None, None], h)
    # A=I => A~=2I => D=2I => normalized = I => output == input
    assert torch.allclose(out, h, atol=1e-5)


def test_readout_masks_padding():
    readout = MaxpoolReadoutLayer()
    hidden = torch.zeros(2, 3, 4)
    hidden[0, 0] = 1.0
    hidden[0, 1] = 100.0   # padding row for sample 0 (child_count 1): must lose
    hidden[1, 2] = 5.0
    pooled, indices = readout(hidden, [1, 3])
    assert pooled.shape == (2, 1, 4)
    assert torch.all(pooled[0, 0] == 1.0)
    assert torch.all(pooled[1, 0] == 5.0)


def test_egcn_forward_shape():
    egcn = EdgeGraphConvolutionalLayer(HIDDEN_DIM, 2)
    h = torch.randn(2, 3, HIDDEN_DIM)
    edges = [None, [(0, 1, 1.0), (1, 2, 1.0)]]
    out = egcn(edges, h)
    assert out.shape == (2, 3, HIDDEN_DIM)


def _toy_maps(n=6):
    xs = [torch.randn(4) for _ in range(n)]
    ys = list(range(n))
    x_map = {'toy': xs}
    y_map = {'toy': ys}
    maintask_map = {'toy': 'classification'}
    builders = {'toy': lambda x, mt: make_chain_tree(x)}
    return x_map, y_map, maintask_map, builders


def test_dataset_modes():
    x_map, y_map, mt_map, builders = _toy_maps()

    lazy = NeuroDataset(x_map, y_map, mt_map, builders, mode='load2neurotree')
    eager = NeuroDataset(x_map, y_map, mt_map, builders, mode='neurotree2load')
    assert len(lazy) == len(eager) == 6

    tree, y, mt, d = lazy[0]
    assert isinstance(tree, NeuroNode) and mt == 'classification' and d == 'toy'

    # eager mode must NOT re-apply the builder; it returns the prebuilt tree reset
    tree_e, _, _, _ = eager[0]
    tree_e.count = 7
    tree_e2, _, _, _ = eager[0]
    assert tree_e2 is tree_e and tree_e2.count == 0

    try:
        NeuroDataset(x_map, y_map, mt_map, builders, mode='typo')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for unknown mode')


def test_dataloader_end_to_end():
    x_map, y_map, mt_map, builders = _toy_maps()
    dataset = NeuroDataset(x_map, y_map, mt_map, builders)
    loader = createNeuroDataloader(dataset, batch_size=3)
    model, _ = make_model()

    for batch, y, mt, d in loader:
        outputs, h_root, _ = model(batch, list(mt))
        assert h_root.shape == (3, HIDDEN_DIM)
        assert len(outputs) == 3


def test_model_moves_with_to():
    model, _ = make_model()
    model = model.to('cpu')
    assert model.ran.zero_hiddens.device.type == 'cpu'
    assert model.multi_feature_extraction_networks.type_bias.device.type == 'cpu'


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
