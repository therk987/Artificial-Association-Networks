"""TAU (transformer cell) tests: engine equivalence, family semantics,
deep-chain stability (the E3 failure mode the residual stream targets)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.encoder_cell.TAU import TransformerChildAttention
from test_flat_engine import make_engines, make_mixed_trees, run_engine, leaf

HIDDEN_DIM = 16


def test_outputs_match_tau():
    recursive, flat = make_engines('tau')
    trees = make_mixed_trees()
    out_r = run_engine(recursive, trees)
    out_f = run_engine(flat, trees)
    assert torch.allclose(out_r, out_f, atol=1e-5), \
        'tau: max diff {}'.format((out_r - out_f).abs().max().item())


def test_gradients_match_tau():
    recursive, flat = make_engines('tau')
    trees = make_mixed_trees()

    out_r = run_engine(recursive, trees)
    out_r.sum().backward()
    grads_r = {n: p.grad.clone() for n, p in recursive.named_parameters() if p.grad is not None}
    for p in recursive.parameters():
        p.grad = None

    out_f = run_engine(flat, trees)
    out_f.sum().backward()
    grads_f = {n: p.grad.clone() for n, p in recursive.named_parameters() if p.grad is not None}

    assert set(grads_r) == set(grads_f)
    for name in grads_r:
        assert torch.allclose(grads_r[name], grads_f[name], atol=1e-5), \
            '{}: max diff {}'.format(name, (grads_r[name] - grads_f[name]).abs().max().item())


def test_none_adjacency_is_identity():
    """Family semantics: no sibling relationships -> no mixing."""
    gnn = TransformerChildAttention(HIDDEN_DIM)
    Wh = torch.randn(3, 4, HIDDEN_DIM)
    out = gnn([None, None, None], Wh)
    assert torch.equal(out, Wh)


def test_masked_attention_respects_adjacency_and_padding():
    """A real row only mixes children allowed by A + I; gradients stay finite
    even with padded rows (the NaN-poisoning case the self-loop guards)."""
    torch.manual_seed(0)
    gnn = TransformerChildAttention(HIDDEN_DIM)
    # node 0: 3 real children with a path adjacency; node 1: 2 real + padding
    A0 = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    A1 = torch.tensor([[0., 1.], [1., 0.]])
    Wh = torch.randn(2, 3, HIDDEN_DIM, requires_grad=True)
    out = gnn([A0, A1], Wh)
    assert out.shape == Wh.shape
    assert torch.isfinite(out).all()
    out[:, :2].sum().backward()  # use only non-padded rows, like the readout
    assert torch.isfinite(Wh.grad).all()
    for p in gnn.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_real_rows_independent_of_padding():
    """A node's real children must get the same result no matter how much
    padding the engine's grouping added (recursive vs flat group widths)."""
    torch.manual_seed(0)
    gnn = TransformerChildAttention(HIDDEN_DIM).eval()
    A = torch.tensor([[0., 1.], [1., 0.]])
    children = torch.randn(1, 2, HIDDEN_DIM)

    with torch.no_grad():
        narrow = gnn([A], children)
        wide = gnn([A], torch.cat([children, torch.zeros(1, 3, HIDDEN_DIM)], dim=1))

    assert torch.allclose(narrow[0, :2], wide[0, :2], atol=1e-6)


def test_none_rows_pass_through_in_mixed_batch():
    """Per-node None semantics inside a mixed group: the None node's children
    are returned exactly as-is while connected nodes ARE mixed (both engines
    rely on this for their output equivalence)."""
    torch.manual_seed(0)
    gnn = TransformerChildAttention(HIDDEN_DIM).eval()
    A = torch.tensor([[0., 1.], [1., 0.]])
    children = torch.randn(2, 2, HIDDEN_DIM)

    with torch.no_grad():
        out = gnn([A, None], children)
    assert torch.equal(out[1], children[1])
    assert not torch.allclose(out[0], children[0])


def test_explicit_identity_differs_from_none():
    """Documented caveat: unlike GCN/GAT, an explicit A=I is NOT equivalent
    to None for TAU (value/FF transforms still apply)."""
    torch.manual_seed(0)
    gnn = TransformerChildAttention(HIDDEN_DIM).eval()
    children = torch.randn(1, 2, HIDDEN_DIM)

    with torch.no_grad():
        out_none = gnn([None], children)
        out_eye = gnn([torch.eye(2)], children)

    assert torch.equal(out_none, children)
    assert not torch.allclose(out_eye, children)


def test_deep_chain_stability():
    """81-level chain (E3 regime): forward and gradients stay finite."""
    torch.manual_seed(0)
    recursive, flat = make_engines('tau')
    node = leaf()
    for _ in range(80):
        node = NeuroNode(None, None, C=[node])
    out = flat(BatchNeuroTree([node]))
    assert torch.isfinite(out).all()
    out.sum().backward()
    for name, p in flat.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


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
