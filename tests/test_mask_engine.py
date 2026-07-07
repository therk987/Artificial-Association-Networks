"""S0 mask engine tests.

(a) MaskedDFCEngine must be output- AND gradient-equivalent to the flat/
    recursive tau engines (same modules, compiler-built masks).
(b) AncestorMaskTransformer: mask semantics — descendants(+self) only,
    no cross-tree leakage in a packed batch, level ids correct, and the
    structured/flat control arms share identical block code.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.encoders.mask_engine import (MaskedDFCEngine,
                                             AncestorMaskTransformer,
                                             compile_ancestor_masks)
from aan.models.feature_encoders.multiencoder_connector import MultiExtractionConnector
from test_flat_engine import (ToyEncoder, make_engines, make_mixed_trees,
                              run_engine, leaf, FEATURE_DIM, HIDDEN_DIM)


def make_mask_engine():
    torch.manual_seed(0)
    fe = MultiExtractionConnector(FEATURE_DIM, {'toy': ToyEncoder()})
    return MaskedDFCEngine(FEATURE_DIM, HIDDEN_DIM, fe, None, version='tau')


def test_outputs_match_flat_tau():
    recursive, flat = make_engines('tau')
    mask_engine = make_mask_engine()
    mask_engine.rnn = flat.rnn
    mask_engine.gnn = flat.gnn
    mask_engine.feature_extraction_networks = flat.feature_extraction_networks

    trees = make_mixed_trees()
    out_f = run_engine(flat, trees)
    out_m = run_engine(mask_engine, trees)
    assert torch.allclose(out_f, out_m, atol=1e-5), \
        'mask vs flat: max diff {}'.format((out_f - out_m).abs().max().item())


def test_gradients_match_flat_tau():
    _, flat = make_engines('tau')
    mask_engine = make_mask_engine()
    mask_engine.rnn = flat.rnn
    mask_engine.gnn = flat.gnn
    mask_engine.feature_extraction_networks = flat.feature_extraction_networks

    trees = make_mixed_trees()
    weights = torch.randn(len(trees), HIDDEN_DIM)  # non-constant loss (final LN)

    out_f = run_engine(flat, trees)
    (out_f * weights).sum().backward()
    grads_f = {n: p.grad.clone() for n, p in flat.named_parameters() if p.grad is not None}
    for p in flat.parameters():
        p.grad = None

    out_m = run_engine(mask_engine, trees)
    (out_m * weights).sum().backward()
    grads_m = {n: p.grad.clone() for n, p in flat.named_parameters() if p.grad is not None}

    assert set(grads_f) == set(grads_m)
    for name in grads_f:
        assert torch.allclose(grads_f[name], grads_m[name], atol=1e-5), \
            '{}: max diff {}'.format(name, (grads_f[name] - grads_m[name]).abs().max().item())


def test_mask_engine_rejects_non_tau():
    fe = MultiExtractionConnector(FEATURE_DIM, {'toy': ToyEncoder()})
    try:
        MaskedDFCEngine(FEATURE_DIM, HIDDEN_DIM, fe, None, version='gaau')
    except ValueError:
        return
    raise AssertionError('expected ValueError for non-tau version')


def _two_chains():
    a_leaf, b_leaf = leaf(), leaf()
    tree_a = NeuroNode(None, None, C=[NeuroNode(None, None, C=[a_leaf])])
    tree_b = NeuroNode(None, None, C=[NeuroNode(None, None, C=[b_leaf])])
    return tree_a, tree_b, a_leaf, b_leaf


def test_ancestor_mask_semantics():
    tree_a, tree_b, a_leaf, _ = _two_chains()
    nodes, allowed, level_ids, roots_pos = compile_ancestor_masks([tree_a, tree_b])
    pos = {id(n): i for i, n in enumerate(nodes)}
    ra, rb = roots_pos
    assert allowed[ra, pos[id(a_leaf)]]           # root sees its leaf
    assert not allowed[pos[id(a_leaf)], ra]       # leaf does not see its root
    # no cross-tree edges
    for i, ni in enumerate(nodes):
        in_a = allowed[ra, i] or i == ra
        in_b = allowed[rb, i] or i == rb
        assert not (in_a and in_b)
    assert level_ids[pos[id(a_leaf)]] == 0 and level_ids[ra] == 2


def test_ancestor_transformer_isolation_and_grads():
    torch.manual_seed(0)
    fe = MultiExtractionConnector(FEATURE_DIM, {'toy': ToyEncoder()})
    model = AncestorMaskTransformer(FEATURE_DIM, HIDDEN_DIM, fe,
                                    n_layers=2, heads=4)
    tree_a, tree_b, _, _ = _two_chains()

    for t in (tree_a, tree_b):
        t.reset_state()
    out_pair = model(BatchNeuroTree([tree_a, tree_b]))
    for t in (tree_a, tree_b):
        t.reset_state()
    out_solo = model(BatchNeuroTree([tree_a]))
    # packing tree_b next to tree_a must not change tree_a's root
    assert torch.allclose(out_pair[0], out_solo[0], atol=1e-6)

    (out_pair * torch.randn_like(out_pair)).sum().backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g.grad).all() for g in
                         [p for p in model.parameters() if p.grad is not None])


def test_ancestor_transformer_weight_shared_and_flat_arm():
    torch.manual_seed(0)
    fe = MultiExtractionConnector(FEATURE_DIM, {'toy': ToyEncoder()})
    shared = AncestorMaskTransformer(FEATURE_DIM, HIDDEN_DIM, fe,
                                     n_layers=3, weight_shared=True)
    assert len(shared.blocks) == 1
    flat_arm = AncestorMaskTransformer(FEATURE_DIM, HIDDEN_DIM, fe,
                                       n_layers=2, structured=False)
    trees = make_mixed_trees()
    for t in trees:
        t.reset_state()
    out = shared(BatchNeuroTree(trees))
    assert out.shape == (len(trees), HIDDEN_DIM) and torch.isfinite(out).all()
    for t in trees:
        t.reset_state()
    out2 = flat_arm(BatchNeuroTree(trees))
    assert out2.shape == (len(trees), HIDDEN_DIM) and torch.isfinite(out2).all()


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
