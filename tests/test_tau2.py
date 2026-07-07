"""tau2 = tau cells + attention-pool readout (B1 modernization)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aan.models.encoders.readout_attention import AttentionPoolReadout
from test_flat_engine import make_engines, make_mixed_trees, run_engine

HIDDEN_DIM = 16


def _make_tau2_engines():
    recursive, flat = make_engines('tau2')
    flat.readout = recursive.readout  # attention readout has parameters now
    return recursive, flat


def test_attention_pool_masks_padding():
    torch.manual_seed(0)
    readout = AttentionPoolReadout(HIDDEN_DIM).eval()
    h = torch.randn(2, 4, HIDDEN_DIM)
    pooled, indices = readout(h, [2, 4])
    assert pooled.shape == (2, 1, HIDDEN_DIM) and indices is None
    # node 0 has 2 real children: its padded rows must not influence it;
    # node 1 counts all 4 rows, so the same perturbation must reach it
    h2 = h.clone()
    h2[0, 2:] = 99.0
    h2[1, 2:] = 99.0
    pooled2, _ = readout(h2, [2, 4])
    assert torch.allclose(pooled[0], pooled2[0], atol=1e-6)
    assert not torch.allclose(pooled[1], pooled2[1])


def test_outputs_and_grads_match_across_engines():
    recursive, flat = _make_tau2_engines()
    trees = make_mixed_trees()
    weights = torch.randn(len(trees), HIDDEN_DIM)

    out_r = run_engine(recursive, trees)
    (out_r * weights).sum().backward()
    grads_r = {n: p.grad.clone() for n, p in recursive.named_parameters()
               if p.grad is not None}
    for p in recursive.parameters():
        p.grad = None

    out_f = run_engine(flat, trees)
    assert torch.allclose(out_r, out_f, atol=1e-5)
    (out_f * weights).sum().backward()
    grads_f = {n: p.grad.clone() for n, p in recursive.named_parameters()
               if p.grad is not None}
    assert set(grads_r) == set(grads_f)
    for name in grads_r:
        assert torch.allclose(grads_r[name], grads_f[name], atol=1e-5), name
    # the readout itself must be trained
    assert any('readout' in n for n in grads_r)


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
