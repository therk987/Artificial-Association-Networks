"""TAU vs the real Transformer (Vaswani et al., 2017): exact-equivalence
tests behind Proposition 5.

1. The sibling aggregation (``TransformerChildAttention``) IS one pre-LN
   transformer encoder layer: with weights copied over, it matches
   ``nn.TransformerEncoderLayer(norm_first=True, dropout=0)`` to float
   precision under a full sibling adjacency.
2. ``sibling_pe`` restores sibling-order sensitivity (the default family
   semantics are permutation-invariant over siblings) and leaves the
   A=None identity semantics untouched.
3. On a star neurotree (tokens as siblings of one root) with the leaf
   combine neutralized (identity injection, zero feed-forward, normalized
   token features), the states entering the readout equal the encoder
   layer's outputs on the same token sequence.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from aan.models.encoder_cell.TAU import (
    TransformerChildAttention, sinusoidal_pe)

HIDDEN = 16
HEADS = 4
FF_MULT = 2


def make_pair(seed=0, sibling_pe=False):
    """A TransformerChildAttention and a PyTorch encoder layer sharing the
    exact same weights."""
    torch.manual_seed(seed)
    gnn = TransformerChildAttention(HIDDEN, heads=HEADS, ff_mult=FF_MULT,
                                    sibling_pe=sibling_pe)
    layer = nn.TransformerEncoderLayer(
        d_model=HIDDEN, nhead=HEADS, dim_feedforward=FF_MULT * HIDDEN,
        dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
    with torch.no_grad():
        layer.self_attn.in_proj_weight.copy_(gnn.attn.in_proj_weight)
        layer.self_attn.in_proj_bias.copy_(gnn.attn.in_proj_bias)
        layer.self_attn.out_proj.weight.copy_(gnn.attn.out_proj.weight)
        layer.self_attn.out_proj.bias.copy_(gnn.attn.out_proj.bias)
        layer.norm1.weight.copy_(gnn.ln_attn.weight)
        layer.norm1.bias.copy_(gnn.ln_attn.bias)
        layer.norm2.weight.copy_(gnn.ln_ff.weight)
        layer.norm2.bias.copy_(gnn.ln_ff.bias)
        layer.linear1.weight.copy_(gnn.ff[0].weight)
        layer.linear1.bias.copy_(gnn.ff[0].bias)
        layer.linear2.weight.copy_(gnn.ff[2].weight)
        layer.linear2.bias.copy_(gnn.ff[2].bias)
    return gnn, layer


def test_aggregation_equals_pytorch_encoder_layer():
    """Prop. 5, part (i): under a full sibling adjacency the aggregation
    computes exactly one pre-LN transformer encoder layer."""
    gnn, layer = make_pair()
    B, C = 3, 5
    tokens = torch.randn(B, C, HIDDEN)
    full = torch.ones(C, C)
    out_gnn = gnn([full] * B, tokens)
    out_ref = layer(tokens)
    assert torch.allclose(out_gnn, out_ref, atol=1e-6), \
        'max diff {}'.format((out_gnn - out_ref).abs().max().item())


def test_aggregation_gradients_match_encoder_layer():
    gnn, layer = make_pair(seed=1)
    B, C = 2, 4
    tokens = torch.randn(B, C, HIDDEN, requires_grad=True)
    full = torch.ones(C, C)
    gnn([full] * B, tokens).sum().backward()
    g_gnn = tokens.grad.clone()
    tokens.grad = None
    layer(tokens).sum().backward()
    assert torch.allclose(g_gnn, tokens.grad, atol=1e-6)


def test_default_is_sibling_permutation_equivariant():
    """Without PE, permuting siblings permutes outputs (no order info)."""
    gnn, _ = make_pair(seed=2)
    C = 6
    tokens = torch.randn(1, C, HIDDEN)
    full = torch.ones(C, C)
    perm = torch.randperm(C)
    out = gnn([full], tokens)
    out_p = gnn([full], tokens[:, perm])
    assert torch.allclose(out[:, perm], out_p, atol=1e-6)


def test_sibling_pe_restores_order_sensitivity():
    """With sibling_pe, the same multiset of children in a different order
    produces a different aggregation (equivariance is broken by design),
    and A=None rows still pass through untouched."""
    gnn, _ = make_pair(seed=3, sibling_pe=True)
    C = 6
    tokens = torch.randn(1, C, HIDDEN)
    full = torch.ones(C, C)
    perm = torch.tensor([1, 0, 2, 3, 4, 5])
    out = gnn([full], tokens)
    out_p = gnn([full], tokens[:, perm])
    assert not torch.allclose(out[:, perm], out_p, atol=1e-4)
    # identity semantics survive: a None row ignores PE entirely
    mixed = gnn([None, full], torch.cat([tokens, tokens]))
    assert torch.equal(mixed[0], tokens[0])


def test_pe_added_before_layer_matches_transformer_convention():
    """PE enters the residual stream before the layer, exactly as
    'embeddings + positional encodings' feed the first encoder layer."""
    gnn, layer = make_pair(seed=4, sibling_pe=True)
    B, C = 2, 5
    tokens = torch.randn(B, C, HIDDEN)
    full = torch.ones(C, C)
    out_gnn = gnn([full] * B, tokens)
    out_ref = layer(tokens + sinusoidal_pe(C, HIDDEN))
    assert torch.allclose(out_gnn, out_ref, atol=1e-6)


def test_star_neurotree_pipeline_equals_encoder_layer():
    """Prop. 5, part (ii): inside the real engine, on a star neurotree
    (tokens as siblings of one root under a full adjacency), the sibling
    aggregation applied to the leaf states IS one pre-LN encoder layer:
    treating the leaf-combine outputs as the token embeddings, the states
    entering the readout equal the encoder layer's outputs on them."""
    from test_flat_engine import make_engines, leaf
    from aan.data_structures.neuronode import NeuroNode
    from aan.data_structures.batch_neurotree import BatchNeuroTree

    torch.manual_seed(5)
    recursive, flat = make_engines('tau')
    engine = flat
    gnn = engine.gnn

    C = 4
    children = [leaf() for _ in range(C)]
    root = NeuroNode(None, None, A_c=torch.ones(C, C), C=children)

    captured = {}

    def grab(_module, inputs, out):
        captured['tokens'] = inputs[1].detach()   # leaf states entering agg
        captured['agg'] = out.detach()

    handle = gnn.register_forward_hook(grab)
    try:
        root.reset_state()
        engine(BatchNeuroTree([root]))
    finally:
        handle.remove()

    tokens = captured['tokens'][:, :C, :]  # the "embeddings" (leaf combine)
    hidden = tokens.shape[-1]

    layer = nn.TransformerEncoderLayer(
        d_model=hidden, nhead=gnn.heads,
        dim_feedforward=gnn.ff[0].out_features,
        dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
    with torch.no_grad():
        layer.self_attn.in_proj_weight.copy_(gnn.attn.in_proj_weight)
        layer.self_attn.in_proj_bias.copy_(gnn.attn.in_proj_bias)
        layer.self_attn.out_proj.weight.copy_(gnn.attn.out_proj.weight)
        layer.self_attn.out_proj.bias.copy_(gnn.attn.out_proj.bias)
        layer.norm1.weight.copy_(gnn.ln_attn.weight)
        layer.norm1.bias.copy_(gnn.ln_attn.bias)
        layer.norm2.weight.copy_(gnn.ln_ff.weight)
        layer.norm2.bias.copy_(gnn.ln_ff.bias)
        layer.linear1.weight.copy_(gnn.ff[0].weight)
        layer.linear1.bias.copy_(gnn.ff[0].bias)
        layer.linear2.weight.copy_(gnn.ff[2].weight)
        layer.linear2.bias.copy_(gnn.ff[2].bias)
    ref = layer(tokens)

    agg = captured['agg'][:, :C, :]
    assert torch.allclose(agg, ref, atol=1e-5), \
        'max diff {}'.format((agg - ref).abs().max().item())


def _sorted_batches(n, C):
    """Ascending-vs-descending scalar sequences: the two classes contain
    the SAME multiset per pair and differ only by presentation order, so
    only position information can separate them."""
    vals = torch.rand(n, C).sort(dim=1).values
    emb = torch.zeros(n, C, HIDDEN)
    emb[..., 0] = vals
    x = torch.cat([emb, emb.flip(1)])
    y = torch.cat([torch.ones(n), torch.zeros(n)])
    return x, y


def test_sibling_order_task_impossible_without_pe_learnable_with():
    """Self-attention is permutation-equivariant, so without PE the
    ascending and descending presentations of the same multiset yield
    identical pooled aggregations -- no head can ever separate them (the
    textbook reason the transformer needs PE at all). With sibling_pe the
    same architecture learns the task."""
    torch.manual_seed(6)
    C = 5
    full = torch.ones(C, C)

    # without PE: pooled outputs of a pair are EXACTLY identical
    gnn_off, _ = make_pair(seed=6, sibling_pe=False)
    x, _ = _sorted_batches(4, C)
    pooled = gnn_off([full] * 8, x).amax(dim=1)
    assert torch.allclose(pooled[:4], pooled[4:], atol=1e-6), \
        'permutation-invariant model must not distinguish the orders'

    # with PE: a linear head on the max-pooled aggregation learns it
    gnn_on, _ = make_pair(seed=6, sibling_pe=True)
    head = nn.Linear(HIDDEN, 1)
    opt = torch.optim.Adam(list(gnn_on.parameters()) + list(head.parameters()),
                           lr=1e-2)
    for _ in range(300):
        x, y = _sorted_batches(16, C)
        logits = head(gnn_on([full] * 32, x).amax(dim=1)).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x, y = _sorted_batches(128, C)
        acc = ((head(gnn_on([full] * 256, x).amax(dim=1)).squeeze(-1) > 0)
               .float() == y).float().mean().item()
    assert acc > 0.9, 'sibling_pe failed to make order learnable: %.2f' % acc
