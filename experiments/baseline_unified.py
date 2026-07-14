"""Unified NON-tree baseline for the AAN low-resource positive-transfer study.

Reviewer question (paper revision): the AAN shows positive transfer when IMDB
is trained jointly with MNIST + SpeechCommands, and the gain grows as the IMDB
training set shrinks. Is that gain produced by the AAN's tree STRUCTURE (the
weight-tied association cell run over the neurotree), or would ANY shared
encoder over the SAME per-domain features give it?

This runner is the parameter-matched control that answers it. It keeps
everything the AAN uses EXCEPT the tree:

  * the SAME per-domain feature extractors as ``run.py`` -- reused
    object-for-object from the PARTS registry, with identical leaf
    preprocessing (e.g. MNIST's ``/255``), each mapping a raw sample to a
    128-d vector;
  * a SHARED trunk applied to every domain in place of the AAN's association
    cell -- an MLP (primary) whose depth is auto-tuned to match the AAN cell's
    parameter count, or a 1-layer Transformer over a single-token+CLS sequence
    (``--trunk transformer``);
  * a single joint classification head (47-way = 10 MNIST + 35 SpeechCommands
    + 2 IMDB) using run.py's joint-label offset scheme.

Reused from ``run.py`` UNCHANGED (imported, never modified): the PARTS
registry, ``set_seed``, ``build_dataset`` (hence the ``domain@frac``
low-resource subsample syntax AND its per-seed pairing so the alone and joint
arms see the same subset), ``evaluate`` and ``stack_outputs``. The per-seed /
per-domain test-accuracy CSV is written in run.py's exact format so the paired
low-resource table is computed the same way for the baseline as for the AAN.

Smoke:
  python experiments/baseline_unified.py --dataset imdb@0.01 --seeds 1234 \
      --epochs 3 --out results/smoke_baseline.csv
  python experiments/baseline_unified.py --dataset mnist,speechcommands,imdb@0.01 \
      --seeds 1234 --epochs 2 --limit 2000
"""
import argparse
import csv
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from aan.config.option import device as default_device
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.data_structures.neurodataloader import createNeuroDataloader
from aan.models.encoders.recursive_encoder import build_cells, build_readout

# Reuse run.py's registry, seed util, subsample syntax, dataset builder and
# evaluation loop UNCHANGED -- imported, never modified (per the task's
# "do not modify run.py" constraint). Importing run.py only defines symbols;
# its CLI lives under ``if __name__ == '__main__'`` so nothing is executed.
from experiments.run import (  # noqa: E402
    PARTS, set_seed, build_dataset, evaluate, stack_outputs,
)

import torch.nn.functional as F  # noqa: E402


# ---------------------------------------------------------------------------
# Feature side: reuse the EXACT per-domain encoders on the EXACT leaf inputs
# ---------------------------------------------------------------------------

def domain_leaf(root):
    """Descend a run.py-built neurotree to its single feature leaf.

    run.py's builders (``mnist_image2neurotree`` / ``sound2neurotree`` /
    ``text2neurotree``) wrap each raw sample in a chain of empty association
    nodes above ONE domain leaf:

        image: root(None) -> mid(None) -> leaf('image', x = data/255)
        sound: root(None) -> leaf('sound', x = waveform)
        text:  root(None) -> leaf('text',  x = token-id tensor)

    The baseline throws away that structure and reads the leaf, which already
    carries exactly the preprocessed ``x`` (and domain ``t_d``) the AAN's
    feature encoder would consume -- so preprocessing is replicated for free by
    reusing the builders through ``build_dataset``.
    """
    node = root
    while node.C:
        node = node.C[0]
    return node


# ---------------------------------------------------------------------------
# Shared trunks (the parameter-matched replacements for the AAN cell)
# ---------------------------------------------------------------------------

_ACTIVATIONS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}


class MLPTrunk(nn.Module):
    """Plain shared MLP: ``input_dim -> hidden`` then ``depth-1`` x
    ``hidden -> hidden``, with the family activation and a (parameter-free)
    LayerNorm per layer mirroring the GRU cell's ``F.layer_norm`` on its
    candidate state. Depth is chosen by the caller to match the AAN cell's
    parameter count."""

    def __init__(self, input_dim, hidden_dim, depth, activation='tanh', norm=True):
        super().__init__()
        act_cls = _ACTIVATIONS[activation]
        layers = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            if norm:
                # elementwise_affine=False -> no parameters, keeps the audit clean
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act_cls())
            d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransformerTrunk(nn.Module):
    """1-layer Transformer encoder over a length-1 feature sequence + CLS.

    Each sample's per-domain 128-d feature (plus the domain one-hot) is
    projected to a single token; a learned CLS token is prepended; a single
    ``TransformerEncoderLayer`` mixes the two, and the CLS output is read out.
    ``dim_feedforward`` is auto-tuned by the caller toward the AAN cell's
    parameter count."""

    def __init__(self, input_dim, hidden_dim, dim_feedforward, nhead=4):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, activation='gelu', batch_first=True,
        )

    def forward(self, x):
        tok = self.in_proj(x).unsqueeze(1)            # (B, 1, H)
        cls = self.cls.expand(x.shape[0], -1, -1)     # (B, 1, H)
        seq = torch.cat([cls, tok], dim=1)            # (B, 2, H)
        out = self.layer(seq)
        return out[:, 0]                              # CLS row


class GatedTrunk(nn.Module):
    """The AAN's own gated cell with the tree removed (the gate-vs-tree
    deconfound). The identical ``GatedRecurrentUnit`` block the GAU/GAAU
    cell uses (weight-tied) is applied ``steps`` times to the flat
    feature+one-hot input from a zero state --- a depth-``steps`` chain
    whose only input is the sample itself. Same gate, same parameter count
    up to the sibling-attention term, no tree/sibling/multi-parent
    machinery."""

    def __init__(self, input_dim, hidden_dim, steps=2):
        super().__init__()
        from aan.models.encoder_cell.GRU import GatedRecurrentUnit
        self.cell = GatedRecurrentUnit(input_dim, hidden_dim)
        self.steps = steps
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h = x.new_zeros(x.shape[0], 1, self.hidden_dim)
        xs = x.unsqueeze(1)
        for _ in range(self.steps):
            h = self.cell(xs, h)
        return h.squeeze(1)


# ---------------------------------------------------------------------------
# Parameter-count auditing against the real AAN association cell
# ---------------------------------------------------------------------------

def count_params(module):
    return sum(p.numel() for p in module.parameters())


def aan_cell_params(version, input_dim, hidden_dim, type_count):
    """Build the REAL AAN association cell (via run.py's own
    ``build_cells``/``build_readout``) and count it -- this is the
    weight-tied 128->128 block the shared trunk replaces, applied at every
    neurotree level. Its rnn consumes ``input_dim + type_count`` (feature +
    domain one-hot bias), so the count depends on how many domains are joined.
    Returns (total, breakdown_dict)."""
    input_dim_with_bias = input_dim + type_count
    rnn, gnn = build_cells(version, input_dim_with_bias, hidden_dim)
    readout = build_readout(version, hidden_dim)
    breakdown = {
        'rnn ({})'.format(type(rnn).__name__): count_params(rnn),
        'gnn ({})'.format(type(gnn).__name__): count_params(gnn),
        'readout ({})'.format(type(readout).__name__): count_params(readout),
    }
    return sum(breakdown.values()), breakdown


def _mlp_params(depth, input_dim, hidden_dim):
    p = input_dim * hidden_dim + hidden_dim          # first layer
    p += (depth - 1) * (hidden_dim * hidden_dim + hidden_dim)
    return p


def choose_mlp_depth(target, input_dim, hidden_dim, max_depth=32):
    """Pick the MLP depth whose parameter count is closest to ``target``."""
    best_depth, best_diff = 1, None
    for depth in range(1, max_depth + 1):
        diff = abs(_mlp_params(depth, input_dim, hidden_dim) - target)
        if best_diff is None or diff < best_diff:
            best_depth, best_diff = depth, diff
    return best_depth


def choose_dim_feedforward(target, input_dim, hidden_dim, nhead):
    """Pick ``dim_feedforward`` so the 1-layer Transformer trunk lands near
    ``target``. Parameter count is exactly affine in dim_feedforward, so two
    reference builds recover the line and we solve for the target."""
    p1 = count_params(TransformerTrunk(input_dim, hidden_dim, 1, nhead))
    p2 = count_params(TransformerTrunk(input_dim, hidden_dim, 2, nhead))
    slope = p2 - p1
    base = p1 - slope  # params at dim_feedforward == 0
    return max(1, round((target - base) / max(slope, 1)))


# ---------------------------------------------------------------------------
# The flat, non-tree model
# ---------------------------------------------------------------------------

class UnifiedBaseline(nn.Module):
    """Flat control: per-domain feature encoder -> shared trunk -> joint head.

    No neurotree, no association cell. For each sample the leaf's domain
    encoder produces a 128-d vector; the domain one-hot bias is appended
    (mirroring the AAN cell's ``input_dim_with_bias`` input); a single shared
    trunk maps it to a 128-d state; and one joint linear head produces the
    47-way (or per-config) logits.

    Exposes the same ``forward(batchNeuroTree, tasks) -> (outputs, h, tree)``
    signature the AAN uses, so run.py's ``evaluate`` and training loop drive it
    unchanged.
    """

    def __init__(self, feature_encoders, class_count, input_dim=128,
                 hidden_dim=128, trunk='mlp', trunk_depth=None,
                 dim_feedforward=None, nhead=4, activation='tanh',
                 norm=True, version='gaau'):
        super().__init__()
        # The SAME encoder objects build_dataset handed us (identical weights,
        # identical preprocessing); registering them makes their params train.
        self.encoders = nn.ModuleDict(feature_encoders)
        self.type_keys = list(feature_encoders.keys())
        self.type_index = {k: i for i, k in enumerate(self.type_keys)}
        self.type_count = len(self.type_keys)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_dim_with_bias = input_dim + self.type_count
        self.register_buffer('type_bias', torch.eye(self.type_count))

        # --- parameter audit against the real AAN cell -------------------
        cell_total, cell_breakdown = aan_cell_params(
            version, input_dim, hidden_dim, self.type_count)
        self.aan_cell_total = cell_total

        self.trunk_kind = trunk
        if trunk == 'mlp':
            depth = trunk_depth or choose_mlp_depth(
                cell_total, self.input_dim_with_bias, hidden_dim)
            self.trunk_depth = depth
            self.trunk = MLPTrunk(self.input_dim_with_bias, hidden_dim, depth,
                                  activation=activation, norm=norm)
        elif trunk == 'transformer':
            ff = dim_feedforward or choose_dim_feedforward(
                cell_total, self.input_dim_with_bias, hidden_dim, nhead)
            self.trunk_dim_feedforward = ff
            self.trunk = TransformerTrunk(self.input_dim_with_bias, hidden_dim,
                                          ff, nhead=nhead)
        elif trunk == 'gru':
            self.trunk_depth = trunk_depth or 2
            self.trunk = GatedTrunk(self.input_dim_with_bias, hidden_dim,
                                    steps=self.trunk_depth)
        else:
            raise ValueError("unknown trunk: {} (expected 'mlp', 'transformer' or 'gru')".format(trunk))

        self.head = nn.Linear(hidden_dim, class_count)

        self._print_param_audit(version, cell_total, cell_breakdown)

    def _print_param_audit(self, version, cell_total, cell_breakdown):
        trunk_total = count_params(self.trunk)
        ratio = trunk_total / max(cell_total, 1)
        if self.trunk_kind == 'mlp':
            desc = 'MLP depth={} act={}'.format(self.trunk_depth,
                                                type(self.trunk.net[-1]).__name__)
        elif self.trunk_kind == 'gru':
            desc = 'GatedTrunk (AAN GRU cell, tree removed) steps={}'.format(
                self.trunk_depth)
        else:
            desc = 'Transformer 1-layer dim_feedforward={}'.format(
                self.trunk_dim_feedforward)
        print('[param-audit] version={} type_count={} input_dim_with_bias={}'.format(
            version, self.type_count, self.input_dim_with_bias), flush=True)
        print('  AAN association cell it replaces : {:>9,d} params'.format(cell_total),
              flush=True)
        for name, n in cell_breakdown.items():
            print('      - {:<28s} {:>9,d}'.format(name, n), flush=True)
        print('  Baseline shared trunk ({:<28s}: {:>9,d} params'.format(desc + ')', trunk_total),
              flush=True)
        within = abs(ratio - 1.0) <= 0.20
        print('  ratio trunk/cell = {:.3f}   (within +/-20%: {})'.format(
            ratio, 'OK' if within else 'NO'), flush=True)
        print('  [context] shared feature encoders: {:,d} params | joint head: {:,d} params'.format(
            sum(count_params(m) for m in self.encoders.values()),
            count_params(self.head)), flush=True)

    def forward(self, batchNeuroTree, tasks=None, node_level=False):
        device = self.head.weight.device
        leaves = [domain_leaf(root) for root in batchNeuroTree.nodes]
        domains = [leaf.t_d for leaf in leaves]
        batch_size = len(leaves)

        features = torch.zeros(batch_size, self.hidden_dim, device=device)
        # group by domain, run each domain encoder ONCE (as the AAN's
        # MultiExtractionConnector does), then scatter back to batch order.
        groups = defaultdict(list)
        for i, d in enumerate(domains):
            groups[d].append(i)
        for d, idxs in groups.items():
            if d not in self.encoders:
                raise KeyError("no feature encoder for domain '{}' (have {}); the "
                               "unified baseline supports flat single-leaf domains "
                               "(image/sound/text)".format(d, self.type_keys))
            sub = BatchNeuroTree([leaves[i] for i in idxs])
            feat = self.encoders[d](sub)
            if feat.dim() == 3:      # (n, 1, H) -> (n, H) if an encoder unsqueezes
                feat = feat.squeeze(1)
            index = torch.as_tensor(idxs, device=device)
            features.index_copy_(0, index, feat.to(device))

        bias_idx = torch.as_tensor([self.type_index[d] for d in domains], device=device)
        x = torch.cat([features, self.type_bias[bias_idx]], dim=-1)
        hidden = self.trunk(x)
        logits = self.head(hidden)
        return logits, hidden, batchNeuroTree


# ---------------------------------------------------------------------------
# Training / evaluation (mirrors run.py's run_one_seed; reuses its evaluate)
# ---------------------------------------------------------------------------

def run_one_seed(args, seed, device):
    set_seed(seed)
    # subsample_seed=seed => identical low-resource subset to run.py, so the
    # baseline's alone/joint arms are paired by seed on the SAME subset.
    train_ds, valid_ds, test_ds, feature_encoders, class_count = build_dataset(
        args.dataset, limit=args.limit, subsample_seed=seed)

    workers = dict(num_workers=args.num_workers)
    train_loader = createNeuroDataloader(train_ds, batch_size=args.batch_size, shuffle=True, **workers)
    valid_loader = createNeuroDataloader(valid_ds, batch_size=args.batch_size, **workers)
    test_loader = createNeuroDataloader(test_ds, batch_size=args.batch_size, **workers)

    model = UnifiedBaseline(
        feature_encoders, class_count,
        input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        trunk=args.trunk, trunk_depth=args.trunk_depth,
        dim_feedforward=args.dim_feedforward, nhead=args.nhead,
        activation=args.activation, norm=not args.no_norm,
        version=args.version,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {'valid_acc': 0.0, 'test_acc': 0.0, 'epoch': -1, 'per_domain': {}}
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()
        for batch, y, mt, d in train_loader:
            outputs, _, _ = model(batch, list(mt))
            logits = stack_outputs(outputs)
            targets = torch.stack(y, dim=0).to(device, dtype=torch.long).view(-1)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        valid_acc, _ = evaluate(model, valid_loader, device)
        if valid_acc >= best['valid_acc']:  # model selection on validation accuracy
            test_acc, per_domain = evaluate(model, test_loader, device)
            best.update(valid_acc=valid_acc, test_acc=test_acc,
                        epoch=epoch, per_domain=per_domain)
        print('  seed {} epoch {}: loss {:.4f} valid {:.4f} ({:.1f}s)'.format(
            seed, epoch, epoch_loss / max(len(train_loader), 1), valid_acc,
            time.perf_counter() - t0), flush=True)

    return {'seed': seed, 'valid_acc': best['valid_acc'], 'test_acc': best['test_acc'],
            'epoch': best['epoch'], 'per_domain': best['per_domain']}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='imdb',
                        help='domain or comma list, with run.py @frac subsample '
                             '(e.g. imdb, imdb@0.1, imdb@0.01, '
                             'mnist,speechcommands,imdb@0.01)')
    parser.add_argument('--trunk', default='mlp',
                        choices=['mlp', 'transformer', 'gru'],
                        help='shared trunk replacing the AAN cell (MLP is '
                             'primary; gru = the AAN gate without the tree)')
    parser.add_argument('--trunk-depth', type=int, default=None,
                        help='MLP depth override (default: auto-match AAN cell params)')
    parser.add_argument('--dim-feedforward', type=int, default=None,
                        help='transformer FFN width override (default: auto-match)')
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--activation', default='tanh', choices=list(_ACTIVATIONS))
    parser.add_argument('--no-norm', action='store_true',
                        help='drop the (parameter-free) per-layer LayerNorm in the MLP trunk')
    parser.add_argument('--version', default='gaau',
                        help='AAN cell version whose param count the trunk matches')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 42, 7, 2024, 31337])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None,
                        help='subsample each domain for a quick smoke run')
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--out', default=None, help='CSV path for per-seed results')
    args = parser.parse_args()

    print('BASELINE unified (non-tree, param-matched) '
          'dataset={} trunk={} device={} seeds={}'.format(
              args.dataset, args.trunk, args.device, args.seeds), flush=True)

    results = []
    for seed in args.seeds:
        results.append(run_one_seed(args, seed, args.device))
        r = results[-1]
        domains = '  '.join('{} {:.4f}'.format(k, v) for k, v in r['per_domain'].items())
        print('seed {} -> test {:.4f} (valid {:.4f}, epoch {})  [{}]'.format(
            r['seed'], r['test_acc'], r['valid_acc'], r['epoch'], domains), flush=True)

    accs = [r['test_acc'] for r in results]
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print('== {} baseline/{}: test acc {:.4f} +/- {:.4f} over {} seeds =='.format(
        args.dataset, args.trunk, mean, std, len(accs)), flush=True)
    domains = sorted(results[0]['per_domain'])
    if len(domains) > 1:
        for domain in domains:
            vals = [r['per_domain'][domain] for r in results]
            dstd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print('   {}: {:.4f} +/- {:.4f}'.format(
                domain, statistics.mean(vals), dstd), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            fields = ['seed', 'valid_acc', 'test_acc', 'epoch'] + domains
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                row = {k: r[k] for k in ('seed', 'valid_acc', 'test_acc', 'epoch')}
                row.update(r['per_domain'])
                writer.writerow(row)
        print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
