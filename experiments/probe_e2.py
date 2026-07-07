"""E2 — does the root vector really contain ALL domains' information?

The paper's Experiment 2 showed classification accuracy is preserved when a
tree carries three domains at once; this script tests the claim directly.
Combined trees (image + sound + text under one root, primary domain rotating)
train the model as in the paper; then the model is FROZEN and linear probes
are fit on held-out root vectors to decode

    - the image label   (10-way)   - the sound label (35-way)
    - the text label    (2-way)    - which domain was primary (3-way)

All probes far above chance = h_root embeds every domain simultaneously,
not just the primary one. Probes are linear so the information must be
linearly present, not merely recoverable by a deep decoder.

    python experiments/probe_e2.py --datasets mnist,speechcommands,imdb \
        --epochs 5 --seeds 1234 42 7
"""
import argparse
import csv
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from aan.config.option import device as default_device
from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks

from run import PARTS, ClassificationHead, set_seed, stack_outputs


class CombinedTreeDataset(Dataset):
    """Trees with one sample from EVERY domain under a shared root.

    The joint-classification target is the PRIMARY domain's label (rotating
    round-robin), exactly the paper's Experiment-2 construction; the other
    domains ride along as context the probe phase will try to read back."""

    def __init__(self, parts, split, size, seed):
        rng = random.Random(seed)
        self.parts = parts
        self.entries = []
        pools = []
        for p in parts:
            xs, ys = p['splits'][split]
            pools.append(len(xs))
        for i in range(size):
            picks = tuple(rng.randrange(n) for n in pools)
            self.entries.append((picks, i % len(parts)))

        self.offsets = {}
        total = 0
        for p in parts:
            self.offsets[p['domain']] = total
            total += p['classes']
        self.total_classes = total
        self.split = split

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        picks, primary = self.entries[idx]
        subtrees, labels = [], []
        for p, k in zip(self.parts, picks):
            xs, ys = p['splits'][self.split]
            subtrees.append(p['builder'](xs[k], 'classification'))
            labels.append(int(ys[k]))
        tree = NeuroNode(None, None, C=subtrees)
        primary_part = self.parts[primary]
        joint_y = self.offsets[primary_part['domain']] + labels[primary]
        return tree, joint_y, labels, primary


def collate(batch):
    trees, joint, labels, primary = zip(*batch)
    return (list(trees), torch.tensor(joint, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(primary, dtype=torch.long))


def encode(model, trees, device):
    for t in trees:
        t.reset_state()
    h_root, _ = model.propagate(BatchNeuroTree(list(trees)))
    return h_root


def fit_probe(train_x, train_y, test_x, test_y, classes, device, steps=400):
    probe = torch.nn.Linear(train_x.shape[1], classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        loss = F.cross_entropy(probe(train_x), train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((probe(test_x).argmax(-1) == test_y).float().mean())
    return acc


def run_seed(args, seed, device, rows):
    set_seed(seed)
    parts = [PARTS[name.strip()](args.limit)
             for name in args.datasets.split(',') if name.strip()]
    train_ds = CombinedTreeDataset(parts, 'train', args.train_size, seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=args.num_workers)

    encoders = {}
    for p in parts:
        encoders.update(p.get('encoders') or {p['domain']: p['encoder']})
    model = ArtificialAssociationNeuralNetworks(
        args.input_dim, args.hidden_dim, encoders, {}, {},
        {'classification': ClassificationHead(args.hidden_dim, train_ds.total_classes)},
        version=args.version, engine=args.engine).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for trees, joint_y, _labels, _primary in loader:
            for t in trees:
                t.reset_state()
            outputs, _, _ = model(BatchNeuroTree(trees), ['classification'] * len(trees))
            logits = stack_outputs(outputs)
            joint_y = joint_y.to(device)
            loss = F.cross_entropy(logits, joint_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += int((logits.argmax(-1) == joint_y).sum())
            total += len(trees)
        print('  seed {} epoch {}: joint train acc {:.4f}'.format(
            seed, epoch, correct / max(total, 1)), flush=True)

    # ---- probe phase: frozen model, held-out combined trees ----
    model.eval()
    probe_ds = CombinedTreeDataset(parts, 'test', args.probe_size, seed + 1)
    probe_loader = DataLoader(probe_ds, batch_size=args.batch_size,
                              collate_fn=collate, num_workers=args.num_workers)
    feats, labels_all, primaries = [], [], []
    with torch.no_grad():
        for trees, _joint, labels, primary in probe_loader:
            feats.append(encode(model, trees, device))
            labels_all.append(labels)
            primaries.append(primary)
    feats = torch.cat(feats).to(device)
    labels_all = torch.cat(labels_all).to(device)
    primaries = torch.cat(primaries).to(device)

    n_train = int(len(feats) * 0.8)
    targets = {p['domain']: (labels_all[:, i], p['classes'])
               for i, p in enumerate(parts)}
    targets['primary_domain'] = (primaries, len(parts))

    for name, (ys, classes) in targets.items():
        acc = fit_probe(feats[:n_train], ys[:n_train],
                        feats[n_train:], ys[n_train:], classes, device)
        rows.append({'seed': seed, 'probe': name, 'acc': acc,
                     'chance': 1.0 / classes})
        print('  seed {} probe[{}] acc {:.4f} (chance {:.3f})'.format(
            seed, name, acc, 1.0 / classes), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', default='mnist,speechcommands,imdb')
    parser.add_argument('--version', default='gaau')
    parser.add_argument('--engine', default='flat', choices=['flat', 'mask', 'recursive'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 42, 7])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train-size', type=int, default=30000)
    parser.add_argument('--probe-size', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        print('--- seed {} ---'.format(seed), flush=True)
        run_seed(args, seed, args.device, rows)

    print('=== probe accuracy over {} seed(s) ==='.format(len(args.seeds)))
    by_probe = {}
    for row in rows:
        by_probe.setdefault(row['probe'], []).append(row['acc'])
    for name in sorted(by_probe):
        vals = by_probe[name]
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print('%-16s %.4f +/- %.4f' % (name, statistics.mean(vals), std))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'probe_e2.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['seed', 'probe', 'acc', 'chance'])
        writer.writeheader()
        writer.writerows(rows)
    print('rows written to', out_path)


if __name__ == '__main__':
    main()
