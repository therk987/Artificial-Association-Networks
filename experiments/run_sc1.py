"""SC1 — structure as attention masks, at matched code and capacity.

One model class (``AncestorMaskTransformer``), one flag:

    --structured 1   a token attends only to its DESCENDANTS (+self); the
                     mask is compiled per sample from the neurotree — the
                     data defines the computation structure.
    --structured 0   the mask is removed (full attention over the packed
                     nodes, level embeddings kept) — the flat-transformer
                     control arm with byte-identical training code.

structured > flat at matched parameters = the neurotree's structural prior
carries information a plain transformer has to relearn from data. Sweeping
--hidden-dim tests whether that advantage survives capacity growth (the
scale question), e.g.:

    python experiments/run_sc1.py --structured 1 --hidden-dim 128
    python experiments/run_sc1.py --structured 0 --hidden-dim 128
    python experiments/run_sc1.py --dataset speechcommands_mfcc ...
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from aan.config.option import device as default_device
from aan.models.encoders.mask_engine import AncestorMaskTransformer
from aan.models.feature_encoders.multiencoder_connector import MultiExtractionConnector
from aan.data_structures.neurodataloader import createNeuroDataloader

from run import PARTS, set_seed, build_dataset  # noqa: F401


class StructuredClassifier(nn.Module):
    """AncestorMaskTransformer encoder + linear head over root embeddings."""

    def __init__(self, input_dim, hidden_dim, feature_encoders, class_count,
                 n_layers, heads, weight_shared, structured):
        super().__init__()
        self.features = MultiExtractionConnector(input_dim, feature_encoders)
        self.encoder = AncestorMaskTransformer(
            input_dim, hidden_dim, self.features,
            n_layers=n_layers, heads=heads,
            weight_shared=weight_shared, structured=structured)
        self.head = nn.Linear(hidden_dim, class_count)

    def forward(self, batch_tree):
        return self.head(self.encoder(batch_tree))


def evaluate(model, loader, device):
    model.eval()
    correct, total = defaultdict(int), defaultdict(int)
    with torch.no_grad():
        for batch, y, _mt, d in loader:
            preds = model(batch).argmax(dim=-1).view(-1)
            targets = torch.stack(y, dim=0).to(device, dtype=torch.long).view(-1)
            hits = (preds == targets).cpu()
            for i, domain in enumerate(d):
                correct[domain] += int(hits[i])
                total[domain] += 1
    overall = sum(correct.values()) / max(sum(total.values()), 1)
    return overall, {k: correct[k] / total[k] for k in sorted(total)}


def run_one_seed(args, seed, device):
    set_seed(seed)
    train_ds, valid_ds, test_ds, feature_encoders, class_count = \
        build_dataset(args.dataset, limit=args.limit, subsample_seed=seed)

    workers = dict(num_workers=args.num_workers)
    train_loader = createNeuroDataloader(train_ds, batch_size=args.batch_size, shuffle=True, **workers)
    valid_loader = createNeuroDataloader(valid_ds, batch_size=args.batch_size, **workers)
    test_loader = createNeuroDataloader(test_ds, batch_size=args.batch_size, **workers)

    model = StructuredClassifier(args.input_dim, args.hidden_dim,
                                 feature_encoders, class_count,
                                 args.n_layers, args.heads,
                                 bool(args.weight_shared),
                                 bool(args.structured)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {'valid_acc': 0.0, 'test_acc': 0.0, 'epoch': -1, 'per_domain': {}}
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()
        for batch, y, _mt, _d in train_loader:
            logits = model(batch)
            targets = torch.stack(y, dim=0).to(device, dtype=torch.long).view(-1)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        valid_acc, _ = evaluate(model, valid_loader, device)
        if valid_acc >= best['valid_acc']:
            test_acc, per_domain = evaluate(model, test_loader, device)
            best.update(valid_acc=valid_acc, test_acc=test_acc,
                        epoch=epoch, per_domain=per_domain)
        print('  seed {} epoch {}: loss {:.4f} valid {:.4f} ({:.1f}s)'.format(
            seed, epoch, epoch_loss / max(len(train_loader), 1), valid_acc,
            time.perf_counter() - t0), flush=True)

    return {'seed': seed, 'structured': args.structured,
            'hidden_dim': args.hidden_dim, 'n_layers': args.n_layers,
            'params': n_params, 'valid_acc': best['valid_acc'],
            'test_acc': best['test_acc'], 'epoch': best['epoch'],
            'per_domain': best['per_domain']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='mnist,speechcommands,imdb')
    parser.add_argument('--structured', type=int, default=1, choices=[0, 1])
    parser.add_argument('--weight-shared', type=int, default=0, choices=[0, 1])
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 42, 7])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    print('dataset={} structured={} hidden={} layers={} seeds={}'.format(
        args.dataset, args.structured, args.hidden_dim, args.n_layers,
        args.seeds), flush=True)

    rows = []
    for seed in args.seeds:
        rows.append(run_one_seed(args, seed, args.device))
        r = rows[-1]
        print('seed {}: valid {:.4f} test {:.4f} @ep{} ({} params)'.format(
            r['seed'], r['valid_acc'], r['test_acc'], r['epoch'], r['params']),
            flush=True)

    accs = [r['test_acc'] for r in rows]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
    print('=== {} structured={} hidden={}: test {:.4f} +/- {:.4f} over {} seed(s)'.format(
        args.dataset, args.structured, args.hidden_dim, mean, std, len(accs)),
        flush=True)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
        'sc1_%s_h%d.csv' % ('struct' if args.structured else 'flat', args.hidden_dim))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    domains = sorted({k for r in rows for k in r['per_domain']})
    with open(out_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['seed', 'structured', 'hidden_dim', 'n_layers',
                         'params', 'valid_acc', 'test_acc', 'epoch'] + domains)
        for r in rows:
            writer.writerow([r['seed'], r['structured'], r['hidden_dim'],
                             r['n_layers'], r['params'], r['valid_acc'],
                             r['test_acc'], r['epoch']]
                            + [r['per_domain'].get(k, '') for k in domains])
    print('rows written to %s' % out_path)


if __name__ == '__main__':
    main()
