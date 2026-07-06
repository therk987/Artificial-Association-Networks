"""Multi-seed experiment runner for AAN classification experiments.

Supports the paper-revision experiment protocol: every configuration is run
over several seeds and reported as mean +/- std (REVISION goal: statistical
grounding for the multi-domain parity/transfer claims).

Run from the repository root, e.g.:
    python experiments/run.py --dataset mnist --version gaau --epochs 5 \
        --seeds 1234 42 7 --out results/mnist_gaau.csv

Use --limit for a quick smoke run (subsample the training set).
"""
import argparse
import csv
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from aan.config.option import device as default_device
from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.neurodataloader import NeuroDataset, createNeuroDataloader
from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, class_count):
        super().__init__()
        self.layer = nn.Linear(hidden_dim, class_count)

    def forward(self, hiddens, tree):
        if isinstance(hiddens, list):
            hiddens = torch.stack(hiddens, dim=0)
        return self.layer(hiddens)


# ---------------------------------------------------------------------------
# dataset registry: name -> builder returning
#   (train_ds, valid_ds, test_ds, feature_encoders, class_count)
# ---------------------------------------------------------------------------

def build_mnist(limit=None):
    from aan.datas.image.load import MNIST_DATA
    from aan.models.feature_encoders.domains.image2vec import LeNet_5

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'aan', 'datas', 'image')
    # raw 28x28 tensors, as in the notebook (transforms are bypassed below)
    mnist_train, mnist_test = MNIST_DATA(data_root, resize_shape=(28, 28))

    train_x = mnist_train.data.unsqueeze(1)
    train_y = mnist_train.targets
    if limit:
        train_x, train_y = train_x[:limit], train_y[:limit]
    n_valid = max(1, len(train_x) // 6)
    valid_x, valid_y = train_x[:n_valid], train_y[:n_valid]
    train_x, train_y = train_x[n_valid:], train_y[n_valid:]
    test_x, test_y = mnist_test.data.unsqueeze(1), mnist_test.targets
    if limit:
        test_x, test_y = test_x[:limit], test_y[:limit]

    def image2neurotree(data, mt):
        leaf = NeuroNode(data.to(dtype=torch.float) / 255, 'image')
        mid = NeuroNode(None, None, C=[leaf])
        return NeuroNode(None, None, C=[mid])

    builders = {'image': image2neurotree}
    maintask_map = {'image': 'classification'}

    def dataset(x, y):
        return NeuroDataset({'image': x}, {'image': y}, maintask_map, builders)

    feature_encoders = {'image': LeNet_5(conv3_kernel=4)}
    return dataset(train_x, train_y), dataset(valid_x, valid_y), dataset(test_x, test_y), \
        feature_encoders, 10


DATASETS = {
    'mnist': build_mnist,
}


# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch, y, mt, d in loader:
            outputs, _, _ = model(batch, list(mt))
            preds = torch.stack(outputs, dim=0).argmax(dim=-1).squeeze(-1)
            targets = torch.stack(y, dim=0).to(device, dtype=torch.long)
            correct += (preds.view(-1) == targets.view(-1)).sum().item()
            total += len(y)
    return correct / max(total, 1)


def run_one_seed(args, seed, device):
    set_seed(seed)
    train_ds, valid_ds, test_ds, feature_encoders, class_count = \
        DATASETS[args.dataset](limit=args.limit)

    train_loader = createNeuroDataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = createNeuroDataloader(valid_ds, batch_size=args.batch_size)
    test_loader = createNeuroDataloader(test_ds, batch_size=args.batch_size)

    model = ArtificialAssociationNeuralNetworks(
        args.input_dim, args.hidden_dim,
        feature_encoders, {}, {},
        {'classification': ClassificationHead(args.hidden_dim, class_count)},
        version=args.version,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid, best_test, best_epoch = 0.0, 0.0, -1
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()
        for batch, y, mt, d in train_loader:
            outputs, _, _ = model(batch, list(mt))
            logits = torch.stack(outputs, dim=0).squeeze(1)
            targets = torch.stack(y, dim=0).to(device, dtype=torch.long).view(-1)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        valid_acc = evaluate(model, valid_loader, device)
        if valid_acc >= best_valid:  # model selection on validation accuracy
            best_valid = valid_acc
            best_test = evaluate(model, test_loader, device)
            best_epoch = epoch
        print('  seed {} epoch {}: loss {:.4f} valid {:.4f} ({:.1f}s)'.format(
            seed, epoch, epoch_loss / max(len(train_loader), 1), valid_acc,
            time.perf_counter() - t0), flush=True)

    return {'seed': seed, 'valid_acc': best_valid, 'test_acc': best_test, 'epoch': best_epoch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='mnist')
    parser.add_argument('--version', default='gaau')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 42, 7, 2024, 31337])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--limit', type=int, default=None,
                        help='subsample train/test for a quick smoke run')
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--out', default=None, help='CSV path for per-seed results')
    args = parser.parse_args()

    print('dataset={} version={} device={} seeds={}'.format(
        args.dataset, args.version, args.device, args.seeds), flush=True)

    results = []
    for seed in args.seeds:
        results.append(run_one_seed(args, seed, args.device))
        print('seed {} -> test {:.4f} (valid {:.4f}, epoch {})'.format(
            results[-1]['seed'], results[-1]['test_acc'],
            results[-1]['valid_acc'], results[-1]['epoch']), flush=True)

    accs = [r['test_acc'] for r in results]
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print('== {} {}: test acc {:.4f} +/- {:.4f} over {} seeds =='.format(
        args.dataset, args.version, mean, std, len(accs)), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['seed', 'valid_acc', 'test_acc', 'epoch'])
            writer.writeheader()
            writer.writerows(results)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
