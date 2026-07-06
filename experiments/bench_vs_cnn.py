"""Speed comparison: AAN (neurotree/DFC pipeline) vs a plain CNN on MNIST.

Both models share the same LeNet-5 backbone so the comparison isolates the
neurotree overhead (tree building, collate, recursive DFC, connectors).
Reported per model:
  - train throughput (samples/sec, full pipeline incl. data loading)
  - forward+backward-only throughput (pre-built batches, no data pipeline)
  - test accuracy after --epochs epochs

Feeds the paper's efficiency table (REVISION plan E7).

Run from the repository root:
    python experiments/bench_vs_cnn.py --epochs 2 --limit 20000
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from aan.config.option import device as default_device
from aan.data_structures.neuronode import NeuroNode
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.data_structures.neurodataloader import NeuroDataset, createNeuroDataloader
from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks
from experiments.run import ClassificationHead, set_seed, stack_outputs


class PlainLeNet(nn.Module):
    """Same backbone as aan.models.feature_encoders.domains.image2vec.LeNet_5
    (conv3_kernel=4 for 28x28), followed by a linear classification head."""

    def __init__(self, class_count=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        self.head = nn.Linear(120, class_count)
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 120)
        return self.head(x)


def load_mnist(limit):
    from aan.datas.image.load import MNIST_DATA
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'aan', 'datas', 'image')
    mnist_train, mnist_test = MNIST_DATA(data_root, resize_shape=(28, 28))
    train_x = mnist_train.data.unsqueeze(1).to(dtype=torch.float) / 255
    train_y = mnist_train.targets
    test_x = mnist_test.data.unsqueeze(1).to(dtype=torch.float) / 255
    test_y = mnist_test.targets
    if limit:
        train_x, train_y = train_x[:limit], train_y[:limit]
        test_x, test_y = test_x[:limit], test_y[:limit]
    return train_x, train_y, test_x, test_y


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# plain CNN
# ---------------------------------------------------------------------------

def bench_cnn(args, train_x, train_y, test_x, test_y, device):
    set_seed(1234)
    model = PlainLeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(train_x, train_y),
                        batch_size=args.batch_size, shuffle=True)

    epoch_times = []
    for _ in range(args.epochs):
        sync(device)
        t0 = time.perf_counter()
        for x, y in loader:
            logits = model(x.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sync(device)
        epoch_times.append(time.perf_counter() - t0)

    # forward+backward only, pre-moved batches
    batches = [(train_x[i:i + args.batch_size].to(device),
                train_y[i:i + args.batch_size].to(device))
               for i in range(0, min(len(train_x), 100 * args.batch_size), args.batch_size)]
    sync(device)
    t0 = time.perf_counter()
    for x, y in batches:
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sync(device)
    fb_time = time.perf_counter() - t0
    fb_samples = sum(len(x) for x, _ in batches)

    model.eval()
    with torch.no_grad():
        preds = model(test_x.to(device)).argmax(1).cpu()
    acc = (preds == test_y).float().mean().item()

    return {
        'train_sps': len(train_x) / statistics.mean(epoch_times),
        'fb_sps': fb_samples / fb_time,
        'epoch_sec': statistics.mean(epoch_times),
        'test_acc': acc,
    }


# ---------------------------------------------------------------------------
# AAN
# ---------------------------------------------------------------------------

def image2neurotree(data, mt):
    leaf = NeuroNode(data, 'image')
    mid = NeuroNode(None, None, C=[leaf])
    return NeuroNode(None, None, C=[mid])


def bench_aan(args, train_x, train_y, test_x, test_y, device, engine):
    from aan.models.feature_encoders.domains.image2vec import LeNet_5

    set_seed(1234)
    model = ArtificialAssociationNeuralNetworks(
        128, 128,
        {'image': LeNet_5(conv3_kernel=4)}, {}, {},
        {'classification': ClassificationHead(128, 10)},
        version=args.version, engine=engine,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = NeuroDataset({'image': train_x}, {'image': train_y},
                           {'image': 'classification'}, {'image': image2neurotree})
    loader = createNeuroDataloader(dataset, batch_size=args.batch_size, shuffle=True)

    def step(batch, y, mt):
        outputs, _, _ = model(batch, list(mt))
        logits = stack_outputs(outputs)
        targets = torch.stack(y, dim=0).to(device, dtype=torch.long).view(-1)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_times = []
    for _ in range(args.epochs):
        sync(device)
        t0 = time.perf_counter()
        for batch, y, mt, d in loader:
            step(batch, y, mt)
        sync(device)
        epoch_times.append(time.perf_counter() - t0)

    # forward+backward only: pre-built neurotree batches (no dataset/collate)
    fb_samples = min(len(train_x), 100 * args.batch_size)
    prebuilt = []
    for i in range(0, fb_samples, args.batch_size):
        trees = [image2neurotree(x, None) for x in train_x[i:i + args.batch_size]]
        ys = tuple(train_y[i:i + args.batch_size])
        prebuilt.append((trees, ys))
    sync(device)
    t0 = time.perf_counter()
    for trees, ys in prebuilt:
        for t in trees:
            t.reset_state()
        step(BatchNeuroTree(trees), ys, ['classification'] * len(trees))
    sync(device)
    fb_time = time.perf_counter() - t0

    test_ds = NeuroDataset({'image': test_x}, {'image': test_y},
                           {'image': 'classification'}, {'image': image2neurotree})
    test_loader = createNeuroDataloader(test_ds, batch_size=args.batch_size)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch, y, mt, d in test_loader:
            outputs, _, _ = model(batch, list(mt))
            preds = stack_outputs(outputs).argmax(dim=-1).view(-1).cpu()
            targets = torch.stack(y, dim=0).view(-1)
            correct += (preds == targets).sum().item()
            total += len(y)

    return {
        'train_sps': len(train_x) / statistics.mean(epoch_times),
        'fb_sps': fb_samples / fb_time,
        'epoch_sec': statistics.mean(epoch_times),
        'test_acc': correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--version', default='gaau')
    parser.add_argument('--device', default=default_device)
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = load_mnist(args.limit)
    print('device={} train={} test={} batch={} epochs={}'.format(
        args.device, len(train_x), len(test_x), args.batch_size, args.epochs), flush=True)

    cnn = bench_cnn(args, train_x, train_y, test_x, test_y, args.device)
    print('CNN  done', flush=True)
    rows = [('CNN (LeNet-5)', cnn)]
    for engine in ('recursive', 'flat'):
        r = bench_aan(args, train_x, train_y, test_x, test_y, args.device, engine)
        print('AAN/{} done'.format(engine), flush=True)
        rows.append(('AAN {}/{}'.format(args.version, engine), r))

    header = '{:<20} {:>14} {:>14} {:>11} {:>9}'.format(
        'model', 'train samp/s', 'fwd+bwd samp/s', 'epoch sec', 'test acc')
    print(header)
    print('-' * len(header))
    for name, r in rows:
        print('{:<20} {:>14.0f} {:>14.0f} {:>11.1f} {:>9.4f}'.format(
            name, r['train_sps'], r['fb_sps'], r['epoch_sec'], r['test_acc']))
    for name, r in rows[1:]:
        print('{}: {:.1f}x slower than CNN (full), {:.1f}x (fwd+bwd only)'.format(
            name, cnn['train_sps'] / r['train_sps'], cnn['fb_sps'] / r['fb_sps']))


if __name__ == '__main__':
    main()
