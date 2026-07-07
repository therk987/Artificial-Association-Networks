"""Multi-seed experiment runner for AAN classification experiments.

Supports the paper-revision experiment protocol: every configuration is run
over several seeds and reported as mean +/- std. Datasets can be combined
with commas for multi-domain simultaneous learning (paper, Experiment 1)
using a joint label space, e.g.:

    python experiments/run.py --dataset mnist --epochs 5 --seeds 1234 42 7
    python experiments/run.py --dataset mnist,speechcommands,imdb --epochs 10

Use --limit for a quick smoke run (subsample each domain's training set).
"""
import argparse
import csv
import os
import random
import statistics
import sys
import time
from collections import defaultdict

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


def stack_outputs(outputs):
    """Main-task outputs: a tensor (single-task fast path) or a list of rows."""
    if torch.is_tensor(outputs):
        return outputs
    return torch.stack(outputs, dim=0).squeeze(1)


# ---------------------------------------------------------------------------
# per-domain "parts": {'domain', 'splits': {name: (xs, ys)}, 'builder',
#                      'encoder', 'classes'} — combined by build_from_parts
# ---------------------------------------------------------------------------

def mnist_image2neurotree(data, mt):
    leaf = NeuroNode(data.to(dtype=torch.float) / 255, 'image')
    mid = NeuroNode(None, None, C=[leaf])
    return NeuroNode(None, None, C=[mid])


def sound2neurotree(data, mt):
    leaf = NeuroNode(data, 'sound')
    return NeuroNode(None, None, C=[leaf])


def text2neurotree(data, mt):
    leaf = NeuroNode(data, 'text')
    return NeuroNode(None, None, C=[leaf])


def _data_root(*parts):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'aan', 'datas', *parts)


def mnist_parts(limit=None):
    from aan.datas.image.load import MNIST_DATA
    from aan.models.feature_encoders.domains.image2vec import LeNet_5

    mnist_train, mnist_test = MNIST_DATA(_data_root('image'), resize_shape=(28, 28))

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

    return {
        'domain': 'image',
        'splits': {'train': (train_x, train_y), 'valid': (valid_x, valid_y),
                   'test': (test_x, test_y)},
        'builder': mnist_image2neurotree,
        'encoder': LeNet_5(conv3_kernel=4),
        'classes': 10,
    }


def speechcommands_parts(limit=None):
    """Speech Commands v0.02, 35 classes, raw 16 kHz waveforms + M5 (paper, Exp. 1).

    All waveforms are decoded ONCE into an fp16 tensor cache (~3.4GB) so
    training streams from memory like MNIST; per-epoch audio decoding made
    the GPU sit mostly idle otherwise.
    """
    from aan.models.feature_encoders.domains.sound2vec import M5_GroupNorm

    data_root = _data_root('sound')
    os.makedirs(data_root, exist_ok=True)
    cache_path = os.path.join(data_root, 'sc_waveforms_fp16.pt')

    if not os.path.exists(cache_path):
        import torchaudio
        blob = {}
        for name in ('training', 'validation', 'testing'):
            ds = torchaudio.datasets.SPEECHCOMMANDS(data_root, download=True, subset=name)
            xs = torch.zeros(len(ds), 1, 16000, dtype=torch.float16)
            labels = []
            print('decoding SC {} ({} files)...'.format(name, len(ds)), flush=True)
            for i in range(len(ds)):
                wav, sr, label, _, _ = ds[i]
                w = wav[:1, :16000]
                xs[i, :, :w.shape[1]] = w.to(torch.float16)
                labels.append(label)
            blob[name] = (xs, labels)
        torch.save(blob, cache_path)
        print('saved', cache_path, flush=True)
    blob = torch.load(cache_path)

    label_names = sorted(set(blob['training'][1]))
    label_index = {name: i for i, name in enumerate(label_names)}

    splits = {}
    for split, name in (('train', 'training'), ('valid', 'validation'), ('test', 'testing')):
        xs, labels = blob[name]
        n = min(len(xs), limit) if limit else len(xs)
        ys = torch.tensor([label_index[l] for l in labels[:n]], dtype=torch.long)
        splits[split] = (xs[:n], ys)

    return {
        'domain': 'sound',
        'splits': splits,
        'builder': sound2neurotree,
        'encoder': M5_GroupNorm(output_size=128),
        'classes': len(label_names),
    }


def mfcc2neurotree(data, mt):
    """(40, T) MFCC frames -> depth-T chain, leaf = first frame (paper, Exp. 3)."""
    node = NeuroNode(data[:, 0], 'mfcc')
    for t in range(1, data.shape[1]):
        node = NeuroNode(data[:, t], 'mfcc', C=[node])
    return node


def speechcommands_mfcc_parts(limit=None):
    """Deep-neurotree SC (paper, Exp. 3): 40-dim MFCC frames as a depth-81
    chain with one FC feature layer — the regime where non-gated cells fail
    to propagate errors. MFCCs are computed once from the waveform cache."""
    sc = speechcommands_parts(limit=None)

    data_root = _data_root('sound')
    cache_path = os.path.join(data_root, 'sc_mfcc_fp16.pt')
    if not os.path.exists(cache_path):
        import torchaudio
        mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        blob = {}
        for split in ('train', 'valid', 'test'):
            xs, ys = sc['splits'][split]
            out = torch.zeros(len(xs), 40, 81, dtype=torch.float16)
            print('MFCC {} ({} clips)...'.format(split, len(xs)), flush=True)
            step = 512
            for i in range(0, len(xs), step):
                out[i:i + step] = mfcc(xs[i:i + step, 0, :].to(torch.float32)).to(torch.float16)
            blob[split] = out
        torch.save(blob, cache_path)
        print('saved', cache_path, flush=True)
    blob = torch.load(cache_path)

    from aan.models.feature_encoders.domains.sound2vec import FullyConnectedLayer_MFCC
    splits = {}
    for split in ('train', 'valid', 'test'):
        xs, ys = sc['splits'][split]
        n = min(len(xs), limit) if limit else len(xs)
        splits[split] = (blob[split][:n], ys[:n])

    return {
        'domain': 'mfcc',
        'splits': splits,
        'builder': mfcc2neurotree,
        'encoder': FullyConnectedLayer_MFCC(input_dim=40, output_dim=128),
        'classes': sc['classes'],
    }


def imdb_parts(limit=None):
    from aan.datas.text.load import IMDB_DATA
    from aan.models.feature_encoders.domains.text2vec import TextCNNEncoder

    train_x, train_y, valid_x, valid_y, test_x, test_y, vocab = \
        IMDB_DATA(_data_root('text'))
    if limit:
        train_x, train_y = train_x[:limit], train_y[:limit]
        valid_x, valid_y = valid_x[:limit], valid_y[:limit]
        test_x, test_y = test_x[:limit], test_y[:limit]

    return {
        'domain': 'text',
        'splits': {'train': (train_x, train_y), 'valid': (valid_x, valid_y),
                   'test': (test_x, test_y)},
        'builder': text2neurotree,
        'encoder': TextCNNEncoder(vocab_size=len(vocab)),
        'classes': 2,
    }


def prebuilt2neurotree(tree, mt):
    tree.reset_state()
    return tree


def algorithms_parts(limit=None, ablate='none'):
    """Sorting-algorithm CFG neurotrees (paper Exp. 4) — the multi-parent,
    level-jumping dataset. E6 degrades exactly those axes via --ablate
    (none / single-parent / no-level-jump / both)."""
    from aan.datas.code.load import SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA
    from aan.models.feature_encoders.domains.code2vec import (Ast2VectorForPython,
                                                              Constant2Vec)
    from aan.data_structures.ablation import ABLATIONS, count_nodes

    data = SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA()
    transform = ABLATIONS[ablate]
    splits = {}
    grown = []
    for split in ('train', 'valid', 'test'):
        pairs, ys = data[split]['x'], data[split]['y']
        if limit:
            pairs, ys = pairs[:limit], ys[:limit]
        trees = []
        for _ast_tree, cfg_tree in pairs:
            new_tree = transform(cfg_tree)
            grown.append(count_nodes(new_tree))
            trees.append(new_tree)
        splits[split] = (trees, torch.tensor(ys, dtype=torch.long))
    print('algorithms ablate={} mean nodes/tree {:.1f}'.format(
        ablate, sum(grown) / max(len(grown), 1)), flush=True)

    return {
        'domain': 'code',
        'splits': splits,
        'builder': prebuilt2neurotree,
        'encoders': {'class': Ast2VectorForPython(input_dim=25, output_dim=128),
                     'Num': Constant2Vec(128)},
        'encoder': None,
        'classes': 6,
    }


PARTS = {
    'mnist': mnist_parts,
    'speechcommands': speechcommands_parts,
    'speechcommands_mfcc': speechcommands_mfcc_parts,
    'imdb': imdb_parts,
    'algorithms': algorithms_parts,
}

# backward-compatible alias (bench scripts referenced DATASETS)
DATASETS = PARTS


def build_from_parts(parts_list):
    """Combine domains into joint-label datasets (paper's 47-class setting)."""
    offsets = {}
    total_classes = 0
    for p in parts_list:
        offsets[p['domain']] = total_classes
        total_classes += p['classes']

    datasets = {}
    for split in ('train', 'valid', 'test'):
        x_map, y_map, builders = {}, {}, {}
        for p in parts_list:
            xs, ys = p['splits'][split]
            offset = offsets[p['domain']]
            if torch.is_tensor(ys):
                ys = ys + offset
            else:
                ys = [y + offset for y in ys]
            x_map[p['domain']] = xs
            y_map[p['domain']] = ys
            per_split = p.get('builder_per_split')
            builders[p['domain']] = per_split[split] if per_split else p['builder']
        maintask_map = {p['domain']: 'classification' for p in parts_list}
        datasets[split] = NeuroDataset(x_map, y_map, maintask_map, builders)

    encoders = {}
    for p in parts_list:
        # a part may span several feature domains (e.g. code: class + Num)
        encoders.update(p.get('encoders') or {p['domain']: p['encoder']})
    return datasets['train'], datasets['valid'], datasets['test'], encoders, total_classes


def _subsample_train(part, fraction, seed):
    """Keep a deterministic random fraction of the TRAIN split only
    (valid/test stay full) — the low-resource setting of the positive
    transfer experiment (E4).

    Seeded with the RUN seed: for a given seed the alone and joint arms see
    the SAME low-resource subset (paired comparison), while different seeds
    draw different subsets (data-selection variance is measured too)."""
    xs, ys = part['splits']['train']
    n = max(1, int(len(xs) * fraction))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(xs), generator=g)[:n]
    if torch.is_tensor(xs):
        xs = xs[perm]
    else:
        xs = [xs[i] for i in perm.tolist()]
    if torch.is_tensor(ys):
        ys = ys[perm]
    else:
        ys = [ys[i] for i in perm.tolist()]
    part['splits']['train'] = (xs, ys)
    return part


def build_dataset(name, limit=None, subsample_seed=0, ablate='none'):
    """``name`` is a comma list of domain specs; ``domain@0.1`` keeps a
    deterministic 10% of that domain's training data (E4 low-resource).
    ``ablate`` applies the E6 structural degradation (algorithms only)."""
    parts_list = []
    for spec in (n.strip() for n in name.split(',') if n.strip()):
        fraction = None
        if '@' in spec:
            spec, frac_str = spec.split('@', 1)
            fraction = float(frac_str)
        if spec not in PARTS:
            raise ValueError('unknown dataset {} (available: {})'.format(
                spec, sorted(PARTS)))
        if spec == 'algorithms':
            part = PARTS[spec](limit, ablate=ablate)
        else:
            part = PARTS[spec](limit)
        if fraction is not None:
            part = _subsample_train(part, fraction, subsample_seed)
        parts_list.append(part)
    return build_from_parts(parts_list)


# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    """Return (overall accuracy, per-domain accuracy dict)."""
    model.eval()
    correct = defaultdict(int)
    total = defaultdict(int)
    with torch.no_grad():
        for batch, y, mt, d in loader:
            outputs, _, _ = model(batch, list(mt))
            preds = stack_outputs(outputs).argmax(dim=-1).view(-1)
            targets = torch.stack(y, dim=0).to(device, dtype=torch.long).view(-1)
            hits = (preds == targets).cpu()
            for i, domain in enumerate(d):
                correct[domain] += int(hits[i])
                total[domain] += 1
    overall = sum(correct.values()) / max(sum(total.values()), 1)
    per_domain = {k: correct[k] / total[k] for k in sorted(total)}
    return overall, per_domain


def run_one_seed(args, seed, device):
    set_seed(seed)
    train_ds, valid_ds, test_ds, feature_encoders, class_count = \
        build_dataset(args.dataset, limit=args.limit, subsample_seed=seed,
                      ablate=args.ablate)

    # NOTE: persistent_workers deadlocked with 3 loaders x 8 workers on
    # torch 2.1 nightly (main thread stuck in poll at epoch boundaries) —
    # workers are recreated per epoch instead, which costs ~1-2s/epoch.
    workers = dict(num_workers=args.num_workers)
    train_loader = createNeuroDataloader(train_ds, batch_size=args.batch_size, shuffle=True, **workers)
    valid_loader = createNeuroDataloader(valid_ds, batch_size=args.batch_size, **workers)
    test_loader = createNeuroDataloader(test_ds, batch_size=args.batch_size, **workers)

    model = ArtificialAssociationNeuralNetworks(
        args.input_dim, args.hidden_dim,
        feature_encoders, {}, {},
        {'classification': ClassificationHead(args.hidden_dim, class_count)},
        version=args.version, engine=args.engine,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='mnist',
                        help='domain name or comma-combined list ({})'.format(sorted(PARTS)))
    parser.add_argument('--version', default='gaau')
    parser.add_argument('--engine', default='flat', choices=['flat', 'mask', 'recursive'])
    parser.add_argument('--num-workers', type=int, default=0,
                        help='dataloader workers (tree building / audio decoding)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1234, 42, 7, 2024, 31337])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--limit', type=int, default=None,
                        help='subsample each domain for a quick smoke run')
    parser.add_argument('--ablate', default='none',
                        choices=['none', 'single-parent', 'no-level-jump', 'both'],
                        help='E6 structural ablation (algorithms dataset)')
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--out', default=None, help='CSV path for per-seed results')
    args = parser.parse_args()

    print('dataset={} version={} engine={} device={} seeds={}'.format(
        args.dataset, args.version, args.engine, args.device, args.seeds), flush=True)

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
    print('== {} {}: test acc {:.4f} +/- {:.4f} over {} seeds =='.format(
        args.dataset, args.version, mean, std, len(accs)), flush=True)
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
        print('wrote', args.out)


if __name__ == '__main__':
    main()
