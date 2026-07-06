"""Profile WHERE the DFC training step spends its time.

Answers "why is AAN slower than a plain CNN": prints
  (1) a phase breakdown (pipeline / forward / backward+step),
  (2) cProfile hotspots inside the forward pass, grouped by component.

Run from the repository root:
    python experiments/profile_dfc.py --batches 50
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from aan.config.option import device as default_device
from aan.data_structures.batch_neurotree import BatchNeuroTree
from aan.models.artificial_association_networks import ArtificialAssociationNeuralNetworks
from experiments.run import ClassificationHead, set_seed, stack_outputs
from experiments.bench_vs_cnn import load_mnist, image2neurotree, sync


def build_batches(train_x, train_y, batch_size, n_batches):
    batches = []
    for i in range(0, n_batches * batch_size, batch_size):
        trees = [image2neurotree(x, None) for x in train_x[i:i + batch_size]]
        ys = torch.stack(list(train_y[i:i + batch_size])).long()
        batches.append((trees, ys))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--version', default='gaau')
    parser.add_argument('--device', default=default_device)
    args = parser.parse_args()
    device = args.device

    set_seed(1234)
    train_x, train_y, _, _ = load_mnist(limit=(args.batches + 5) * args.batch_size)

    from aan.models.feature_encoders.domains.image2vec import LeNet_5
    model = ArtificialAssociationNeuralNetworks(
        128, 128, {'image': LeNet_5(conv3_kernel=4)}, {}, {},
        {'classification': ClassificationHead(128, 10)}, version=args.version,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = build_batches(train_x, train_y, args.batch_size, args.batches)
    n_samples = args.batches * args.batch_size

    def reset_all(trees):
        for t in trees:
            t.reset_state()

    def forward(trees, ys):
        outputs, _, _ = model(BatchNeuroTree(trees), ['classification'] * len(trees))
        logits = stack_outputs(outputs)
        return F.cross_entropy(logits, ys.to(device))

    # warmup
    for trees, ys in batches[:3]:
        reset_all(trees)
        loss = forward(trees, ys)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # ---- phase 1: tree build cost (data pipeline proxy) ----
    t0 = time.perf_counter()
    for i in range(0, args.batches * args.batch_size, args.batch_size):
        [image2neurotree(x, None) for x in train_x[i:i + args.batch_size]]
    build_time = time.perf_counter() - t0

    # ---- phase 2: forward only ----
    for trees, _ in batches:
        reset_all(trees)
    sync(device); t0 = time.perf_counter()
    with torch.no_grad():
        for trees, ys in batches:
            forward(trees, ys)
    sync(device)
    fwd_time = time.perf_counter() - t0

    # ---- phase 3: full step ----
    for trees, _ in batches:
        reset_all(trees)
    sync(device); t0 = time.perf_counter()
    for trees, ys in batches:
        loss = forward(trees, ys)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    sync(device)
    full_time = time.perf_counter() - t0

    print('device={} batches={} batch_size={}'.format(device, args.batches, args.batch_size))
    print('phase breakdown (ms per batch of {}):'.format(args.batch_size))
    print('  tree build          {:8.2f}'.format(1000 * build_time / args.batches))
    print('  forward (no grad)   {:8.2f}'.format(1000 * fwd_time / args.batches))
    print('  fwd+bwd+step        {:8.2f}'.format(1000 * full_time / args.batches))
    print('  throughput          {:8.0f} samples/s'.format(n_samples / full_time))

    # ---- cProfile the forward pass (python-side attribution) ----
    for trees, _ in batches:
        reset_all(trees)
    prof = cProfile.Profile()
    prof.enable()
    with torch.no_grad():
        for trees, ys in batches:
            forward(trees, ys)
    sync(device)
    prof.disable()

    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream).sort_stats('cumulative')
    stats.print_stats(r'aan/|experiments/', 25)
    print(stream.getvalue())

    # grouped totals by component (tottime, i.e., exclusive python time)
    groups = {
        'BatchNeuroTree bookkeeping': ['batch_neurotree.py'],
        'DFC recursion (encoder)': ['recursive_encoder.py'],
        'feature connector (Psi)': ['multiencoder_connector.py'],
        'task connectors (Phi)': ['multimaintask_connector.py', 'multisubtask_connector.py'],
        'cells (GRU/GAT/GCN/readout)': ['GRU.py', 'GATs.py', 'GCN.py', 'RNN.py', 'readout_max.py'],
        'feature extractor (LeNet)': ['image2vec.py'],
    }
    totals = {k: 0.0 for k in groups}
    stats2 = pstats.Stats(prof)
    for func, (cc, nc, tt, ct, callers) in stats2.stats.items():
        fname = func[0]
        for gname, needles in groups.items():
            if any(n in fname for n in needles):
                totals[gname] += tt
                break
    print('exclusive python time by component (s, over {} forward batches):'.format(args.batches))
    for gname, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print('  {:<28} {:8.3f}  ({:5.1f} ms/batch)'.format(gname, t, 1000 * t / args.batches))


if __name__ == '__main__':
    main()
