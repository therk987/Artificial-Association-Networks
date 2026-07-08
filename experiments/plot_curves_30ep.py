"""Appendix-style per-epoch accuracy curves for the 30-epoch protocol runs.

The original paper's appendix (Figs. 21-24) plots full test-accuracy
trajectories because domains and models peak at different epochs; the
tables then report the best-validation epoch per arm. This script restores
that transparency for the revision experiments from the run.py logs
(which record validation accuracy at every epoch):

    Figure e3_curves:  SC-MFCC depth-81, one line per cell (mean over
                       seeds, +/- std band) — the deep-chain trajectories.
    Figure e4_curves:  one panel per low-resource setting, alone vs joint
                       (joint curves are overall validation accuracy and
                       are labeled as such; per-domain test numbers come
                       from the CSVs).

    python experiments/plot_curves_30ep.py --results results --out results/figs
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINE = re.compile(r'seed (\d+) epoch (\d+): loss [\d.]+ valid ([\d.]+)')


def parse_log(path):
    """{seed: [valid_acc per epoch]} from a run.py log."""
    curves = {}
    with open(path) as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                seed, epoch, valid = int(m.group(1)), int(m.group(2)), float(m.group(3))
                curves.setdefault(seed, [])
                # epochs print in order per seed; pad defensively
                while len(curves[seed]) <= epoch:
                    curves[seed].append(None)
                curves[seed][epoch] = valid
    return curves


def merged_curves(paths):
    merged = {}
    for p in paths:
        for seed, ys in parse_log(p).items():
            merged[seed] = ys
    return merged


def mean_band(curves):
    """(epochs, mean, lo, hi) across seeds, ignoring missing tails."""
    if not curves:
        return [], [], [], []
    n = max(len(ys) for ys in curves.values())
    xs, mean, lo, hi = [], [], [], []
    for e in range(n):
        vals = [ys[e] for ys in curves.values() if e < len(ys) and ys[e] is not None]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        xs.append(e)
        mean.append(m)
        lo.append(m - s)
        hi.append(m + s)
    return xs, mean, lo, hi


def plot_e3(results, out_dir):
    cells = ['ran', 'gau', 'tau', 'ptau', 'gtau']
    fig, ax = plt.subplots(figsize=(6, 4))
    for cell in cells:
        paths = sorted(glob.glob(os.path.join(results, 'e3_30ep_%s_s*.log' % cell)))
        xs, mean, lo, hi = mean_band(merged_curves(paths))
        if not xs:
            continue
        ax.plot(xs, mean, label=cell.upper() if cell != 'ptau' else 'TAU-pre')
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation accuracy')
    ax.set_title('SC-MFCC (depth 81), 30 epochs, five seeds')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, 'e3_curves.pdf')
    fig.savefig(path)
    print('wrote', path)


E4_PANELS = [
    ('imdb100', 'IMDB 100%'), ('imdb10', 'IMDB 10%'), ('imdb01', 'IMDB 1%'),
    ('sc10', 'SC 10%'), ('sc01', 'SC 1%'),
]


def plot_e4(results, out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6), sharey=False)
    for ax, (tag, title) in zip(axes, E4_PANELS):
        alone = merged_curves(sorted(glob.glob(
            os.path.join(results, 'e4_30ep_%s_alone.log' % tag))))
        joint_tag = '3domain' if tag == 'imdb100' else tag
        joint = merged_curves(sorted(glob.glob(
            os.path.join(results, 'e4_30ep_%s_joint_s*.log' % joint_tag))))
        for label, curves in (('alone', alone), ('joint (overall)', joint)):
            xs, mean, lo, hi = mean_band(curves)
            if not xs:
                continue
            ax.plot(xs, mean, label=label)
            ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Validation accuracy')
    fig.tight_layout()
    path = os.path.join(out_dir, 'e4_curves.pdf')
    fig.savefig(path)
    print('wrote', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--out', default='results/figs')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot_e3(args.results, args.out)
    plot_e4(args.results, args.out)


if __name__ == '__main__':
    main()
