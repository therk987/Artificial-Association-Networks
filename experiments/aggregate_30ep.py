"""Aggregate the 30-epoch protocol results into paper-ready statistics.

Prints, for whatever CSVs exist so far (partial sets are labeled):

    E3  — per-cell test accuracy mean+/-std over seeds, best epochs,
          paired GTAU-GAU differences with t statistic
    E4  — per setting (imdb100/10/01, sc10/01): alone vs joint accuracy
          of the LOW-RESOURCE domain, paired differences, t statistic,
          per-arm best-epoch ranges (the original tables' Model(epoch)
          convention)
    E1  — joint 3-domain reproduction numbers (overall + per domain)

    python experiments/aggregate_30ep.py --results results
"""
import argparse
import csv
import glob
import math
import os
import statistics as st


def rows_from(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        for r in csv.DictReader(open(p)):
            out[int(r['seed'])] = r
    return out


def acc(rows, col='test_acc'):
    return {s: float(r[col]) * 100 for s, r in rows.items()}


def fmt(vals):
    xs = list(vals.values())
    if not xs:
        return 'n/a'
    if len(xs) == 1:
        return '%.2f (n=1)' % xs[0]
    return '%.2f +/- %.2f (n=%d)' % (st.mean(xs), st.stdev(xs), len(xs))


def epochs_of(rows):
    return sorted(int(r['epoch']) for r in rows.values())


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = [b[s] - a[s] for s in seeds]
    m, sd = st.mean(d), st.stdev(d)
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else float('inf')
    return d, m, t, len(seeds)


def report_e3(results):
    print('===== E3 (SC-MFCC depth 81, 30ep) =====')
    per_cell = {}
    for cell in ('ran', 'gau', 'tau', 'ptau', 'gtau'):
        rows = rows_from(os.path.join(results, 'e3_30ep_%s_s*.csv' % cell))
        per_cell[cell] = acc(rows)
        print('%5s  %s   best-epochs %s' % (cell, fmt(per_cell[cell]),
                                            epochs_of(rows)))
    p = paired(per_cell.get('gau', {}), per_cell.get('gtau', {}))
    if p:
        d, m, t, n = p
        print('GTAU-GAU paired: %s  mean %+.2f  t(%d)=%.2f'
              % ([round(x, 2) for x in d], m, n - 1, t))


E4_SETTINGS = [
    ('imdb100', 'text', 'e4_30ep_imdb100_alone.csv', 'e4_30ep_3domain_joint_s*.csv'),
    ('imdb10', 'text', 'e4_30ep_imdb10_alone.csv', 'e4_30ep_imdb10_joint_s*.csv'),
    ('imdb01', 'text', 'e4_30ep_imdb01_alone.csv', 'e4_30ep_imdb01_joint_s*.csv'),
    ('sc10', 'sound', 'e4_30ep_sc10_alone.csv', 'e4_30ep_sc10_joint_s*.csv'),
    ('sc01', 'sound', 'e4_30ep_sc01_alone.csv', 'e4_30ep_sc01_joint_s*.csv'),
]


def report_e4(results):
    print('===== E4 low-resource (30ep, best-valid selection) =====')
    for tag, col, alone_glob, joint_glob in E4_SETTINGS:
        alone_rows = rows_from(os.path.join(results, alone_glob))
        joint_rows = rows_from(os.path.join(results, joint_glob))
        alone = acc(alone_rows, col if alone_rows and col in next(iter(alone_rows.values())) else 'test_acc')
        joint = acc(joint_rows, col) if joint_rows and col in next(iter(joint_rows.values())) else {}
        print('%-8s alone %s @ep%s' % (tag, fmt(alone), epochs_of(alone_rows)))
        print('         joint %s @ep%s' % (fmt(joint), epochs_of(joint_rows)))
        p = paired(alone, joint)
        if p:
            d, m, t, n = p
            print('         paired diffs %s  mean %+.2f  t(%d)=%.2f'
                  % ([round(x, 2) for x in d], m, n - 1, t))


def report_e1(results):
    print('===== E1 reproduction (joint 3-domain, 30ep) =====')
    rows = rows_from(os.path.join(results, 'e4_30ep_3domain_joint_s*.csv'))
    if not rows:
        print('n/a')
        return
    print('overall %s' % fmt(acc(rows)))
    for col in ('image', 'sound', 'text'):
        if col in next(iter(rows.values())):
            print('%6s  %s' % (col, fmt(acc(rows, col))))
    for name, pattern in (('MNIST alone', 'e4_30ep_mnist_alone.csv'),
                          ('SC alone', 'e4_30ep_sc_alone.csv')):
        r = rows_from(os.path.join(results, pattern))
        if r:
            print('%12s %s @ep%s' % (name, fmt(acc(r)), epochs_of(r)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    report_e3(args.results)
    report_e4(args.results)
    report_e1(args.results)


if __name__ == '__main__':
    main()
