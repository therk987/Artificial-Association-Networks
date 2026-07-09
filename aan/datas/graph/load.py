"""UPFD fake-news propagation graphs (paper, Experiment 5, 'graph').

UPFD-GOS = the ``gossipcop`` graphs of Dou et al. 2021 with the 10-dimensional
``profile`` node features and the dataset's official train/val/test split
(1092/546/3826 graphs). Each sample is the news propagation graph: node 0 is
the news item, the remaining nodes are spreading users; edges are directed as
shipped (news/tweet -> retweet).

The graphs ship as a compact ``.pt`` cache. If the cache is missing it is
rebuilt from either (a) ``torch_geometric.datasets.UPFD`` when PyG is
installed, or (b) the raw UPFD files (``A.txt``, ``new_profile_feature.npz``,
``node_graph_id.npy``, ``graph_labels.npy``, ``*_idx.npy``) exactly the way
PyG processes them — point ``AAN_UPFD_RAW`` at the raw directory. The raw
archive itself comes from the UPFD authors' Google Drive (what PyG downloads).
"""
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

_RAW_DIR_CANDIDATES = (
    os.environ.get('AAN_UPFD_RAW', ''),
    os.path.join(_HERE, 'UPFD', 'gossipcop', 'raw'),
    '/Users/seokjunkim/bigedu/dolphin-coding-mlops/dataset/graph/UPFD/gossipcop/raw',
)


def _first_existing(candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def _build_cache_from_pyg(cache_path, name, feature):
    from torch_geometric.datasets import UPFD
    root = os.path.join(_HERE, 'UPFD')
    blob = {}
    for split, pyg_split in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        ds = UPFD(root=root, name=name, feature=feature, split=pyg_split)
        xs, ys = [], []
        for data in ds:
            xs.append((data.x.to(torch.float32), data.edge_index.to(torch.long)))
            ys.append(int(data.y))
        blob[split] = {'x': xs, 'y': torch.tensor(ys, dtype=torch.long)}
    torch.save(blob, cache_path)
    print('saved', cache_path, '(via torch_geometric)', flush=True)


def _build_cache_from_raw(cache_path, raw_dir, feature):
    import numpy as np
    import scipy.sparse as sp

    x_all = sp.load_npz(os.path.join(raw_dir, 'new_{}_feature.npz'.format(feature)))
    x_all = torch.from_numpy(np.asarray(x_all.todense(), dtype=np.float32))
    node_graph_id = np.load(os.path.join(raw_dir, 'node_graph_id.npy')).astype(np.int64)
    y_all = np.load(os.path.join(raw_dir, 'graph_labels.npy'))
    y_all = torch.from_numpy(np.unique(y_all, return_inverse=True)[1]).to(torch.long)

    # nodes are grouped by graph id; per-graph node ranges
    counts = np.bincount(node_graph_id, minlength=int(node_graph_id.max()) + 1)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    edges = np.loadtxt(os.path.join(raw_dir, 'A.txt'),
                       delimiter=',', dtype=np.int64)
    src_graph = node_graph_id[edges[:, 0]]
    order = np.argsort(src_graph, kind='stable')
    edges, src_graph = edges[order], src_graph[order]
    edge_starts = np.searchsorted(src_graph, np.arange(len(counts)))
    edge_ends = np.searchsorted(src_graph, np.arange(len(counts)), side='right')

    def graph(i):
        lo, hi = offsets[i], offsets[i + 1]
        x = x_all[lo:hi]
        e = edges[edge_starts[i]:edge_ends[i]] - lo  # local node indices
        edge_index = torch.from_numpy(e.T.copy()).to(torch.long)
        return x, edge_index

    blob = {}
    for split, fname in (('train', 'train_idx.npy'), ('valid', 'val_idx.npy'),
                         ('test', 'test_idx.npy')):
        idx = np.load(os.path.join(raw_dir, fname)).astype(np.int64)
        xs = [graph(i) for i in idx]
        blob[split] = {'x': xs, 'y': y_all[torch.from_numpy(idx)]}
    torch.save(blob, cache_path)
    print('saved', cache_path, '(from raw files at {})'.format(raw_dir), flush=True)


def UPFD_DATA(name='gossipcop', feature='profile'):
    """Return {'train'|'valid'|'test': (list of (x (N,F), edge_index (2,E)), LongTensor y)}."""
    cache_path = os.path.join(_HERE, 'upfd_{}_{}.pt'.format(name, feature))
    if not os.path.exists(cache_path):
        try:
            _build_cache_from_pyg(cache_path, name, feature)
        except ImportError:
            raw_dir = _first_existing(_RAW_DIR_CANDIDATES)
            if raw_dir is None:
                raise RuntimeError(
                    'UPFD cache {} not found, torch_geometric is not installed '
                    'and no raw directory found; set AAN_UPFD_RAW to the UPFD '
                    '{}/raw directory or copy the cache from a machine that '
                    'has it.'.format(cache_path, name))
            _build_cache_from_raw(cache_path, raw_dir, feature)
    blob = torch.load(cache_path)
    result = {split: (blob[split]['x'], blob[split]['y'])
              for split in ('train', 'valid', 'test')}
    print('LOAD : UPFD-{}  train {} valid {} test {}'.format(
        name, len(result['train'][0]), len(result['valid'][0]),
        len(result['test'][0])), flush=True)
    return result
