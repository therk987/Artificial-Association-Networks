"""Stanford Sentiment Treebank constituency neurotrees (paper, Experiment 5, 'tree').

Construction (matches the original research environment, dolphin-coding-mlops):
the HuggingFace ``sst`` PTB strings were parsed with a cs224u-style PTBParser
into neurotrees where EVERY node is a ``'language'`` neuronode holding a single
vocabulary index — leaves hold the lower-cased word's index, internal nodes
hold the index of ``'$UNK'`` — and the sentence label is the float sentiment
score rounded to a binary class (``round(label)``). Splits: 8544/1101/2210.

The trees ship as a compact structure cache (``sst3_neurotree_cache.pkl``,
nested ``(vocab_index, (children...))`` tuples). If the cache is missing it is
rebuilt once from the legacy research pickle (``sst3-neurotree.pkl``, 164MB,
legacy NeuroNode objects) — set ``AAN_LEGACY_SST_PKL`` if it lives elsewhere.

The leaf encoder uses a frozen GloVe-6B-300d embedding over the training
vocabulary (``sst_train_vocab.pkl``, cs224u ``get_vocab(mincount=2)``, 8736
entries incl. '$UNK'); words missing from GloVe get fixed random vectors in
[-0.5, 0.5] (cs224u ``create_pretrained_embedding``). The resulting matrix is
cached as ``sst_glove300_embedding.pt``; rebuilding it needs the GloVe text
file (``AAN_GLOVE_300D`` or the legacy default path).
"""
import os
import pickle
import random

import torch

from aan.data_structures.neuronode import NeuroNode

_LEGACY_SST_PKL_CANDIDATES = (
    os.environ.get('AAN_LEGACY_SST_PKL', ''),
    '/Users/seokjunkim/bigedu/dolphin-coding-mlops/dataset/text/sentiment/sst3-neurotree.pkl',
)
_GLOVE_300D_CANDIDATES = (
    os.environ.get('AAN_GLOVE_300D', ''),
    '/Users/seokjunkim/bigedu/dolphin-coding-mlops/models/pretrained/glove.6B/glove.6B.300d.txt',
)


def _first_existing(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _legacy_tree_to_compact(node):
    """Legacy NeuroNode tree -> nested (vocab_index, (children...)) tuples."""
    return (int(node.x.item()), tuple(_legacy_tree_to_compact(c) for c in node.C))


def _build_compact_cache(cache_path):
    legacy_pkl = _first_existing(_LEGACY_SST_PKL_CANDIDATES)
    if legacy_pkl is None:
        raise RuntimeError(
            'SST cache {} not found and the legacy research pickle '
            '(sst3-neurotree.pkl) is unavailable; set AAN_LEGACY_SST_PKL or '
            'copy the cache from a machine that has it.'.format(cache_path))
    from aan.datas.code.load import load_compat_pickle
    print('building SST compact cache from', legacy_pkl, flush=True)
    legacy = load_compat_pickle(legacy_pkl)
    blob = {}
    for split in ('train', 'valid', 'test'):
        trees = [_legacy_tree_to_compact(t) for t in legacy[split]['x']]
        ys = [int(y) for y in legacy[split]['y']]
        blob[split] = {'x': trees, 'y': ys}
    with open(cache_path, 'wb') as f:
        pickle.dump(blob, f)
    print('saved', cache_path, flush=True)


def _compact_to_neurotree(compact, index_tensors):
    """Rebuild a NeuroNode tree; equal vocab indices share one x tensor."""
    idx, children = compact
    x = index_tensors.get(idx)
    if x is None:
        x = index_tensors[idx] = torch.tensor([idx], dtype=torch.long)
    if children:
        return NeuroNode(x, 'language',
                         C=[_compact_to_neurotree(c, index_tensors) for c in children])
    return NeuroNode(x, 'language')


def SST_DATA(root, seed=1234):
    """Return {'train'|'valid'|'test': (list of NeuroNode trees, LongTensor y)}.

    Each split is shuffled deterministically so ``--limit`` smoke runs see a
    label mix (full runs are order-independent).
    """
    cache_path = os.path.join(root, 'sst3_neurotree_cache.pkl')
    if not os.path.exists(cache_path):
        _build_compact_cache(cache_path)
    with open(cache_path, 'rb') as f:
        blob = pickle.load(f)

    index_tensors = {}
    result = {}
    g = torch.Generator().manual_seed(seed)
    for split in ('train', 'valid', 'test'):
        trees = [_compact_to_neurotree(c, index_tensors) for c in blob[split]['x']]
        ys = torch.tensor(blob[split]['y'], dtype=torch.long)
        perm = torch.randperm(len(trees), generator=g)
        trees = [trees[i] for i in perm.tolist()]
        ys = ys[perm]
        result[split] = (trees, ys)
    print('LOAD : SST  train {} valid {} test {}'.format(
        len(result['train'][0]), len(result['valid'][0]), len(result['test'][0])),
        flush=True)
    return result


def _randvec(rng, n=300, lower=-0.5, upper=0.5):
    return [rng.uniform(lower, upper) for _ in range(n)]


def sst_glove_embedding(root, seed=1234):
    """(vocab_size, 300) float32 matrix aligned with sst_train_vocab.pkl."""
    cache_path = os.path.join(root, 'sst_glove300_embedding.pt')
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    vocab_path = os.path.join(root, 'sst_train_vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    glove_path = _first_existing(_GLOVE_300D_CANDIDATES)
    if glove_path is None:
        raise RuntimeError(
            'embedding cache {} not found and glove.6B.300d.txt is '
            'unavailable; set AAN_GLOVE_300D or copy the cache from a machine '
            'that has it.'.format(cache_path))

    print('building SST GloVe embedding from', glove_path, flush=True)
    wanted = {w: i for i, w in enumerate(vocab)}
    matrix = [None] * len(vocab)
    with open(glove_path, encoding='utf8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            i = wanted.get(parts[0])
            if i is not None:
                matrix[i] = [float(v) for v in parts[1:]]
    rng = random.Random(seed)  # OOV words (incl. '$UNK') get fixed random rows
    missing = 0
    for i, row in enumerate(matrix):
        if row is None:
            matrix[i] = _randvec(rng)
            missing += 1
    embedding = torch.tensor(matrix, dtype=torch.float32)
    torch.save(embedding, cache_path)
    print('saved {} ({} words, {} not in GloVe)'.format(
        cache_path, len(vocab), missing), flush=True)
    return embedding
