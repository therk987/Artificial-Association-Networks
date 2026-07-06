"""IMDB loader without torchtext (deprecated upstream).

Downloads the aclImdb archive, tokenizes with a simple regex tokenizer,
builds a frequency vocabulary from the training split, and caches the
tensorized result. Matches the paper's split sizes (Table III):
17,500 train / 7,500 valid / 25,000 test.
"""
import os
import pickle
import re
import tarfile
import urllib.request

import torch

IMDB_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
PAD_IDX = 0
UNK_IDX = 1

_TOKEN_RE = re.compile(r"[A-Za-z']+|[.,!?;]")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _download_and_extract(root):
    archive = os.path.join(root, 'aclImdb_v1.tar.gz')
    extracted = os.path.join(root, 'aclImdb')
    if os.path.isdir(extracted):
        return extracted
    os.makedirs(root, exist_ok=True)
    if not os.path.exists(archive):
        print('downloading IMDB (84MB)...', flush=True)
        urllib.request.urlretrieve(IMDB_URL, archive)
    print('extracting IMDB...', flush=True)
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(root)
    return extracted


def _read_split(base, split):
    texts, labels = [], []
    for label_name, label in (('neg', 0), ('pos', 1)):
        folder = os.path.join(base, split, label_name)
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(folder, fname), encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(label)
    return texts, labels


def _tensorize(texts, vocab, max_len):
    out = torch.full((len(texts), max_len), PAD_IDX, dtype=torch.long)
    for i, text in enumerate(texts):
        ids = [vocab.get(tok, UNK_IDX) for tok in tokenize(text)[:max_len]]
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def IMDB_DATA(root, vocab_size=25000, max_len=400, valid_ratio=0.3, seed=1234):
    """Return (train_x, train_y, valid_x, valid_y, test_x, test_y, vocab)."""
    cache = os.path.join(root, 'imdb_cache_v{}_l{}_s.pkl'.format(vocab_size, max_len))
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)

    base = _download_and_extract(root)
    train_texts, train_labels = _read_split(base, 'train')
    test_texts, test_labels = _read_split(base, 'test')

    print('building vocab...', flush=True)
    freq = {}
    for text in train_texts:
        for tok in tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1
    most_common = sorted(freq.items(), key=lambda kv: -kv[1])[:vocab_size - 2]
    vocab = {'<pad>': PAD_IDX, '<unk>': UNK_IDX}
    for tok, _ in most_common:
        vocab[tok] = len(vocab)

    print('tensorizing...', flush=True)
    train_x = _tensorize(train_texts, vocab, max_len)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_x = _tensorize(test_texts, vocab, max_len)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    # shuffle both splits (folders are ordered by label, so unshuffled
    # slices — e.g. --limit smoke runs — would be single-class)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_x), generator=g)
    train_x, train_y = train_x[perm], train_y[perm]
    test_perm = torch.randperm(len(test_x), generator=g)
    test_x, test_y = test_x[test_perm], test_y[test_perm]
    n_valid = int(len(train_x) * valid_ratio)
    valid_x, valid_y = train_x[:n_valid], train_y[:n_valid]
    train_x, train_y = train_x[n_valid:], train_y[n_valid:]

    result = (train_x, train_y, valid_x, valid_y, test_x, test_y, vocab)
    with open(cache, 'wb') as f:
        pickle.dump(result, f)
    print('IMDB ready: train {} valid {} test {} vocab {}'.format(
        len(train_x), len(valid_x), len(test_x), len(vocab)), flush=True)
    return result
