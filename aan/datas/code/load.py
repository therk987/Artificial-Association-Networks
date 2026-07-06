"""Loader for the sorting-algorithm source-code neurotree dataset (paper, Exp. 4).

The pickle was produced in the original research environment where NeuroNode
lived in ``representation.datastructure.neuronode``; ``CompatUnpickler`` remaps
those references onto the current ``data_structures.neuronode.NeuroNode`` so
the archive stays loadable.
"""
import os
import pickle

import numpy as np

from aan.data_structures.neuronode import NeuroNode

_HERE = os.path.dirname(os.path.abspath(__file__))
SORT_CFG_PKL = os.path.join(_HERE, 'sort', 'result', 'sort_ast_neurotree_cfg.pkl')

_LEGACY_NEURONODE_MODULES = {
    'representation.datastructure.neuronode',
    'models.torch.dataloader.neuronode',
    'dataloader.neuronode',
}


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'NeuroNode' and module in _LEGACY_NEURONODE_MODULES:
            return NeuroNode
        return super().find_class(module, name)


def load_compat_pickle(path):
    with open(path, 'rb') as f:
        return CompatUnpickler(f).load()


def A2Edge(A_c):
    """Convert an adjacency matrix into an edge list [(i, j, weight), ...]."""
    if A_c is None:
        return None
    A_c = np.asarray(A_c)
    A_c = A_c.reshape(A_c.shape[-2], A_c.shape[-1])
    rows, cols = np.nonzero(A_c)
    return [(int(i), int(j), float(A_c[i, j])) for i, j in zip(rows, cols)]


def _reset_trees(sample):
    for item in sample:
        if isinstance(item, NeuroNode):
            item.reset_state()


def SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA(random_state=1234, edge_features=False):
    """Return {'train','valid','test'} dicts of the sort-algorithm neurotrees.

    Each x is a pair (AST-only tree, CFG neurotree); the label is the
    algorithm class parsed from the file path.
    """
    from sklearn.model_selection import train_test_split

    data = load_compat_pickle(SORT_CFG_PKL)

    labeldict = {}
    X, Y = [], []
    for key, value1, value2 in data:
        labelname = key.split('/')[-2]
        label = labeldict.setdefault(labelname, len(labeldict))

        _reset_trees((value1, value2))
        if edge_features:
            value2.A_c = A2Edge(value2.A_c)
        X.append((value1, value2, key))
        Y.append(label)

    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train, test_size=test_size, random_state=random_state, stratify=Y_train)

    def pack(xs, ys):
        dataset = {'x': [], 'y': [], 'f': []}
        for (x1, x2, f), y in zip(xs, ys):
            dataset['x'].append((x1, x2))
            dataset['y'].append(y)
            dataset['f'].append(f)
        return dataset

    print('LOAD : SORT AST NEUROTREE CFG')
    splits = {
        'train': pack(X_train, Y_train),
        'valid': pack(X_valid, Y_valid),
        'test': pack(X_test, Y_test),
    }
    for name, split in splits.items():
        print(name, ':', len(split['x']))
    splits['label_names'] = {v: k for k, v in labeldict.items()}
    return splits
