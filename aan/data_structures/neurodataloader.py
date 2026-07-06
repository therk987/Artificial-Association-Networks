import torch
from torch.utils.data import Dataset

from aan.data_structures.batch_neurotree import BatchNeuroTree


class NeuroDataset(Dataset):
    """Dataset over several domains at once.

    ``x_map`` / ``y_map`` map a domain name to its samples/labels, and
    ``xmt2neurotree_map`` maps a domain name to a builder function
    ``(x, maintask) -> NeuroNode`` that wraps the raw sample in a neurotree.

    modes:
      - ``'load2neurotree'``: keep raw samples and build the neurotree lazily
        in ``__getitem__`` (a fresh tree per access).
      - ``'neurotree2load'``: build every neurotree once up front;
        ``__getitem__`` resets the tree's per-pass state before returning it.
    """

    MODES = ('load2neurotree', 'neurotree2load')

    def __init__(self, x_map, y_map, maintask_map, xmt2neurotree_map, mode='load2neurotree'):
        if mode not in self.MODES:
            raise ValueError('unknown mode: {} (expected one of {})'.format(mode, self.MODES))

        self.xmt2neurotree_map = xmt2neurotree_map
        self.mode = mode
        self.X, self.Y, self.MT, self.D = self._flatten(x_map, y_map, maintask_map)

        if mode == 'neurotree2load':
            self.X = [
                self.xmt2neurotree_map[d](x, mt)
                for x, mt, d in zip(self.X, self.MT, self.D)
            ]

    def _flatten(self, x_map, y_map, maintask_map):
        X, Y, MT, D = [], [], [], []
        for domain_name in x_map.keys():
            d_X = x_map[domain_name]
            d_Y = y_map[domain_name]
            d_maintask = maintask_map[domain_name]
            if isinstance(d_maintask, str):
                d_maintask = [d_maintask] * len(d_X)

            for x, y, maintask in zip(d_X, d_Y, d_maintask):
                X.append(x)
                Y.append(y)
                MT.append(maintask)
                D.append(domain_name)
        return X, Y, MT, D

    def __getitem__(self, index):
        if self.mode == 'neurotree2load':
            tree = self.X[index]
            tree.reset_state()
        else:
            tree = self.xmt2neurotree_map[self.D[index]](self.X[index], self.MT[index])
        return tree, self.Y[index], self.MT[index], self.D[index]

    def __len__(self):
        return len(self.X)

    def custom_collate_fn(self, data):
        x, y, mt, d = zip(*data)
        return BatchNeuroTree(x), y, mt, d


def createNeuroDataloader(
        neuroDataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False, drop_last=False,
        timeout=0, worker_init_fn=None,
        multiprocessing_context=None, generator=None,
        prefetch_factor=None,
        persistent_workers=False,
):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=neuroDataset.custom_collate_fn,
        sampler=sampler,
        batch_sampler=batch_sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    # prefetch_factor is only valid with worker processes; its default differs
    # across torch versions, so only forward it when explicitly usable.
    if num_workers > 0 and prefetch_factor is not None:
        kwargs['prefetch_factor'] = prefetch_factor
    return torch.utils.data.DataLoader(neuroDataset, **kwargs)
