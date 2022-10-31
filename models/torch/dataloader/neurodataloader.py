import copy
from torch.utils.data import Dataset

import torch
from config.option import device


class NeuroDataset(Dataset):
    """ Diabetes dataset."""  # Initialize your data, download, etc.

    def __init__(self, domain_x, domain_y, domain_st, domain_mt):
        domains = domain_x.keys()
        X = []
        Y = []
        D = []
        ST = []
        MT = []

        for domain_name in domains:
            d_X = domain_x[domain_name]
            d_Y = domain_y[domain_name]
            if type(domain_st) == str:
                d_ST = [domain_st for _ in range(len(d_X))]
            elif type(domain_st[domain_name]) == str:
                d_ST = [domain_st[domain_name] for _ in range(len(d_X))]
            else:
                d_ST = domain_st[domain_name]

            if type(domain_mt) == str:
                d_MT = [domain_mt for _ in range(len(d_X))]
            elif type(domain_mt[domain_name]) == str:
                d_MT = [domain_mt[domain_name] for _ in range(len(d_X))]
            else:
                d_MT = domain_mt[domain_name]

            for x, y, st, mt in zip(d_X, d_Y, d_ST, d_MT):
                if domain_name == 'class':
                    # f_n, x_nn = x
                    # X.append(x_nn)
                    X.append(x)
                    Y.append(y)
                    D.append(domain_name)
                    ST.append(st)
                    MT.append(mt)
                else:
                    X.append(x)
                    Y.append(y)
                    D.append(domain_name)
                    ST.append(st)
                    MT.append(mt)

        self.X = X
        self.Y = Y
        self.D = D
        self.ST = ST
        self.MT = MT
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.D[index], self.ST[index], self.MT[index], self.Y[index]

    def __len__(self):
        return self.len

    def custom_collate_fn(self, data):
        # to Dataloader with BatchGraphTree
        x, d, st, mt, y = zip(*data)
        x2 = copy.deepcopy(x)
        return x2, d, st, mt, y
        # inputs, domains, tasks, labels = zip(*data)
        # batch_x = torch.stack(inputs, dim = 0).to(device, dtype=torch.float)
        # batch_y = torch.stack(labels, dim = 0).to(device, dtype=torch.long)
        # return batch_x, domains, tasks, batch_y




def createNeuroDataloader(
                            neuroDataset,
                            batch_size = 1,
                            shuffle = False,
                            sampler = None,
                            batch_sampler = None,
                            num_workers = 0,
                            pin_memory: bool = False, drop_last: bool = False,
                            timeout: float = 0, worker_init_fn = None,
                            multiprocessing_context=None, generator=None,
                            prefetch_factor: int = 2,
                            persistent_workers: bool = False,
                          ):


    neuroDataloader = torch.utils.data.DataLoader(neuroDataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=neuroDataset.custom_collate_fn,
                                              sampler=sampler,
                                              batch_sampler=batch_sampler,
                                              pin_memory = pin_memory,
                                              drop_last = drop_last,
                                              timeout = timeout,
                                              worker_init_fn = worker_init_fn,
                                              multiprocessing_context = multiprocessing_context,
                                              generator = generator,
                                              prefetch_factor = prefetch_factor,
                                              persistent_workers = persistent_workers)

    return neuroDataloader



#
#
# def createNeuroDataloader(
#         domain_x, domain_y,
#         domain_st, domain_mt,
#                             batch_size = 1,
#                             shuffle = False,
#                             sampler = None,
#                             batch_sampler = None,
#                             num_workers = 0,
#                             pin_memory: bool = False, drop_last: bool = False,
#                             timeout: float = 0, worker_init_fn = None,
#                             multiprocessing_context=None, generator=None,
#                             prefetch_factor: int = 2,
#                             persistent_workers: bool = False,
#                           ):
#
#     neuroDataset = NeuroDataset(domain_x, domain_y, domain_st, domain_mt)
#
#
#
#     neuroDataloader = torch.utils.data.DataLoader(neuroDataset,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=num_workers,
#                                               collate_fn=neuroDataset.custom_collate_fn,
#                                               sampler=sampler,
#                                               batch_sampler=batch_sampler,
#                                               pin_memory = pin_memory,
#                                               drop_last = drop_last,
#                                               timeout = timeout,
#                                               worker_init_fn = worker_init_fn,
#                                               multiprocessing_context = multiprocessing_context,
#                                               generator = generator,
#                                               prefetch_factor = prefetch_factor,
#                                               persistent_workers = persistent_workers)
#
#     return neuroDataset, neuroDataloader
#
#
