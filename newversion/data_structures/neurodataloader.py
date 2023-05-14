import copy
from torch.utils.data import Dataset
import torch
from data_structures.batch_neurotree import BatchNeuroTree
from data_structures.neuronode import NeuroNode


class NeuroDataset(Dataset):
    """ Diabetes dataset."""  # Initialize your data, download, etc.


    def neurotree2load(self, x_map, y_map, maintask_map, xmt2neurotree_map):

        domains = x_map.keys()
        X = []
        Y = []
        D = []
        MT = []


        for domain_name in domains:
            d_X = x_map[domain_name]
            d_Y = y_map[domain_name]
            d_xmt2neurotree = xmt2neurotree_map[domain_name]
            d_maintask = maintask_map[domain_name]
            if type(d_maintask) == str:
                d_maintask = [d_maintask for _ in range(len(d_X))]

            for x, y, maintask in zip(d_X, d_Y, d_maintask):
                x_neurotree = d_xmt2neurotree(x, maintask)
                X.append(x_neurotree)
                Y.append(y)
                MT.append(maintask)
                D.append(domain_name)
        return X, Y, MT, D


    def load2neurotree(self, x_map, y_map, maintask_map):
        domains = x_map.keys()
        X = []
        Y = []
        D = []
        MT = []

        for domain_name in domains:
            d_X = x_map[domain_name]
            d_Y = y_map[domain_name]
            d_maintask = maintask_map[domain_name]
            if type(d_maintask) == str:
                d_maintask = [d_maintask for _ in range(len(d_X))]

            for x, y, maintask in zip(d_X, d_Y, d_maintask):
                X.append(x)
                Y.append(y)
                MT.append(maintask)
                D.append(domain_name)
        return X, Y, MT, D



    def __init__(self, x_map, y_map, maintask_map, xmt2neurotree_map, mode = 'load2neurotree'):
        self.xmt2neurotree_map = xmt2neurotree_map

        if mode == 'neurotree2load':
            self.X, self.Y, self.MT, self.D = self.neurotree2load(x_map, y_map, maintask_map, xmt2neurotree_map)
        elif mode == 'load2neurotree':
            self.X, self.Y, self.MT, self.D = self.load2neurotree(x_map, y_map, maintask_map)
        else:
            assert mode + 'mode not founded'

        self.mode = mode
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.xmt2neurotree_map[self.D[index]](self.X[index], self.MT[index]),\
               self.Y[index], \
               self.MT[index],\
               self.D[index]

    def __len__(self):
        return self.len

    def custom_collate_fn(self, data):
        # to Dataloader with BatchGraphTree
        x, y, mt, d = zip(*data)
        # x2 = copy.deepcopy(x)
        return BatchNeuroTree(x), y, mt, d
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


    return torch.utils.data.DataLoader(neuroDataset,
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
                                              persistent_workers = persistent_workers

                                       )



