from data_structures.batch_neurotree import BatchNeuroTree

from collections import defaultdict
from torch import nn
import torch

class MultiMainTaskConnector(nn.Module):
    def __init__(self, hidden_dim, task_networks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_networks = task_networks

        for modelname in self.task_networks.keys():
            self.__setattr__(f'{modelname}', task_networks[modelname])


        self._reset_params()
        # self.batch_tools = BatchAssociationTree()
    def _reset_params(self):
        for seq_layers in self.task_networks.values():
            for i, net in enumerate(seq_layers.parameters()):
                if type(net) == nn.Linear:
                    torch.nn.init.xavier_uniform_(net.weight.data)



    def forward(self, h_root, batchNeuroTree: BatchNeuroTree, tasks):
        return self.multi_tasking(h_root, batchNeuroTree, tasks)

    def multi_tasking(self, h_root, allBatchNeuroTree:BatchNeuroTree, tasks):
        batch_tree_dict = defaultdict(list)
        batch_hidden_dict = defaultdict(list)
        batch_indices_to_task_indices = defaultdict(tuple)
        batch_output_dict = defaultdict(torch.Tensor)
        for i in range(len(allBatchNeuroTree.nodes)):
            task_name = tasks[i]
            batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
            batch_tree_dict[task_name].append(allBatchNeuroTree.nodes[i])
            batch_hidden_dict[task_name].append(h_root[i])
        for task_name, task_tree_list in batch_tree_dict.items():
            batch_neurotree = BatchNeuroTree(task_tree_list)
            batch_hiddens = torch.stack(batch_hidden_dict[task_name], dim=0)
            task_output = self.task_networks[task_name](
                batch_hiddens,
                batch_neurotree
            )  # Leaf Convolution
            batch_output_dict[task_name] = task_output
        outputs = []
        for i, (task_name, task_index) in batch_indices_to_task_indices.items():
            outputs.append(batch_output_dict[task_name][task_index])
        return outputs

    def loss(self, taskname, domains, preds, targets):
        return self.task_networks[taskname].loss(domains, preds, targets)



    def dict_sum(self, dict_1, dict_2):
        #         dict_1 << dict_2
        for key1 in dict_2.keys():
            for key2, value in dict_1[key1].items():
                dict_1[key2] += value
        return dict_1



