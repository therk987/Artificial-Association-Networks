from collections import defaultdict

import torch
from torch import nn

from aan.data_structures.batch_neurotree import BatchNeuroTree


class MultiSubTaskConnector(nn.Module):
    """Node-level subtask networks Phi_s (paper, Section III-E).

    Groups nodes by subtask name, runs each subtask network on its group,
    stores the predictions back on the nodes, and restores batch order.
    Nodes with subtask ``None`` produce ``None`` outputs.
    """

    def __init__(self, hidden_dim, task_networks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_networks = nn.ModuleDict(task_networks)

    def forward(self, h_root, batchNeuroTree):
        return self.multi_tasking(h_root, batchNeuroTree)

    def loss(self, taskname, domains, preds, targets):
        return self.task_networks[taskname].loss(domains, preds, targets)

    def multi_tasking(self, h_root, allBatchNeuroTree: BatchNeuroTree):
        batch_tree_dict = defaultdict(list)
        batch_hidden_dict = defaultdict(list)
        batch_indices_to_task_indices = {}
        batch_output_dict = {}

        for i in range(len(allBatchNeuroTree.nodes)):
            task_name = allBatchNeuroTree.getSubtask(i)
            batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
            batch_tree_dict[task_name].append(allBatchNeuroTree.nodes[i])
            batch_hidden_dict[task_name].append(h_root[i])

        for task_name, task_tree_list in batch_tree_dict.items():
            batch_neurotree = BatchNeuroTree(task_tree_list)
            if task_name is not None:
                task_output = self.task_networks[task_name](
                    batch_hidden_dict[task_name],
                    batch_neurotree,
                )
                batch_neurotree.setPredicts(task_output)
            else:
                task_output = [None] * len(task_tree_list)

            batch_output_dict[task_name] = task_output

        return [
            batch_output_dict[task_name][task_index]
            for task_name, task_index in batch_indices_to_task_indices.values()
        ]
