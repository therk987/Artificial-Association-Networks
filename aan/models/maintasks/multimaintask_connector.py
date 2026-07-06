from collections import defaultdict

import torch
from torch import nn

from aan.data_structures.batch_neurotree import BatchNeuroTree


class MultiMainTaskConnector(nn.Module):
    """Main task networks Phi_m applied to the root hidden state
    (paper, Section III-E, eq. 21).

    Groups samples by main task name, runs each task network on its group,
    and restores batch order.
    """

    def __init__(self, hidden_dim, task_networks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_networks = nn.ModuleDict(task_networks)

    def forward(self, h_root, batchNeuroTree: BatchNeuroTree, tasks):
        return self.multi_tasking(h_root, batchNeuroTree, tasks)

    def loss(self, taskname, domains, preds, targets):
        return self.task_networks[taskname].loss(domains, preds, targets)

    def multi_tasking(self, h_root, allBatchNeuroTree: BatchNeuroTree, tasks):
        # single-task fast path (the common case): run the head once on the
        # whole batch and return the batched tensor — no per-row gathering
        first_task = tasks[0]
        if all(t == first_task for t in tasks):
            if not torch.is_tensor(h_root):
                h_root = torch.stack(list(h_root), dim=0)
            return self.task_networks[first_task](h_root, allBatchNeuroTree)

        batch_tree_dict = defaultdict(list)
        batch_hidden_dict = defaultdict(list)
        batch_indices_to_task_indices = {}
        batch_output_dict = {}

        for i in range(len(allBatchNeuroTree.nodes)):
            task_name = tasks[i]
            batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
            batch_tree_dict[task_name].append(allBatchNeuroTree.nodes[i])
            batch_hidden_dict[task_name].append(h_root[i])

        for task_name, task_tree_list in batch_tree_dict.items():
            batch_neurotree = BatchNeuroTree(task_tree_list)
            batch_hiddens = torch.stack(batch_hidden_dict[task_name], dim=0)
            batch_output_dict[task_name] = self.task_networks[task_name](
                batch_hiddens,
                batch_neurotree,
            )

        return [
            batch_output_dict[task_name][task_index]
            for task_name, task_index in batch_indices_to_task_indices.values()
        ]
