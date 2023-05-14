import torch.nn as nn
import torch.nn.functional as F
from data_structures.batch_neurotree import BatchNeuroTree

class AssociationLayer(nn.Module):
    def __init__(self, layer, task_networks):
        super().__init__()
        self.layer = layer
        self.task_networks = task_networks

    def forward(self, batch_tree: BatchNeuroTree, x_1, h_1, root=False):
        if root:
            A_c = [None for _ in range(len(batch_tree.nodes))]
        else:
            A_c = batch_tree.getChildAdjacencyMatrix()

        hidden, task_hidden = self.layer(A_c, x_1, h_1)
        node_count = hidden.shape[1]
        hidden = hidden.unsqueeze(1)

        hidden, indices = F.max_pool2d(hidden, kernel_size=(node_count, 1), return_indices=True)
        hidden = hidden.squeeze(1)

        if root:
            batch_tree.setHiddens(task_hidden.squeeze(1))
            # task_out = self.task_networks(association_node.t_t, task_hidden.squeeze(1))
            # association_node.pred = task_out

        else:
            batch_tree.setIndices(indices)
        # if association_node.t_t:
            child_count = batch_tree.get_child_count()
            for i, t_hidden in zip(range(child_count), task_hidden.split(1, dim=1)):
                i_child_batch, child_idx = batch_tree.get_child(i)
                i_child_batch.setHiddens(t_hidden.squeeze(1))
        return hidden
