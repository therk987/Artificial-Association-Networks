import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxunpoolReadoutLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, batch_tree):
        # hidden = hidden.unsqueeze(1)


        #             print(h.shape, len(association_node.C))
        node_count = batch_tree.get_child_count()

        if node_count > 0:
            indices = batch_tree.get('indices')
            batch_indices = torch.stack(indices, dim=0)
            # print(hidden.shape, node_count, batch_indices.shape)

            # print(batch_indices.shape, h.shape)
            h = F.max_unpool2d(hidden, batch_indices, kernel_size=(node_count, 1))
        h = h.squeeze(1)

        return h
