import torch
import torch.nn as nn
import torch.nn.functional as F

from data_structures.batch_neurotree import BatchNeuroTree

class DeconvolutionalAssociationLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer


    def forward(self, batch_tree: BatchNeuroTree, h, root = False):
        if root:
            A_c = [None for _ in range(len(batch_tree.nodes))]
        else:
            A_c = batch_tree.getChildAdjacencyMatrix()

        if root:
            pass
        else:
            h = h.unsqueeze(1)

            node_count = batch_tree.get_child_count()

            if node_count > 0:
                indices = batch_tree.getIndices()
                batch_indices = torch.stack(indices, dim = 0)
                h = F.max_unpool2d(h, batch_indices, kernel_size=(node_count, 1))
            h = h.squeeze(1)

        features, hiddens = self.layer(A_c, h)
        # node_count = hiddens.shape[1]
        return features, hiddens
