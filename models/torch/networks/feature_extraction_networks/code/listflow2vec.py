import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from config.option import device


class ListFlowCNN(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, output_dim):
        super().__init__()

        self.conv_seq = [
            nn.Conv2d(1, 32, kernel_size=(1 ,7), stride=(1, 7), padding = (0 ,7)).to(device),
            nn.Conv2d(32, 64, kernel_size=(1 ,2), stride=1, padding = (0 ,1)).to(device),
            nn.Conv2d(64, output_dim, kernel_size=(1 ,2), stride=1, padding = (0 ,1)).to(device)
        ]
        self.parameter_init()
        self.return_indices = True

    def pad_sequence(self, batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        clamped_batch = batch.clamp(-7 ,17)
        return minmax(clamped_batch, -7, 17, -1, 1)


    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)


    def forward(self, batch_tree):
        batch_x = batch_tree.get('x')
        x = [ torch.Tensor(_) for _ in  batch_x ]
        x = self.pad_sequence(x).to(device)
        batch_tree.set('conv_x', x)

        x = x.view(x.shape[0], 1, 1, x.shape[-1])
        pooling_indices = defaultdict(list)
        for i, conv in enumerate(self.conv_seq):
            x = conv(x)
            x = F.normalize(x)
            x = F.relu(x)

            if i == len(self.conv_seq) - 1:
                x_shape = x.shape[-2:]
                x, indices = F.adaptive_max_pool2d(x, 1, return_indices = True)
            #                 pooling_indices.append((indices, adaptive_shape))
            else:
                x_shape = x.shape[-2:]
                x, indices = F.max_pool2d(x, kernel_size = (1 ,2), return_indices = True)
            #                 pooling_indices.append((indices, output_shape))

            if self.return_indices:
                batch_tree.set('vf_indices ' +str(i), indices)
                batch_tree.set('vf_shape ' +str(i), [x_shape for _ in range(len(indices))])
        x = x.view(-1, 128)

        return x


    def __repr__(self):
        return self.__class__.__name__
