import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from config.option import device

class DecListFlowCNN(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, output_dim):
        super().__init__()

        self.deconv_seq = [
            nn.ConvTranspose2d(output_dim, 64, kernel_size=(1 ,2), stride=1, padding = (0 ,1)).to(device),
            nn.ConvTranspose2d(64, 32, kernel_size=(1 ,2), stride=1, padding = (0 ,1)).to(device),
            nn.ConvTranspose2d(32, 1, kernel_size=(1 ,7), stride=(1, 7), padding = (0 ,7)).to(device),
        ]
        self.parameter_init()


    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)


    def forward(self, x_hat, batch_tree):
        x = x_hat.view(x_hat.shape[0], -1, 1, 1)

        for i, deconv in enumerate(self.deconv_seq):
            indices = batch_tree.get('vf_indices ' +str(len(self.deconv_seq) - i - 1))
            shape = batch_tree.get('vf_shape ' +str(len(self.deconv_seq) - i - 1))

            indices = torch.stack(indices, dim = 0)
            if i == 0:


                x = F.max_unpool2d(x, indices = indices, kernel_size = shape[0], output_size = shape[0])

            else:
                x = F.max_unpool2d(x, indices = indices, kernel_size = (1 ,2), output_size = shape[0])


            x = deconv(x)
            if i != len(self.deconv_seq) - 1:
                x = F.normalize(x)
                x = F.relu(x)

        x = x.view(x.shape[0], -1)
        batch_tree.set('deconv_x', x)
        return x

    def __repr__(self):
        return self.__class__.__name__
