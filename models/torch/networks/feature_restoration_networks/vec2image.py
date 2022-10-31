import torch
from torch import nn
import torch.nn.functional as F
import pickle

class Deconv_LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.conv1 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.parameter_init()

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(0, 1, 2))

    def loss(self, association_node):
        if 'conv_x' in association_node.__dict__.keys():
            conv_x = association_node.conv_x
            deconv_x = association_node.deconv_x
            if conv_x is None or deconv_x is None:
                return 0
            else:
                return - self.gaussian_likelihood(conv_x, self.log_scale, deconv_x)

    #         return F.binary_cross_entropy(deconv_x, conv_x)

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree):
        batch_h = batch_tree.get('dh')
        dh = torch.stack(batch_h)

        x = dh
        x = x.view(-1, 128, 1, 1)
        x = self.conv3(x)
        x = F.tanh(x)
        x = self.conv2(x)
        x = F.tanh(x)
        x = self.conv1(x)
        x = F.tanh(x)
        batch_tree.set('pred', x)
        return x

