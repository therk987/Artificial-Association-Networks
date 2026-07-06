import torch
from torch import nn
import torch.nn.functional as F

from aan.data_structures.batch_neurotree import BatchNeuroTree


class LeNet_5(nn.Module):
    """Image feature extractor psi_image based on LeNet-5 (paper, Table XI).

    Produces a 120-dim conv feature zero-padded to ``output_dim``.
    ``conv3_kernel`` depends on the input resolution: 5 for 32x32 inputs,
    4 for raw 28x28 MNIST.
    """

    def __init__(self, output_dim=128, conv3_kernel=5):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=conv3_kernel, stride=1)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.conv1.weight.device
        x = torch.stack(batch_tree.getX()).to(device, dtype=torch.float)

        x = torch.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 120)
        zeros = torch.zeros(x.shape[0], self.output_dim - 120, device=device)
        return torch.cat([x, zeros], dim=-1)

    def setting_pretrained_model(self, model):
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
