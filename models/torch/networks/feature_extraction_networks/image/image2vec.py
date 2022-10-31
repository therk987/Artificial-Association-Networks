import torch
from torch import nn
from torch.nn import functional as F
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from config.option import device

from torch import nn
import torch.nn.functional as F


# class LeNet_5(nn.Module):
#     def __init__(self):
#         super(LeNet_5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
#         self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
#         self.fc1 = nn.Linear(120, 84)
#         self.fc2 = nn.Linear(84, 10)
#         self.parameter_init()
#
#     def parameter_init(self):
#         for param in self.parameters():
#             if len(param.shape) > 1:
#                 nn.init.xavier_uniform_(param, gain=1.414)
#
#     def forward(self, x):
#         x = F.tanh(self.conv1(x))
#         x = F.avg_pool2d(x, 2, 2)
#         x = F.tanh(self.conv2(x))
#         x = F.avg_pool2d(x, 2, 2)
#         x = F.tanh(self.conv3(x))
#         x = x.view(-1, 120)
#         x = F.tanh(self.fc1(x))
#         x = self.fc2(x)
#         return F.softmax(x, dim=1)


class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree: BatchNeuroTree):
        batch_x = batch_tree.get('x')
        x = torch.stack(batch_x).to(device)

        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 2, 2)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        zeros = torch.zeros(x.shape[0], 8).to(device)
        out = torch.cat([x, zeros], dim=-1)
        return out

    def setting_pretrained_model(self, model):
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3

#
# class LeNet_5(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
#
#         self.parameter_init()
#
#     def parameter_init(self):
#         for param in self.parameters():
#             if len(param.shape) > 1:
#                 nn.init.xavier_uniform_(param, gain=1.414)
#
#     def forward(self, batch_tree:BatchNeuroTree):
#         batch_x = batch_tree.get('x')
#         x = torch.stack(batch_x).to(device)
#         x = self.conv1(x)
#         x = F.tanh(x)
#         x = self.conv2(x)
#         x = F.tanh(x)
#         x = self.conv3(x)
#         x = F.tanh(x)
#         x = x.view(-1, 128)
#         return x
#
