import torch
import torch.nn as nn
import torch.nn.functional as F
from config.option import device

class FullyConnectedLayer_MFCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(40, 128)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree):

        batch_x = batch_tree.get('x')
        batch_x = torch.stack(batch_x, dim = 0).to(device)


        x = self.fc_in(batch_x)
        x = F.normalize(x)
        x = F.relu(x)
        return x
