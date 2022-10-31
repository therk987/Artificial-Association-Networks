import torch
from torch import nn
from torch.nn import functional as F
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from config.option import device

class Tabular2Vec(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.layer1 = nn.Linear(4,output_size)
        self.parameter_init()

    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)

    def forward(self, batch_tree:BatchNeuroTree):
        batch_x = batch_tree.get('x')
        x = torch.stack(batch_x).to(device, dtype = torch.float)
        h = self.layer1(x)
        h = F.normalize(h)
        return F.relu(h)

