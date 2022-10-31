import torch
import torch.nn as nn
import pickle

class DecAst2Vector(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W = nn.Linear(self.hidden_dim, self.output_dim)
        self.parameter_init()




    def parameter_init(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param, gain=1.414)




    def forward(self, batch_tree):
        batch_h = batch_tree.get('dh')
        dh = torch.stack(batch_h)
        Wh = self.W(dh)
        # Wh = torch.matmul(dh, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print(Wh.shape)
        # batch_tree.set('pred', Wh.squeeze(1))
        return Wh

    def __repr__(self):
        return self.__class__.__name__
