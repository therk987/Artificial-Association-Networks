import torch
from torch import nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
from config.option import device


import torch
from torch import nn
import torch.nn.functional as F
import pickle


class Ast2Vector(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, output_dim):
        super().__init__()

        root = os.path.dirname(os.path.realpath(__file__))
        vector_path = '/'.join(root.split('/')[:-4]) + '/pretrained/vectors-node/vectors-node.pkl'

        self.vectors, self.keys = self.loadData(vector_path)

        self.output_dim = output_dim
        input_dim = 30

        #         self.embedding = nn.Embedding(len(self.keys), input_dim, padding_idx = 1)
        freeze = True

        self.vectors = np.concatenate([self.vectors, np.zeros((1, 30))], axis=0)

        ast_vectors = torch.FloatTensor(self.vectors)

        self.embedding = nn.Embedding.from_pretrained(
            ast_vectors, freeze=True).to(device)

        self.embedding.weight.data.copy_(torch.Tensor(self.vectors))
        self.NULL_INDEX = ast_vectors.shape[0] - 1

        # self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        self.W = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)



    def loadData(self, name):

        with open(name, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()

        return data

    def forward(self, batch_tree):
        ast_x = batch_tree.get('x')
        indices = []
        for e in ast_x:
            if e in self.keys.keys():
                indices.append(self.keys[e])
            else:
                #                 print(e,'???')
                indices.append(self.NULL_INDEX)

        embedded = self.embedding(torch.Tensor(indices).to(device, dtype=torch.long))
        embedded = F.normalize(embedded)
        embedded = F.tanh(embedded)

        batch_tree.set('x', embedded)
        Wh = self.W(embedded)
        # Wh = torch.matmul(embedded, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        h_prime = F.normalize(Wh)
        return F.relu(h_prime)

    def __repr__(self):
        return self.__class__.__name__
        # return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#
# class Ast2Vector(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, output_dim):
#         super().__init__()
#
#         root = os.path.dirname(os.path.realpath(__file__))
#         vector_path = '/'.join(root.split('/')[:-3]) + '/pretrained/vectors-node/vectors-node.pkl'
#
#         self.vectors, self.keys = self.loadData(vector_path)
#
#         self.output_dim = output_dim
#         input_dim = 30
#
#         self.embedding = nn.Embedding(len(self.keys), input_dim, padding_idx=1)
#         self.embedding.weight.data.copy_(torch.Tensor(self.vectors).to(device))
#
#         self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim))).to(device)
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#     def loadData(self, name):
#         with open(name, 'rb') as f:
#             u = pickle._Unpickler(f)
#             u.encoding = 'latin1'
#             data = u.load()
#
#         return data
#
#     def forward(self, batch_tree):
#         indices = []
#         for e in batch_tree.get('x'):
#             indices.append(self.keys[e])
#
#         embedded = self.embedding(torch.Tensor(indices).to(dtype=torch.long))
#         Wh = torch.matmul(embedded, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         h_prime = F.normalize(Wh)
#         return F.relu(h_prime)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
#
# root = os.path.dirname(os.path.realpath(__file__))
# print('/'.join(root.split('/')[:-4]))
# vector_path = '/'.join(root.split('/')[:-4]) + '/pretrained/vectors-node/vectors-node.pkl'
# print(vector_path)