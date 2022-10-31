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


class Ast2VectorForPython(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.embedding, self.keys = self.loadData()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim != output_dim:
            self.zero_vector = torch.zeros(self.output_dim - self.input_dim).to(device)

        self.NULL_INDEX = self.keys['<unk>']


    def loadData(self):

        root = os.path.dirname(os.path.realpath(__file__))
        vector_path = '/'.join(root.split('/')[:-5]) + '/dataset/code/codeclone/sort/result/sort_ast_neurotree_cfg_vocab.pkl'
        with open(vector_path, 'rb') as f:
            python_ast_classes = pickle.load(f)
        ast_vocab = {}

        ast_vocab['<unk>'] = 0
        for i, key in enumerate(python_ast_classes):
            ast_vocab[key] = len(ast_vocab.keys())


        embedding = nn.Embedding(len(ast_vocab.keys()), 25, padding_idx=0)


        return embedding, ast_vocab

    def forward(self, batch_tree):
        ast_x = batch_tree.get('x')
        ast_y = batch_tree.get('y')
        indices = []
        for e in ast_x:
            try:
                indices.append(self.keys[e])
            except:
                indices.append(self.keys['<unk>'])
        embedded = self.embedding(torch.Tensor(indices).to(device, dtype=torch.long))
        embedded = F.normalize(embedded)
        embedded = F.tanh(embedded)
        batch_tree.set('x', embedded)

        targets = []
        for e in ast_y:
            try:
                targets.append(self.keys[e])
            except:
                targets.append(self.keys['<unk>'])
        batch_tree.set('label', targets)

        if self.input_dim != self.output_dim:
            embedded = torch.cat([embedded, self.zero_vector.unsqueeze(0).repeat(embedded.shape[0], 1)], dim = 1)

        return embedded
        # Wh = self.layer1(embedded)
        # h_prime = F.normalize(Wh)
        # return F.relu(h_prime)


