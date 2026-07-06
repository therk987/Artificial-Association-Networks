import os
import pickle

import torch
from torch import nn
import torch.nn.functional as F

from aan.data_structures.batch_neurotree import BatchNeuroTree

DEFAULT_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'datas', 'code', 'sort', 'result', 'sort_ast_neurotree_cfg_vocab.pkl')


class Ast2VectorForPython(nn.Module):
    """AST node-type embedding psi_ast (paper, Section III-D, Experiment 4).

    Embeds each AST node type into ``input_dim`` dimensions and zero-pads to
    ``output_dim``. Also writes the vocabulary index of each node type back
    onto the node as ``label`` for the parent-prediction subtask.
    """

    def __init__(self, input_dim=25, output_dim=128, vocab_path=DEFAULT_VOCAB_PATH):
        super().__init__()

        self.keys = self.load_vocab(vocab_path)
        self.embedding = nn.Embedding(len(self.keys), input_dim, padding_idx=0)

        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim != output_dim:
            self.register_buffer('zero_vector', torch.zeros(output_dim - input_dim))

        self.NULL_INDEX = self.keys['<unk>']

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, 'rb') as f:
            python_ast_classes = pickle.load(f)
        ast_vocab = {'<unk>': 0}
        for key in python_ast_classes:
            ast_vocab[key] = len(ast_vocab)
        return ast_vocab

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.embedding.weight.device
        ast_x = batch_tree.getX()
        ast_y = batch_tree.getY()

        indices = [self.keys.get(e, self.NULL_INDEX) for e in ast_x]
        embedded = self.embedding(torch.tensor(indices, device=device, dtype=torch.long))
        embedded = torch.tanh(F.normalize(embedded))
        batch_tree.set('x', embedded)

        targets = [self.keys.get(e, self.NULL_INDEX) for e in ast_y]
        batch_tree.set('label', targets)

        if self.input_dim != self.output_dim:
            embedded = torch.cat(
                [embedded, self.zero_vector.unsqueeze(0).repeat(embedded.shape[0], 1)], dim=1)
        return embedded


class Constant2Vec(nn.Module):
    """Constant-value scale embedding psi_constant (paper, Section III-D).

    Constant values are pre-quantized to the range around [-1, 2]; three FC
    layers lift the scalar to ``out_features`` dimensions.
    """

    def __init__(self, out_features=128):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, batch_tree: BatchNeuroTree):
        device = self.fc1.weight.device
        xs = [x if torch.is_tensor(x) else torch.tensor([float(x)]) for x in batch_tree.getX()]
        h = torch.stack(xs).to(device, dtype=torch.float).view(-1, 1)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        h = F.normalize(h)
        return F.relu(h)
