import torch
import torch.nn as nn

from aan.models.encoder_cell.GCN import GraphConvolutionalLayer
from aan.models.encoder_cell.GATs import GraphAttentionLayer
from aan.models.encoder_cell.RNN import RecurrentNeuralNetwork
from aan.models.encoders.readout_max import MaxpoolReadoutLayer
from aan.data_structures.batch_neurotree import BatchNeuroTree

SUPPORTED_VERSIONS = ('lan', 'laan')


class LevelAssociationNeuralNetworks(nn.Module):
    """LAN-series encoder: separate weights per neurotree level.

    Same DFC propagation as the RAN series, but the cell at level ``lv`` has
    its own parameters, which makes the FC-layer/MLP structure a special case
    (paper, Section III-G2). ``max_lv`` bounds the tree depth that can be
    processed.
    """

    def __init__(self, input_dim, hidden_dim,
                 feature_extraction_networks, task_networks, max_lv,
                 version='lan'):
        super().__init__()

        if version not in SUPPORTED_VERSIONS:
            raise ValueError('unknown version: {} (expected one of {})'.format(version, SUPPORTED_VERSIONS))

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_lv = max_lv

        self.feature_extraction_networks = feature_extraction_networks
        self.task_networks = task_networks

        self.type_count = len(self.feature_extraction_networks.type_keys)
        self.input_dim_with_bias = self.input_dim + self.type_count

        self.register_buffer('zero_hiddens', torch.zeros(self.hidden_dim))

        gnn_cls = GraphConvolutionalLayer if version == 'lan' else GraphAttentionLayer
        self.rnn = nn.ModuleList(
            [RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim) for _ in range(max_lv)])
        self.gnn = nn.ModuleList(
            [gnn_cls(self.hidden_dim) for _ in range(max_lv)])

        self.readout = MaxpoolReadoutLayer()

    def deepFirstConvolution(self, all_batch_tree: BatchNeuroTree, level, node_level):
        if level >= self.max_lv:
            raise ValueError(
                'neurotree depth exceeds max_lv={}; increase max_lv or use the RAN series'.format(self.max_lv))

        all_batch_tree.visit()
        batch_tree = all_batch_tree.get_no_calculated()
        if len(batch_tree.nodes) == 0:
            return

        max_child_count = batch_tree.get_child_count()
        for i in range(max_child_count):
            batch_child, _ = batch_tree.get_child(number=i)
            self.deepFirstConvolution(batch_child, level + 1, node_level)

        node_features = self.feature_extraction_networks(batch_tree)

        if max_child_count == 0:  # leaf nodes
            hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
        else:
            child_hiddens = batch_tree.get_child_hiddens(self.zero_hiddens)
            A_c = batch_tree.getChildAdjacencyMatrix()
            hiddens = self.gnn[level](A_c, child_hiddens)
            hiddens, indices = self.readout(hiddens, batch_tree.getChildCount())
            batch_tree.setIndices(indices)

        hiddens = self.rnn[level](node_features, hiddens)

        batch_tree.setHiddens(hiddens.squeeze(1))
        if level == 0:
            batch_tree.set_xh(node_features.squeeze(1), hiddens.squeeze(1))

        if node_level:
            self.task_networks(hiddens, batch_tree)

    def propagation(self, batch_tree: BatchNeuroTree, node_level=False):
        self.deepFirstConvolution(batch_tree, 0, node_level)
        return batch_tree.getHiddens()

    def forward(self, batch_tree: BatchNeuroTree, node_level=False):
        return self.propagation(batch_tree, node_level)
