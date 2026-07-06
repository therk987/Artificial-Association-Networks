import torch
import torch.nn as nn

from aan.models.encoder_cell.EGCN import EdgeGraphConvolutionalLayer
from aan.models.encoder_cell.GCN import GraphConvolutionalLayer
from aan.models.encoder_cell.GATs import GraphAttentionLayer
from aan.models.encoder_cell.RNN import RecurrentNeuralNetwork
from aan.models.encoder_cell.GRU import GatedRecurrentUnit
from aan.models.encoders.readout_max import MaxpoolReadoutLayer
from aan.data_structures.batch_neurotree import BatchNeuroTree

VERSION_ALIASES = {'egaau': 'egau'}
SUPPORTED_VERSIONS = ('ran', 'raan', 'gau', 'gaau', 'egau')


def build_cells(version, input_dim_with_bias, hidden_dim):
    """Map a model version name to its (recurrent cell, graph cell) pair.

    ran:  RNN + GCN      raan: RNN + GAT
    gau:  GRU + GCN      gaau: GRU + GAT     egau: GRU + edge-GCN
    """
    version = VERSION_ALIASES.get(version, version)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError('unknown version: {} (expected one of {})'.format(version, SUPPORTED_VERSIONS))

    if version in ('ran', 'raan'):
        rnn = RecurrentNeuralNetwork(input_dim_with_bias, hidden_dim)
    else:
        rnn = GatedRecurrentUnit(input_dim_with_bias, hidden_dim)

    if version in ('ran', 'gau'):
        gnn = GraphConvolutionalLayer(hidden_dim)
    elif version in ('raan', 'gaau'):
        gnn = GraphAttentionLayer(hidden_dim)
    else:
        gnn = EdgeGraphConvolutionalLayer(hidden_dim, 2)

    return rnn, gnn


class RecursiveAssociationNeuralNetworks(nn.Module):
    """RAN-series encoder: one shared cell applied at every neurotree level.

    Performs depth-first convolution (DFC, paper Algorithm 4): information
    propagates from the leaf nodes to the root, each node is computed exactly
    once, and revisits through additional parents reuse the stored hidden.
    """

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 feature_extraction_networks,
                 task_networks,
                 version='gaau'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.feature_extraction_networks = feature_extraction_networks
        self.task_networks = task_networks

        self.type_count = len(self.feature_extraction_networks.type_keys)
        self.input_dim_with_bias = self.input_dim + self.type_count

        self.register_buffer('zero_hiddens', torch.zeros(self.hidden_dim))

        self.rnn, self.gnn = build_cells(version, self.input_dim_with_bias, hidden_dim)
        self.readout = MaxpoolReadoutLayer()

        # Root dx/dh accumulation is only needed by the DFD (decoder) pass;
        # the AAN wrapper enables it when restoration networks are configured.
        self.store_deconv_inputs = False

    def deepFirstConvolution(self, all_batch_tree: BatchNeuroTree, level, node_level):
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
            # Children's hidden states are aggregated under the children's
            # relational information A_c stored in the current node.
            A_c = batch_tree.getChildAdjacencyMatrix()
            hiddens = self.gnn(A_c, child_hiddens)
            hiddens, indices = self.readout(hiddens, batch_tree.getChildCount())
            batch_tree.setIndices(indices)

        hiddens = self.rnn(node_features, hiddens)

        batch_tree.setHiddens(hiddens.squeeze(1))
        if level == 0 and self.store_deconv_inputs:
            batch_tree.set_xh(node_features.squeeze(1), hiddens.squeeze(1))

        if node_level:
            self.task_networks(hiddens, batch_tree)

    def propagation(self, batch_tree: BatchNeuroTree, node_level=False):
        self.deepFirstConvolution(batch_tree, 0, node_level)
        return batch_tree.getHiddens()

    def forward(self, batch_tree: BatchNeuroTree, node_level=False):
        return self.propagation(batch_tree, node_level)
