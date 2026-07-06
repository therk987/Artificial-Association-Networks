from collections import defaultdict

import torch
from torch import nn

from aan.data_structures.batch_neurotree import BatchNeuroTree


class MultiRestorationConnector(nn.Module):
    """Multi-feature restoration networks Psi^-1 (paper, Appendix C).

    The decoding counterpart of MultiExtractionConnector: groups nodes by
    domain, runs the domain's restoration network, and restores batch order.
    Used by the DFD (deconvolution) pass for autoencoder-style models.
    """

    def __init__(self, feature_dim, feature_restoration_dict):
        super().__init__()
        self.feature_restoration_networks = nn.ModuleDict(feature_restoration_dict)
        self.type_keys = list(feature_restoration_dict.keys())
        self.feature_dim = feature_dim

        type_count = len(self.type_keys)
        self.register_buffer('type_bias', torch.eye(type_count))
        self.register_buffer('layer_input', torch.zeros(1, feature_dim))
        self.register_buffer('layer_input_bias', torch.zeros(1, type_count))

    def loss(self, domain_name, conv_x, deconv_x):
        return self.feature_restoration_networks[domain_name].loss(conv_x, deconv_x)

    def typeDecoding(self, dx, batch_tree):
        batch_tree_dict = defaultdict(list)
        batch_indices_to_type_indices = {}
        batch_output_dict = {}
        batch_type_label_dict = {}

        for i in range(len(batch_tree.nodes)):
            domain_name = batch_tree.get_i('t_d', i)
            batch_indices_to_type_indices[i] = (domain_name, len(batch_tree_dict[domain_name]))
            batch_tree_dict[domain_name].append(batch_tree.nodes[i])

        for domain_name, domain_tree_list in batch_tree_dict.items():
            if domain_name is None:
                type_features = [None] * len(domain_tree_list)
                type_labels = [None] * len(domain_tree_list)
            else:
                domain_batch_tree = BatchNeuroTree(domain_tree_list)
                type_features = self.feature_restoration_networks[domain_name](domain_batch_tree)
                type_labels = [self.type_keys.index(domain_name)] * len(domain_tree_list)

            batch_output_dict[domain_name] = type_features
            batch_type_label_dict[domain_name] = type_labels

        outputs = []
        for i, (domain_name, domain_index) in batch_indices_to_type_indices.items():
            batch_tree.set_i('decoder-pred', batch_output_dict[domain_name][domain_index], i)
            batch_tree.set_i('domain_label', batch_type_label_dict[domain_name][domain_index], i)
            outputs.append(batch_output_dict[domain_name][domain_index])
        return outputs

    def forward(self, dx, batch_tree):
        return self.typeDecoding(dx, batch_tree), []
