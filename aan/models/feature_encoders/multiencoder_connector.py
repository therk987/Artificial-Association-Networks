from collections import defaultdict

import torch
from torch import nn

from aan.data_structures.batch_neurotree import BatchNeuroTree


class MultiExtractionConnector(nn.Module):
    """Multi-feature extraction networks Psi (paper, Section III-A, Alg. 6).

    Groups a batch of neuronodes by domain, runs the domain's feature
    extraction network on each group, concatenates the domain one-hot bias,
    and restores the original batch order.

    Nodes whose domain is ``None`` (or ``'layer'``) act as empty association
    nodes: their feature is a zero vector with a zero domain bias.
    """

    def __init__(self, output_dim, feature_extraction_dict):
        super().__init__()
        self.feature_extraction_networks = nn.ModuleDict(feature_extraction_dict)
        self.type_keys = list(feature_extraction_dict.keys())
        self.output_dim = output_dim

        type_count = len(self.type_keys)
        self.register_buffer('type_bias', torch.eye(type_count))
        self.register_buffer('layer_input', torch.zeros(1, output_dim))
        self.register_buffer('layer_input_bias', torch.zeros(1, type_count))

    def typeEmbedding(self, batch_tree):
        batch_tree_dict = defaultdict(list)
        batch_indices_to_type_indices = {}
        batch_output_dict = {}

        for i in range(len(batch_tree.nodes)):
            domain_name = batch_tree.get_i('t_d', i)
            batch_indices_to_type_indices[i] = (domain_name, len(batch_tree_dict[domain_name]))
            batch_tree_dict[domain_name].append(batch_tree.nodes[i])

        for domain_name, domain_tree_list in batch_tree_dict.items():
            if domain_name is None or domain_name == 'layer':
                type_features = self.layer_input.repeat(len(domain_tree_list), 1)
                type_vector = self.layer_input_bias.repeat(len(domain_tree_list), 1)
            else:
                domain_batch_tree = BatchNeuroTree(domain_tree_list)
                type_features = self.feature_extraction_networks[domain_name](domain_batch_tree)
                type_vector = self.type_bias[self.type_keys.index(domain_name)] \
                    .unsqueeze(0).repeat(len(domain_tree_list), 1)

            batch_output_dict[domain_name] = torch.cat([type_features, type_vector], dim=-1)

        outputs = [
            batch_output_dict[domain_name][domain_index]
            for domain_name, domain_index in batch_indices_to_type_indices.values()
        ]
        return torch.stack(outputs, dim=0).unsqueeze(1)

    def forward(self, batch_tree):
        return self.typeEmbedding(batch_tree)
