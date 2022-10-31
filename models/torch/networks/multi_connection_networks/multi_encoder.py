from collections import defaultdict
import torch
from torch import nn
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from config.option import device



class MultiExtractionConnector(nn.Module):
    def __init__(self, output_dim, feature_extraction_dict):
        super().__init__()
        self.feature_extraction_networks = feature_extraction_dict
        self.type_keys = list(self.feature_extraction_networks.keys())
        self.type_bias = torch.eye(len(self.type_keys)).to(device)
        self.layer_input = torch.zeros(1, output_dim).to(device)
        self.layer_input_bias = torch.zeros(1, len(self.type_keys)).to(device)
        self.output_dim = output_dim  # [128, output_dim]

    # def to(self, device):
    #     for key in self.feature_extraction_networks.keys():
    #         self.feature_extraction_networks[key].to(device)
    #     # self.layer_input.to(device)
    #     # self.layer_input_bias.to(device)
    #     # self.type_bias.to(device)
    #     return self

    def parameters(self):
        param = list(super().parameters())
        for model in self.feature_extraction_networks.values():
            param.extend(model.parameters())
        return param

    def typeEmbedding(self, batch_tree):
        # batch_features = []
        # batch_type_vectors = []

        batch_tree_dict = defaultdict(list)
        batch_domain_indices = defaultdict(list)

        batch_indices_to_type_indices = defaultdict(tuple)
        batch_output_dict = defaultdict(torch.Tensor)

        for i in range(len(batch_tree.nodes)):
            domain_name = batch_tree.get_i('t_d', i)
            # if domain_name is not None:
            #     if domain_name.split('%')[0][-7:] == 'Channel':
            #         domain_name = 'vf'

            batch_indices_to_type_indices[i] = (domain_name, len(batch_tree_dict[domain_name]))
            batch_tree_dict[domain_name].append(batch_tree.nodes[i])
            batch_domain_indices[domain_name].append(i)

        for (domain_name, domain_tree_list), batch_indices in zip(batch_tree_dict.items(), batch_domain_indices.values()):
            if domain_name == None or domain_name == 'layer':
                type_features = self.layer_input.repeat(len(domain_tree_list), 1)
                type_vector = self.layer_input_bias.repeat(len(domain_tree_list), 1)
            else:
                # type_features = torch.stack(batch_x, dim=0)
                batch_association_tree = BatchNeuroTree(domain_tree_list)
                type_features = self.feature_extraction_networks[domain_name](batch_association_tree)  # Leaf Convolution
                type_vector = self.type_bias[self.type_keys.index(domain_name)].unsqueeze(0).repeat(len(domain_tree_list), 1)

            # print(type_features.shape, type_vector.shape)
            batch_output_dict[domain_name] = torch.cat([type_features, type_vector], dim=-1)
        outputs = []
        batch_indices_outputs = []
        for i, (domain_name, domain_index) in batch_indices_to_type_indices.items():
            # print(domain_name, domain_index)
            outputs.append(batch_output_dict[domain_name][domain_index])
            # batch_indices_outputs.append(i)
        outputs = torch.stack(outputs, dim=0).unsqueeze(1)
        return outputs


    def forward(self, batch_tree):
        batch_features = self.typeEmbedding(batch_tree)

        return batch_features
