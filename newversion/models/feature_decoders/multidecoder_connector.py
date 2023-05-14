import torch
import torch.nn as nn
from collections import defaultdict
from data_structures.batch_neurotree import BatchNeuroTree
from config.option import device


class MultiRestorationConnector(nn.Module):

    def __init__(self, feature_dim, feature_restoration_dict):
        super().__init__()

        self.feature_restoration_dict = feature_restoration_dict
        for modelname in self.feature_restoration_dict.keys():
            self.__setattr__(f'{modelname}', feature_restoration_dict[modelname])

        self.type_keys = list(self.feature_restoration_dict.keys())
        self.type_bias = torch.eye(len(self.type_keys)).to(device)
        self.layer_input = torch.zeros(1, feature_dim).to(device)
        self.layer_input_bias = torch.zeros(1, len(self.type_keys)).to(device)
        self.feature_dim = feature_dim  # [128, output_dim]

    # def parameters(self):
    #     param = list(super().parameters())
    #
    #     for model in self.feature_restoration_dict.values():
    #         param.extend(model.parameters())
    #     return param

    def loss(self, domain_name, conv_x, deconv_x):
        self.feature_restoration_dict[domain_name].loss(conv_x, deconv_x)

    def typeDecoding(self, dx, batch_tree):

        batch_tree_dict = defaultdict(list)
        batch_dx_dict = defaultdict(list)


        batch_output_dict = defaultdict(torch.Tensor)
        batch_type_label_dict = defaultdict(torch.Tensor)

        batch_domain_indices = defaultdict(list)

        batch_indices_to_type_indices = defaultdict(tuple)

        for i in range(len(batch_tree.nodes)):
            domain_name = batch_tree.get_i('t_d', i)
            batch_indices_to_type_indices[i] = (domain_name, len(batch_tree_dict[domain_name]))
            batch_tree_dict[domain_name].append(batch_tree.nodes[i])
            batch_dx_dict[domain_name].append(dx[i])
            batch_domain_indices[domain_name].append(i)


        for (domain_name, domain_tree_list), batch_indices in zip(batch_tree_dict.items(), batch_domain_indices.values()):
            if domain_name == None:
                #                 continue
                type_features = [None for _ in range(len(domain_tree_list))]
                type_labels = [None for _ in range(len(domain_tree_list))]

            else:

                # dx = batch_dx_dict[domain_name]

                batch_association_tree = BatchNeuroTree(domain_tree_list)
                # print(domain_name, type_features.shape)
                type_features = self.feature_restoration_dict[domain_name](batch_association_tree)  ## decoding
                type_labels = [self.type_keys.index(domain_name) for _ in range(len(domain_tree_list))]

            batch_output_dict[domain_name] = type_features
            batch_type_label_dict[domain_name] = type_labels

        outputs = []
        type_outputs = []
        # batch_indices_outputs = []

        for i, (domain_name, domain_index) in batch_indices_to_type_indices.items():
            batch_tree.set_i('decoder-pred', batch_output_dict[domain_name][domain_index], i)
            # batch_tree.set_i('domain_pred', batch_type_vec_dict[domain_name][domain_index], i)
            batch_tree.set_i('domain_label', batch_type_label_dict[domain_name][domain_index], i)
            outputs.append(batch_output_dict[domain_name][domain_index])
            # type_outputs.append((batch_type_vec_dict[domain_name][domain_index],
            #                      batch_type_label_dict[domain_name][domain_index]))
            # batch_indices_outputs.append(i)
        return outputs, type_outputs


    def forward(self, dx, batch_tree):
        batch_features, type_outputs = self.typeDecoding(dx, batch_tree)
        return batch_features, type_outputs

#
# class MultiRestorationConnector(nn.Module):
#
#     def __init__(self, feature_dim, feature_restoration_dict):
#         super().__init__()
#
#         self.feature_restoration_dict = feature_restoration_dict
#         self.type_keys = list(self.feature_restoration_dict.keys())
#         self.type_bias = torch.eye(len(self.type_keys)).to(device)
#         self.layer_input = torch.zeros(1, feature_dim).to(device)
#         self.layer_input_bias = torch.zeros(1, len(self.type_keys)).to(device)
#         self.feature_dim = feature_dim  # [128, output_dim]
#
#     def parameters(self):
#         param = list(super().parameters())
#
#         for model in self.feature_restoration_dict.values():
#             param.extend(model.parameters())
#         return param
#
#     def loss(self, domain_name, conv_x, deconv_x):
#         self.feature_restoration_dict[domain_name].loss(conv_x, deconv_x)
#
#     def typeDecoding(self, node_hiddens, batch_tree):
#
#         batch_tree_dict = defaultdict(list)
#         batch_feature_dict = defaultdict(list)
#
#         batch_domain_vec_dict = defaultdict(list)
#
#         batch_output_dict = defaultdict(torch.Tensor)
#         batch_type_vec_dict = defaultdict(torch.Tensor)
#         batch_type_label_dict = defaultdict(torch.Tensor)
#
#         batch_domain_indices = defaultdict(list)
#
#         batch_indices_to_type_indices = defaultdict(tuple)
#
#         node_features, type_infos = node_hiddens.split([self.feature_dim, len(self.type_keys)], dim=-1)
#
#         for i in range(len(batch_tree.nodes)):
#             domain_name = batch_tree.get_i('t_d', i)
#
#             batch_indices_to_type_indices[i] = (domain_name, len(batch_tree_dict[domain_name]))
#
#             batch_tree_dict[domain_name].append(batch_tree.nodes[i])
#             batch_feature_dict[domain_name].append(node_features[i])
#
#             batch_domain_vec_dict[domain_name].append(type_infos[i])
#             batch_domain_indices[domain_name].append(i)
#
#
#         for (domain_name, domain_tree_list), batch_indices in zip(batch_tree_dict.items(), batch_domain_indices.values()):
#             if domain_name == None:
#                 #                 continue
#                 type_features = [None for _ in range(len(domain_tree_list))]
#                 type_vectors = [None for _ in range(len(domain_tree_list))]
#                 type_labels = [None for _ in range(len(domain_tree_list))]
#
#             else:
#
#                 batch_x = batch_feature_dict[domain_name]
#                 type_features = torch.stack(batch_x, dim=0)
#                 batch_association_tree = BatchNeuroTree(domain_tree_list)
#                 # print(domain_name, type_features.shape)
#                 type_features = self.feature_restoration_dict[domain_name](type_features, batch_association_tree)  ## decoding
#
#                 type_vectors = batch_domain_vec_dict[domain_name]
#                 type_vectors = torch.stack(type_vectors, dim=0)  ## decoded type info
#                 type_labels = [self.type_keys.index(domain_name) for _ in range(len(domain_tree_list))]
#
#             batch_output_dict[domain_name] = type_features
#
#             batch_type_vec_dict[domain_name] = type_vectors
#             batch_type_label_dict[domain_name] = type_labels
#
#         outputs = []
#         type_outputs = []
#         batch_indices_outputs = []
#
#         for i, (domain_name, domain_index) in batch_indices_to_type_indices.items():
#             batch_tree.set_i('pred', batch_output_dict[domain_name][domain_index], i)
#             batch_tree.set_i('domain_pred', batch_type_vec_dict[domain_name][domain_index], i)
#             batch_tree.set_i('domain_label', batch_type_label_dict[domain_name][domain_index], i)
#
#             outputs.append(batch_output_dict[domain_name][domain_index])
#             type_outputs.append((batch_type_vec_dict[domain_name][domain_index],
#                                  batch_type_label_dict[domain_name][domain_index]))
#
#             batch_indices_outputs.append(i)
#
#
#         return outputs, type_outputs, batch_indices_outputs
#
#
#     def forward(self, batch_domain, batch_inputs):
#         batch_features, type_outputs, batch_indices = self.typeDecoding(batch_domain, batch_inputs)
#         return batch_features, type_outputs
