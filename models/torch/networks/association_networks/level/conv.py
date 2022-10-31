import torch
import torch.nn as nn
from models.torch.networks.layers.conv.GCN import GraphConvolutionalLayer
from models.torch.networks.layers.conv.GATs import GraphAttentionLayer
from models.torch.networks.association_layers.conv import AssociationLayer
from models.torch.networks.layers.conv.RNN import RecurrentNeuralNetwork
from models.torch.networks.layers.conv.GRU import GatedRecurrentUnit
from models.torch.networks.layers.conv.EGCN import EdgeGraphConvolutionalLayer
from models.torch.networks.layers.conv.EGATs import EdgeGraphAttentionLayer
from models.torch.networks.layers.conv.ERNN import EdgeRecurrentNeuralNetwork

from models.torch.networks.layers.readout.readout_max import MaxpoolReadoutLayer
from config.option import device




class LevelAssociationNeuralNetworks(nn.Module):

    def __init__(self, input_dim, hidden_dim, edge_dim,
                 feature_extraction_networks, task_networks, max_lv,
                 version='lan'):

        super().__init__()

        self.return_indices = True

        self.input_dim = input_dim  # [input_dim, 10]
        self.hidden_dim = hidden_dim  # [128, output_dim]

        self.feature_extraction_networks = feature_extraction_networks
        self.task_networks = task_networks

        self.type_count = len(self.feature_extraction_networks.type_keys)

        self.input_dim_with_bias = self.input_dim + self.type_count

        self.zero_hiddens = torch.zeros(self.hidden_dim).to(device)
        self.max_lv = max_lv

        rnn = []
        gnn = []


        if version == 'lan':
            for i in range(max_lv):
                self.__setattr__(f'rnn{i}', EdgeRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim, edge_dim))
                self.__setattr__(f'gnn{i}', EdgeGraphConvolutionalLayer(self.hidden_dim, edge_dim))
                rnn.append(self.__getattr__(f'rnn{i}'))
                gnn.append(self.__getattr__(f'gnn{i}'))

                # self.__dict__[f'rnn{i}'] = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
                # self.__dict__[f'gnn{i}'] = GraphConvolutionalLayer(self.hidden_dim)
                # rnn.append(self.__dict__[f'rnn{i}'])
                # gnn.append(self.__dict__[f'gnn{i}'])

            # rnn = [RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim).to(device) for _ in range(max_lv)]
            # gnn = [GraphConvolutionalLayer(self.hidden_dim).to(device) for _ in range(max_lv)]

        elif version == 'laan':
            for i in range(max_lv):
                # self.__setattr__(f'A{i}', A[i])
                self.__setattr__(f'rnn{i}', EdgeRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim, edge_dim))
                self.__setattr__(f'gnn{i}', EdgeGraphAttentionLayer(self.hidden_dim, edge_dim))
                rnn.append(self.__getattr__(f'rnn{i}'))
                gnn.append(self.__getattr__(f'gnn{i}'))
                # self.__dict__[f'rnn{i}'] = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
                # self.__dict__[f'gnn{i}'] = GraphAttentionLayer(self.hidden_dim)

            # rnn = [RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim).to(device) for _ in range(max_lv)]
            # gnn = [GraphAttentionLayer(self.hidden_dim).to(device) for _ in range(max_lv)]
        # elif version == 'gau':
        #     rnn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
        #     gnn = GraphConvolutionalLayer(self.hidden_dim).to(device)
        # elif version == 'gaau':
        #     rnn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
        #     gnn = GraphAttentionLayer(self.hidden_dim).to(device)
        else:
            assert False  ## ERROR!



        self.readout = MaxpoolReadoutLayer()

        self.rnn = rnn
        self.gnn = gnn

        # self.layer = AssociationLayer(gnn, self.task_networks).to(device)

    def deepFirstConvolution(self, all_batch_tree, level, all_nodes, node_level):
        all_batch_tree.visit()
        batch_tree = all_batch_tree.get_no_calculated()
        if len(batch_tree.nodes) == 0:
            return
        else:
            max_child_count = max(batch_tree.get('C_count'))
            for i in range(max_child_count):
                batch_child, has_child_indices = batch_tree.get_child(number=i)
                self.deepFirstConvolution(batch_child, level + 1, all_nodes, node_level)
           # Feature Extraction Process!!
            node_features = self.feature_extraction_networks(batch_tree)


            # all_nodes.append(batch_tree)
            if max_child_count == 0:  # Leaf Node!
                init_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
                hiddens = init_hiddens
            else:
                # child_hiddens = batch_tree.get_child_hiddens()
                child_hiddens = batch_tree.get_child_hiddens(self.zero_hiddens)
                # the important thing is that the vector list from the children is used as the input of GTNN.
                # And A_c of the current node is used as children's relational information.
                A_c = batch_tree.get('A_c')
                # print(batch_tree.get('t_d'))

                # hiddens, indices = self.gnn[level](A_c, child_hiddens)
                hiddens = self.gnn[level](A_c, child_hiddens)
                # hiddens = self.gnn[level](A_c, child_hiddens)
                hiddens, indices = self.readout(hiddens, batch_tree.get('C_count'))
                batch_tree.set('indices', indices)

            hiddens = self.rnn[level](node_features, hiddens)
            batch_tree.set('h', hiddens.squeeze(1))
            if level == 0:
                batch_tree.set_xh(node_features.squeeze(1), hiddens.squeeze(1))

            if node_level:
                self.task_networks(hiddens, batch_tree)
            # hiddens = all_batch_tree.get('h')
            return
            # return hiddens

    def propagation(self, batch_tree, node_level = False):

        level = 0
        all_nodes = []
        self.deepFirstConvolution(batch_tree, level, all_nodes, node_level)

        root_hiddens = batch_tree.get('h')
        return root_hiddens

    def forward(self, batch_tree, node_level = False):
        return self.propagation(batch_tree, node_level)



#
# class LevelAssociationNeuralNetworks(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim,
#                  feature_extraction_networks, task_networks,
#                  version='lan', max_lv=0):
#
#         super().__init__()
#
#         self.return_indices = True
#         self.input_dim = input_dim  # [input_dim, 10]
#         self.hidden_dim = hidden_dim  # [128, output_dim]
#
#         self.feature_extraction_networks = feature_extraction_networks
#         self.task_networks = task_networks
#
#         self.type_count = len(self.feature_extraction_networks.type_keys)
#
#         self.input_dim_with_bias = self.input_dim + self.type_count
#
#         self.zero_hiddens = torch.zeros(1, 1, self.hidden_dim).to(device)
#         self.zero_inputs_hiddens = torch.zeros(1, 1, self.input_dim_with_bias + self.hidden_dim).to(device)
#
#         self.max_lv = max_lv
#
#         self.layers = []
#         for lv in range(self.max_lv):
#             if version == 'lan':
#                 gnn = GraphConvolutionalLayer(self.input_dim_with_bias, hidden_dim).to(device)
#             elif version == 'laan':
#                 gnn = GraphAttentionLayer(self.input_dim_with_bias, hidden_dim).to(device)
#             elif version == 'reslan':
#                 gnn = GraphConvolutionalLayer(self.input_dim_with_bias, hidden_dim).to(device)
#             elif version == 'reslaan':
#                 gnn = GraphConvolutionalLayer(self.input_dim_with_bias, hidden_dim).to(device)
#             else:
#                 assert False  ## ERROR!
#
#             layer = AssociationLayer(gnn, self.feature_extraction_networks, self.task_networks).to(device)
#             self.layers.append(layer)
#
#     def deepFirstConvolution(self, tree, level, graph_tree):
#
#         child_features = []
#         child_hiddens = []
#
#         for i, t in enumerate(tree.C):
#             child_feature, child_hidden = self.deepFirstConvolution(t, level + 1, graph_tree)
#             child_features.append(child_feature)
#             child_hiddens.append(child_hidden)
#
#         # Feature Extraction Process!!
#         node_features = self.feature_extraction_networks(tree.t_d, tree.x)
#
#         if len(tree.C) == 0:  # Leaf Node!
#             initial_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
#             return node_features, initial_hiddens
#
#         else:
#             child_features = torch.cat(child_features, dim=1)
#             child_hiddens = torch.cat(child_hiddens, dim=1)
#             # the important thing is that the vector list from the children is used as the input of GTNN.
#             # And A_c of the current node is used as children's relational information.
#             hiddens = self.layers[level](tree, child_features, child_hiddens)
#
#             return node_features, hiddens
#
#     def propagation(self, tree):
#
#         level = 0
#         graph_tree = []
#         node_features, hiddens = self.deepFirstConvolution(tree.root, level + 1, graph_tree)
#         graphs = torch.eye(node_features.shape[1]).to(device).unsqueeze(0).repeat(node_features.shape[0], 1, 1)
#         root_out, task_out = self.layers[level](tree.root, node_features, hiddens, root=True)
#
#         root_out = root_out.squeeze(1)
#
#         return root_out
#
#     def forward(self, tree):
#         return self.dfs(tree.root).squeeze(1)
#
#     def dfs(self, tree):
#         stack = []
#         result = []
#
#         stack.append(tree);  # 0번 노드를 루트로 가정하고 루트부터 순회
#         while len(stack) != 0:
#             _next = stack.pop();  # stack 에서 하나 뺌!
#             result.append(_next);  # value 저장
#             stack.extend(_next.C)
#
#         while len(result) != 0:
#             node = result.pop()
#             node_features = self.feature_extraction_networks(node.t_d, node.x)
#
#             # the important thing is that the vector list from the children is used as the input of GTNN.
#             # And A_c of the current node is used as relational information.
#             if len(node.C) == 0:
#                 # initial hidden
#                 hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
#
#             else:
#                 child_features = torch.cat(node.child_x, dim=1)
#                 child_hiddens = torch.cat(node.child_h, dim=1)
#                 hiddens, graphs = self.layers[node.lv](node, child_features, child_hiddens, root=False)
#                 node.child_x.clear()
#                 node.child_h.clear()
#
#             ## conv
#             if node.parent is None:
#                 ## root!
#                 I = torch.eye(node_features.shape[1]).to(device).unsqueeze(0).repeat(node_features.shape[0], 1, 1)
#                 root_out, root_graphs = self.layers[node.lv + 1](node, node_features, hiddens, root=True)
#                 return root_out
#
#             else:
#                 node.parent.child_x.append(node_features)
#                 node.parent.child_h.append(hiddens)
#
#
