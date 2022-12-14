import torch
import torch.nn as nn
import torch.nn.functional as F
from models.torch.networks.layers.conv.EGCN import EdgeGraphConvolutionalLayer
from models.torch.networks.layers.conv.EGATs import EdgeGraphAttentionLayer

from models.torch.networks.layers.conv.GCN import GraphConvolutionalLayer
from models.torch.networks.layers.conv.GATs import GraphAttentionLayer
from models.torch.networks.layers.readout.readout_max import MaxpoolReadoutLayer

from models.torch.networks.layers.conv.RNN import RecurrentNeuralNetwork
from models.torch.networks.layers.conv.GRU import GatedRecurrentUnit
from models.torch.networks.layers.conv.ERNN import EdgeRecurrentNeuralNetwork
from models.torch.networks.layers.conv.EGRU import EdgeGatedRecurrentUnit
from models.torch.networks.layers.conv.MPNNs import MPNN
from config.option import device

class RecursiveAssociationNeuralNetworks(nn.Module):

    def __init__(self, input_dim, hidden_dim, edge_dim,
                 feature_extraction_networks, task_networks,
                 version='ran'):

        super().__init__()

        self.return_indices = True

        self.input_dim = input_dim  # [input_dim, 10]
        self.hidden_dim = hidden_dim  # [128, output_dim]
        self.edge_dim = edge_dim

        self.feature_extraction_networks = feature_extraction_networks
        self.task_networks = task_networks

        # self.dropout = nn.Dropout(0.5)

        self.type_count = len(self.feature_extraction_networks.type_keys)

        self.input_dim_with_bias = self.input_dim + self.type_count

        self.zero_hiddens = torch.zeros(self.hidden_dim).to(device)

        if version == 'ran':
            # rnn = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
            # gnn = GraphConvolutionalLayer(self.hidden_dim)
            rnn = EdgeRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim, 1)
            gnn = EdgeGraphConvolutionalLayer(self.hidden_dim, 1)

        elif version == 'raan':
            rnn = EdgeRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim, 1)
            gnn = EdgeGraphAttentionLayer(self.hidden_dim, 1)


        elif version == 'gau':
            # rnn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
            # gnn = GraphConvolutionalLayer(self.hidden_dim)
            rnn = EdgeGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim, 1)
            gnn = EdgeGraphConvolutionalLayer(self.hidden_dim, 1)

        elif version == 'gaau':
            # rnn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
            # gnn = GraphAttentionLayer(self.hidden_dim)
            rnn = EdgeGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim, 1)
            gnn = EdgeGraphAttentionLayer(self.hidden_dim, 1)

        elif version == 'erau':
            rnn = EdgeRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim, edge_dim)
            gnn = EdgeGraphConvolutionalLayer(self.hidden_dim, edge_dim)
        elif version == 'eraau':
            rnn = EdgeRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim, edge_dim)
            gnn = EdgeGraphAttentionLayer(self.hidden_dim, edge_dim)


        elif version == 'egau':
            rnn = EdgeGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim, edge_dim)
            gnn = EdgeGraphConvolutionalLayer(self.hidden_dim, edge_dim)
        elif version == 'egaau':
            rnn = EdgeGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim, edge_dim)
            gnn = EdgeGraphAttentionLayer(self.hidden_dim, edge_dim)

        elif version == 'gmpnn':
            rnn = EdgeGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim, edge_dim)
            gnn = MPNN(self.hidden_dim, edge_dim)



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
                # init_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], self.edge_dim)
                init_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
                hiddens = init_hiddens
            else:
                child_hiddens = batch_tree.get_child_hiddens(self.zero_hiddens)
                # the important thing is that the vector list from the children is used as the input of GTNN.
                # And A_c of the current node is used as children's relational information.
                A_c = batch_tree.get('A_c')
                # print(batch_tree.get('t_d'))

                # hiddens, indices  = self.gnn(A_c, child_hiddens)
                # hiddens, indices = self.readout(hiddens)

                hiddens  = self.gnn(A_c, child_hiddens)
                hiddens, indices = self.readout(hiddens, batch_tree.get('C_count'))
                batch_tree.set('indices', indices)

            hiddens = self.rnn(node_features, hiddens)


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

    # def dfs(self, tree, node_level):
    #     stack = []
    #     result = []
    #     stack.append(tree);  # 0??? ????????? ????????? ???????????? ???????????? ??????
    #     while len(stack) != 0:
    #         _next = stack.pop();  # stack ?????? ?????? ???!
    #         result.append(_next);  # value ??????
    #         for i in range(_next.get_child_count()):
    #             batch_node, batch_indices = _next.get_child(i)
    #             batch_node.batch_child_indices = batch_indices
    #             stack.append(batch_node)
    #
    #     # all_nodes = []
    #     while len(result) != 0:
    #         #         print(resultStack.pop().t);		# ?????? ???????????? ??????
    #         all_batch_tree = result.pop()
    #         # all_nodes.append(all_batch_tree)
    #         batch_tree = all_batch_tree.get_no_calculated()
    #
    #         if len(batch_tree.nodes) == 0:
    #             continue
    #
    #         ## Feature Extraction Process!! ##
    #         node_features = self.feature_extraction_networks(batch_tree)
    #         # the important thing is that the vector list from the children is used as the input of GTNN.
    #         # And A_c of the current node is used as relational information.
    #         if batch_tree.get_child_count() == 0:
    #             # initial hidden
    #             initial_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
    #             hiddens = self.rnn(node_features, initial_hiddens)
    #             batch_tree.set('h', hiddens.squeeze(1))
    #             if node_level:
    #                 self.task_networks(batch_tree)
    #
    #         else:
    #             child_count = batch_tree.get_child_count()
    #             child_hiddens = []
    #             for child_i in range(child_count):
    #                 child_i_batch_tree, has_child_indices = batch_tree.get_child(child_i)
    #                 child_hidden = child_i_batch_tree.get('h')
    #                 child_hidden = batch_tree.batch_assemble(child_hidden, has_child_indices, self.zero_hiddens)
    #                 child_hiddens.append(child_hidden)
    #
    #             child_hiddens = torch.stack(child_hiddens, dim=1)
    #             # print(child_hiddens.shape, 'assemble2')
    #             # the important thing is that the vector list from the children is used as the input of GTNN.
    #             # And A_c of the current node is used as children's relational information.
    #             A_c = batch_tree.get('A_c')
    #             hiddens = self.gnn(A_c, child_hiddens)
    #             hiddens, indices = self.readout(hiddens)
    #             hiddens = self.rnn(node_features, hiddens)
    #             batch_tree.set('h', hiddens.squeeze(1))
    #             batch_tree.set('indices', indices)
    #             if node_level:
    #                 self.task_networks(batch_tree)
    #
    #     return tree.get('h')

#
#
# class RecursiveAssociationNeuralNetworks(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim,
#                  feature_extraction_networks, task_networks,
#                  version='ran'):
#
#         super().__init__()
#
#         self.return_indices = True
#
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
#         self.zero_hiddens = torch.zeros(self.hidden_dim).to(device)
#
#         if version == 'ran':
#             rnn = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphConvolutionalLayer(self.hidden_dim).to(device)
#         elif version == 'raan':
#             rnn = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphAttentionLayer(self.hidden_dim).to(device)
#         elif version == 'gau':
#             rnn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphConvolutionalLayer(self.hidden_dim).to(device)
#         elif version == 'gaau':
#             rnn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphAttentionLayer(self.hidden_dim).to(device)
#         else:
#             assert False  ## ERROR!
#
#         self.readout = MaxpoolReadoutLayer()
#
#         self.rnn = rnn
#         self.gnn = gnn
#
#         # self.layer = AssociationLayer(gnn, self.task_networks).to(device)
#
#     def deepFirstConvolution(self, all_batch_tree, level, all_nodes, node_level):
#         all_batch_tree.visit()
#         batch_tree = all_batch_tree.get_no_calculated()
#
#         if len(batch_tree.nodes) == 0:
#             hiddens = all_batch_tree.get('h')
#             return hiddens
#
#         else:
#             max_child_count = max(batch_tree.get('C_count'))
#
#             child_hiddens = []
#             for i in range(max_child_count):
#                 batch_child, has_child_indices = batch_tree.get_child(number=i)
#
#                 child_hidden = self.deepFirstConvolution(batch_child, level + 1, all_nodes, node_level)
#                 child_hidden = batch_tree.batch_assemble(child_hidden, has_child_indices, self.zero_hiddens)
#                 # print(child_hidden.shape, 'assemble')
#                 child_hiddens.append(child_hidden)
#
#             # Feature Extraction Process!!
#             node_features = self.feature_extraction_networks(batch_tree)
#             # all_nodes.append(batch_tree)
#
#             if max_child_count == 0:  # Leaf Node!
#                 initial_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
#                 hiddens = self.rnn(node_features, initial_hiddens)
#                 # print(hiddens.shape, '??')
#                 batch_tree.set('h', hiddens.squeeze(1))
#                 hiddens = all_batch_tree.get('h')
#                 if node_level:
#                     self.task_networks(batch_tree)
#
#                 # print(torch.stack(hiddens, dim = 0).shape, '???')
#
#                 return hiddens
#
#             else:
#                 child_hiddens = torch.stack(child_hiddens, dim=1)
#                 # print(child_hiddens.shape, 'assemble2')
#                 # the important thing is that the vector list from the children is used as the input of GTNN.
#                 # And A_c of the current node is used as children's relational information.
#                 A_c = batch_tree.get('A_c')
#                 hiddens = self.gnn(A_c, child_hiddens)
#                 hiddens, indices = self.readout(hiddens)
#                 hiddens = self.rnn(node_features, hiddens)
#                 batch_tree.set('h', hiddens.squeeze(1))
#                 batch_tree.set('indices', indices)
#                 if node_level:
#                     self.task_networks(batch_tree)
#                 hiddens = all_batch_tree.get('h')
#                 return hiddens
#
#     def propagation(self, batch_tree, node_level = False):
#
#         level = 0
#         all_nodes = []
#         root_hiddens = self.deepFirstConvolution(batch_tree, level + 1, all_nodes, node_level)
#         return root_hiddens
#
#     def forward(self, batch_tree, node_level = False):
#         return self.propagation(batch_tree, node_level)
#
#     def dfs(self, tree, node_level):
#         stack = []
#         result = []
#         stack.append(tree);  # 0??? ????????? ????????? ???????????? ???????????? ??????
#         while len(stack) != 0:
#             _next = stack.pop();  # stack ?????? ?????? ???!
#             result.append(_next);  # value ??????
#             for i in range(_next.get_child_count()):
#                 batch_node, batch_indices = _next.get_child(i)
#                 batch_node.batch_child_indices = batch_indices
#                 stack.append(batch_node)
#
#         # all_nodes = []
#         while len(result) != 0:
#             #         print(resultStack.pop().t);		# ?????? ???????????? ??????
#             all_batch_tree = result.pop()
#             # all_nodes.append(all_batch_tree)
#             batch_tree = all_batch_tree.get_no_calculated()
#
#             if len(batch_tree.nodes) == 0:
#                 continue
#
#             ## Feature Extraction Process!! ##
#             node_features = self.feature_extraction_networks(batch_tree)
#             # the important thing is that the vector list from the children is used as the input of GTNN.
#             # And A_c of the current node is used as relational information.
#             if batch_tree.get_child_count() == 0:
#                 # initial hidden
#                 initial_hiddens = self.zero_hiddens.repeat(node_features.shape[0], node_features.shape[1], 1)
#                 hiddens = self.rnn(node_features, initial_hiddens)
#                 batch_tree.set('h', hiddens.squeeze(1))
#                 if node_level:
#                     self.task_networks(batch_tree)
#
#             else:
#                 child_count = batch_tree.get_child_count()
#                 child_hiddens = []
#                 for child_i in range(child_count):
#                     child_i_batch_tree, has_child_indices = batch_tree.get_child(child_i)
#                     child_hidden = child_i_batch_tree.get('h')
#                     child_hidden = batch_tree.batch_assemble(child_hidden, has_child_indices, self.zero_hiddens)
#                     child_hiddens.append(child_hidden)
#
#                 child_hiddens = torch.stack(child_hiddens, dim=1)
#                 # print(child_hiddens.shape, 'assemble2')
#                 # the important thing is that the vector list from the children is used as the input of GTNN.
#                 # And A_c of the current node is used as children's relational information.
#                 A_c = batch_tree.get('A_c')
#                 hiddens = self.gnn(A_c, child_hiddens)
#                 hiddens, indices = self.readout(hiddens)
#                 hiddens = self.rnn(node_features, hiddens)
#                 batch_tree.set('h', hiddens.squeeze(1))
#                 batch_tree.set('indices', indices)
#                 if node_level:
#                     self.task_networks(batch_tree)
#
#         return tree.get('h')
