import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torch.networks.layers.conv.GCN import GraphConvolutionalLayer
from models.torch.networks.layers.conv.GATs import GraphAttentionLayer


from models.torch.networks.layers.conv.RNN import RecurrentNeuralNetwork
from models.torch.networks.layers.conv.GRU import GatedRecurrentUnit
# from models.torch.networks.layers.deconv.DRNN import DeconvRecurrentNeuralNetwork
# from models.torch.networks.layers.deconv.DGRU import DeconvGatedRecurrentUnit
from models.torch.networks.layers.readout.readout_unmax import MaxunpoolReadoutLayer
from config.option import device




class DeconvolutionalRecursiveAssociationNetworks(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 multi_restoration_connector, version='ran'):
        super().__init__()

        self.multi_restoration_connector = multi_restoration_connector

        self.input_dim = input_dim  # [input_dim, 10]
        self.hidden_dim = hidden_dim  # [128, output_dim]

        self.type_count = len(self.multi_restoration_connector.type_keys)
        self.input_dim_with_bias = self.input_dim + self.type_count
        self.zero_input_with_bias = torch.zeros(self.input_dim_with_bias).to(device)

        if version == 'ran':
            rdn = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
            gnn = GraphConvolutionalLayer(self.hidden_dim).to(device)
        elif version == 'raan':
            rdn = RecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
            gnn = GraphAttentionLayer(self.hidden_dim).to(device)
        elif version == 'gau':
            rdn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
            gnn = GraphConvolutionalLayer(hidden_dim).to(device)
        elif version == 'gaau':
            rdn = GatedRecurrentUnit(self.input_dim_with_bias, hidden_dim)
            gnn = GraphAttentionLayer(hidden_dim).to(device)
        else:
            assert False  ## ERROR!

        self.rdn = rdn
        self.gnn = gnn
        self.unpool = MaxunpoolReadoutLayer()
        self.out = nn.Linear(self.hidden_dim, self.input_dim_with_bias)

    def propagation(self, inputs, h_root, batch_tree):
        level = 0
        outputs = []
        batch_tree = self.deepFirstDeconvolution(inputs, h_root, batch_tree, level + 1, outputs)
        return batch_tree, outputs

    def deepFirstDeconvolution(self, dx, dh, all_batch_tree, level, outputs):
        all_batch_tree.set_xh(dx, dh)
        all_batch_tree.dvisit()
        batch_tree = all_batch_tree.get_final_visit()
        if len(batch_tree.nodes) == 0:
            return
        discount = torch.Tensor(batch_tree.get('count')).unsqueeze(1).to(device)
        inputs = batch_tree.get('dx')
        hidden = batch_tree.get('dh')
        inputs = torch.stack(inputs, dim = 0)
        hidden = torch.stack(hidden, dim = 0)
        inputs = inputs / discount
        hidden = hidden / discount
        # Feature Restoration Process!!
        hiddens = self.rdn(inputs.unsqueeze(1), hidden.unsqueeze(1))
        node_features = self.out(hiddens)
        node_features = F.layer_norm(node_features, node_features.shape[1:])
        node_features = F.relu(node_features).squeeze(1)
        batch_tree.set('dh', hiddens.squeeze(1))
        # batch_tree.set('dx', node_features)
        # deconv_out, deconv_type_out = self.multi_restoration_connector(node_features.squeeze(1), batch_tree)
        deconv_out, deconv_type_out = self.multi_restoration_connector(node_features, batch_tree)
        outputs.append(deconv_out)

        node_count = batch_tree.get_child_count()
        if node_count == 0:
            return
        else:
            hiddens = hiddens.unsqueeze(1)
            unpooled_hiddens = self.unpool(hiddens, batch_tree)
            A_c = batch_tree.get('A_c')
            child_hiddens = self.gnn(A_c, unpooled_hiddens)

            for i in range(node_count):
                c_hidden = child_hiddens[:, node_count - i - 1]
                batch_child_node, indices = batch_tree.get_child(node_count - i - 1)
                indices_node_features = batch_child_node.batch_disassemble(node_features, indices)
                indices_c_hidden = batch_child_node.batch_disassemble(c_hidden, indices)
                self.deepFirstDeconvolution(indices_node_features, indices_c_hidden, batch_child_node, level + 1, outputs)

            return


    def forward(self, h_root, batch_tree):

        # root_hiddens = batch_tree.get('h')
        # root_hiddens = batch_tree.get('h')
        # batch_tree.set('dh', root_hiddens)
        # inputs = self.zero_input_with_bias.unsqueeze(0).unsqueeze(1).repeat(len(batch_tree),1, 1)
        inputs = self.zero_input_with_bias.unsqueeze(0).repeat(len(batch_tree), 1)
        # batch_tree.set('dnf', inputs)
        return self.propagation(inputs, h_root, batch_tree)



#
# import torch
# import torch.nn as nn
#
# from models.torch.networks.layers.conv.GCN import GraphConvolutionalLayer
# from models.torch.networks.layers.conv.GATs import GraphAttentionLayer
#
# from models.torch.networks.layers.deconv.DRNN import DeconvRecurrentNeuralNetwork
# from models.torch.networks.layers.deconv.DGRU import DeconvGatedRecurrentUnit
# from models.torch.networks.layers.readout.readout_unmax import MaxunpoolReadoutLayer
# from config.option import device
#
#
# class DeconvolutionalRecursiveAssociationNetworks(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim,
#                  multi_restoration_connector, version='ran'):
#         super().__init__()
#
#         self.multi_restoration_connector = multi_restoration_connector
#
#         self.input_dim = input_dim  # [input_dim, 10]
#         self.hidden_dim = hidden_dim  # [128, output_dim]
#
#         self.type_count = len(self.multi_restoration_connector.type_keys)
#         self.input_dim_with_bias = self.input_dim + self.type_count
#
#         if version == 'ran':
#             rdn = DeconvRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphConvolutionalLayer(self.input_dim_with_bias + self.hidden_dim).to(device)
#         elif version == 'raan':
#             rdn = DeconvRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphAttentionLayer(self.input_dim_with_bias + self.hidden_dim).to(device)
#         elif version == 'gau':
#             rdn = DeconvRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
#
#             # rdn = DeconvGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim).to(device)
#             gnn = GraphConvolutionalLayer(self.input_dim_with_bias + hidden_dim).to(device)
#         elif version == 'gaau':
#             rdn = DeconvRecurrentNeuralNetwork(self.input_dim_with_bias, hidden_dim)
#             gnn = GraphConvolutionalLayer(self.input_dim_with_bias + hidden_dim).to(device)
#             # rdn = DeconvGatedRecurrentUnit(self.input_dim_with_bias, hidden_dim).to(device)
#             # gnn = GraphAttentionLayer(self.input_dim_with_bias + hidden_dim).to(device)
#         else:
#             assert False  ## ERROR!
#
#         self.rdn = rdn
#         self.gnn = gnn
#         self.unpool = MaxunpoolReadoutLayer()
#
#     def propagation(self, batch_tree):
#         level = 0
#         outputs = []
#         batch_tree = self.deepFirstDeconvolution(batch_tree, level + 1, outputs)
#         return batch_tree, outputs
#
#     def deepFirstDeconvolution(self, all_batch_tree, level, outputs):
#
#         all_batch_tree.dvisit()
#         batch_tree = all_batch_tree.get_calculated()
#         if len(batch_tree.nodes) == 0:
#             return batch_tree
#
#         node_count = batch_tree.get_child_count()
#         # print('deconv', level, batch_tree.get('count'))
#
#         hidden = batch_tree.get('dh')
#         hidden = torch.stack(hidden, dim = 0)
#
#         # Feature Restoration Process!!
#         node_features, hiddens = self.rdn(hidden)
#         deconv_out, deconv_type_out = self.multi_restoration_connector(node_features, batch_tree)
#         outputs.append(deconv_out)
#
#         if node_count == 0:
#
#             return batch_tree
#
#         else:
#             hiddens = hiddens.unsqueeze(1)
#             unpooled_hiddens = self.unpool(hiddens, batch_tree)
#             A_c = batch_tree.get('A_c')
#             child_hiddens = self.gnn(A_c, unpooled_hiddens)
#
#             for i in range(node_count):
#                 c_hidden = child_hiddens[:, node_count - i - 1]
#                 batch_child_node, indices = batch_tree.get_child(node_count - i - 1)
#                 indices_c_hidden = batch_child_node.batch_disassemble(c_hidden, indices)
#                 batch_child_node.set('dh', indices_c_hidden)
#                 self.deepFirstDeconvolution(batch_child_node, level + 1, outputs)
#                 # indices_c_hidden = batch_child_node.batch_disassemble(c_hidden, indices)
#                 # self.deepFirstDeconvolution(indices_c_hidden, batch_child_node, level + 1, outputs)
#
#             return batch_tree
#
#
#     def forward(self, batch_tree):
#         root_hiddens = batch_tree.get('h')
#         batch_tree.set('dh', root_hiddens)
#         return self.propagation(batch_tree)
