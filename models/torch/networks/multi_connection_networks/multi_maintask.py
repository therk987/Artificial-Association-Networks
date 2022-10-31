from collections import defaultdict
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from torch import nn
import torch

from collections import defaultdict
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from torch import nn
import torch

class MultiMainTaskConnector(nn.Module):
    def __init__(self, hidden_dim, task_networks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_networks = task_networks
        self._reset_params()
        # self.batch_tools = BatchAssociationTree()
    def _reset_params(self):
        for seq_layers in self.task_networks.values():
            for i, net in enumerate(seq_layers.parameters()):
                if type(net) == nn.Linear:
                    torch.nn.init.xavier_uniform_(net.weight.data)

    def parameters(self):
        param = list(super().parameters())
        for model in self.task_networks.values():
            param.extend(model.parameters())
        return param

    def forward(self, h_root, batchNeuroTree, tasks):
        return self.multi_tasking(h_root, batchNeuroTree, tasks)

    def multi_tasking(self, h_root, allBatchNeuroTree, tasks):
        batch_tree_dict = defaultdict(list)
        batch_hidden_dict = defaultdict(list)
        batch_indices_to_task_indices = defaultdict(tuple)
        batch_output_dict = defaultdict(torch.Tensor)
        for i in range(len(allBatchNeuroTree.nodes)):
            task_name = tasks[i]
            batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
            batch_tree_dict[task_name].append(allBatchNeuroTree.nodes[i])
            batch_hidden_dict[task_name].append(h_root[i])
        for task_name, task_tree_list in batch_tree_dict.items():
            batch_neurotree = BatchNeuroTree(task_tree_list)
            batch_hiddens = torch.stack(batch_hidden_dict[task_name], dim=0)
            task_output = self.task_networks[task_name](
                batch_hiddens,
                batch_neurotree
            )  # Leaf Convolution
            batch_output_dict[task_name] = task_output
        outputs = []
        for i, (task_name, task_index) in batch_indices_to_task_indices.items():
            outputs.append(batch_output_dict[task_name][task_index])
        return outputs

    def loss(self, taskname, domains, preds, targets):
        return self.task_networks[taskname].loss(domains, preds, targets)



    def dict_sum(self, dict_1, dict_2):
        #         dict_1 << dict_2
        for key1 in dict_2.keys():
            for key2, value in dict_1[key1].items():
                dict_1[key2] += value
        return dict_1

#
# class MultiMainTaskConnector(nn.Module):
#     def __init__(self, hidden_dim, task_networks):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.task_networks = task_networks
#         self._reset_params()
#
#         # self.batch_tools = BatchAssociationTree()
#
#
#     def _reset_params(self):
#         for seq_layers in self.task_networks.values():
#             for i, net in enumerate(seq_layers.parameters()):
#                 if type(net) == nn.Linear:
#                     torch.nn.init.xavier_uniform_(net.weight.data)
#
#     def parameters(self):
#         param = list(super().parameters())
#
#         for model in self.task_networks.values():
#             param.extend(model.parameters())
#         return param
#
#     def forward(self, h_root, batchNeuroTree, tasks):
#
#         return self.multi_tasking(h_root, batchNeuroTree, tasks)
#
#     def multi_tasking(self, h_root, allBatchNeuroTree, tasks):
#         batch_tree_dict = defaultdict(list)
#         batch_hidden_dict = defaultdict(list)
#         batch_indices_to_task_indices = defaultdict(tuple)
#         batch_output_dict = defaultdict(torch.Tensor)
#         for i in range(len(allBatchNeuroTree.nodes)):
#             task_name = tasks[i]
#             batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
#             batch_tree_dict[task_name].append(allBatchNeuroTree.nodes[i])
#             batch_hidden_dict[task_name].append(h_root[i])
#
#         for task_name, task_tree_list in batch_tree_dict.items():
#             batch_neurotree = BatchNeuroTree(task_tree_list)
#             batch_hiddens = torch.stack(batch_hidden_dict[task_name], dim=0)
#
#             task_output = self.task_networks[task_name](
#                 batch_hiddens,
#                 batch_neurotree
#             )  # Leaf Convolution
#
#             batch_output_dict[task_name] = task_output
#
#         outputs = []
#         # batch_indices_outputs = []
#         for i, (task_name, task_index) in batch_indices_to_task_indices.items():
#             outputs.append(batch_output_dict[task_name][task_index])
#             # batch_indices_outputs.append(i)
#         # outputs = torch.stack(outputs, dim=0).unsqueeze(1)
#         # print(outputs.shape)
#         # batch_tree.set('pred', outputs)
#         return outputs
#
#
#
#     def loss(self, taskname, domains, preds, targets):
#         return self.task_networks[taskname].loss(domains, preds, targets)
#


    # def forward(self, tasks, hiddens):
    #     outputs, batch_task_indices = self.works(tasks, hiddens)
    #     return outputs
    #
    # def recursive_loss(self, tree):
    #     loss, correct, loss_sum = self.loss(tree)
    #     for child in tree.C:
    #         child_loss, child_correct, child_loss_sum = self.recursive_loss(child)
    #         #             loss += child_loss
    #         loss = self.dict_sum(loss, child_loss)
    #         correct = self.dict_sum(correct, child_correct)
    #         loss_sum += child_loss_sum
    #     return loss, correct, loss_sum

    # def dict_sum(self, dict_1, dict_2):
    #
    #     #         dict_1 << dict_2
    #     for key1 in dict_2.keys():
    #         for key2, value in dict_1[key1].items():
    #             dict_1[key2] += value
    #
    #     return dict_1

    # def loss(self, result_tree_dict):
    #     loss_sum = 0
    #     loss = defaultdict(int)
    #     correct = defaultdict(int)
    #
    #     for task in result_tree_dict.keys():
    #         task_loss, task_correct = self.task_networks[task].loss(result_tree_dict[task])
    #         loss[task] += task_loss
    #         loss_sum += task_loss
    #         if task_correct is not None:
    #             correct[task] += task_correct
    #
    #     return loss, correct, loss_sum


