from collections import defaultdict
from data_structures.batch_neurotree import BatchNeuroTree
from torch import nn
import torch


class MultiSubTaskConnector(nn.Module):
    def __init__(self, hidden_dim, task_networks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_networks = task_networks
        for modelname in self.task_networks.keys():
            self.__setattr__(f'{modelname}', task_networks[modelname])

        self._reset_params()

        # self.batch_tools = BatchAssociationTree()


    def _reset_params(self):
        for seq_layers in self.task_networks.values():
            for i, net in enumerate(seq_layers.parameters()):
                if type(net) == nn.Linear:
                    torch.nn.init.xavier_uniform_(net.weight.data)


    def forward(self, h_root, batchNeuroTree):
        return self.multi_tasking(h_root, batchNeuroTree)


    def loss(self, taskname, domains, preds, targets):
        return self.task_networks[taskname].loss(domains, preds, targets)


    def multi_tasking(self, h_root, allBatchNeuroTree: BatchNeuroTree):
        batch_tree_dict = defaultdict(list)
        batch_hidden_dict = defaultdict(list)
        batch_task_indices = defaultdict(list)
        batch_indices_to_task_indices = defaultdict(tuple)
        batch_output_dict = defaultdict(torch.Tensor)
        for i in range(len(allBatchNeuroTree.nodes)):
            task_name = allBatchNeuroTree.getSubtask(i)
            batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
            batch_tree_dict[task_name].append(allBatchNeuroTree.nodes[i])
            batch_hidden_dict[task_name].append(h_root[i])
            batch_task_indices[task_name].append(i)

        for task_name, task_tree_list in batch_tree_dict.items():
            batch_neurotree = BatchNeuroTree(task_tree_list)
            if task_name is not None:
                task_output = self.task_networks[task_name](
                    batch_hidden_dict[task_name],
                    batch_neurotree
                )
                batch_neurotree.setPredicts(task_output)

            else:
                tree_count = len(task_tree_list)
                task_output = [None for _ in range(tree_count)]

            batch_output_dict[task_name] = task_output

        outputs = []
        # batch_indices_outputs = []
        for i, (task_name, task_index) in batch_indices_to_task_indices.items():
            outputs.append(batch_output_dict[task_name][task_index])
            # batch_indices_outputs.append(i)
        # outputs = torch.stack(outputs, dim=0).unsqueeze(1)
        # print(outputs.shape)
        # batch_tree.set('pred', outputs)
        return outputs





    # def forward(self, tasks, hiddens):
    #     outputs, batch_task_indices = self.works(tasks, hiddens)
    #     return outputs

    def recursive_loss(self, tree):
        loss, correct, loss_sum = self.loss(tree)
        for child in tree.C:
            child_loss, child_correct, child_loss_sum = self.recursive_loss(child)
            #             loss += child_loss
            loss = self.dict_sum(loss, child_loss)
            correct = self.dict_sum(correct, child_correct)
            loss_sum += child_loss_sum
        return loss, correct, loss_sum

    def dict_sum(self, dict_1, dict_2):

        #         dict_1 << dict_2
        for key1 in dict_2.keys():
            for key2, value in dict_1[key1].items():
                dict_1[key2] += value

        return dict_1
