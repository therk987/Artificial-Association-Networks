from collections import defaultdict
from models.torch.dataloader.batch_neurotree import BatchNeuroTree
from torch import nn
import torch

class MultiTaskConnector(nn.Module):
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

    def forward(self, batch_tree):

        return self.multi_tasking(batch_tree)

    def multi_tasking(self, batch_tree):
        batch_tree_dict = defaultdict(list)
        batch_task_indices = defaultdict(list)
        batch_indices_to_task_indices = defaultdict(tuple)
        batch_output_dict = defaultdict(torch.Tensor)
        for i in range(len(batch_tree.nodes)):
            task_name = batch_tree.get_i('t_t', i)
            batch_indices_to_task_indices[i] = (task_name, len(batch_tree_dict[task_name]))
            batch_tree_dict[task_name].append(batch_tree.nodes[i])
            batch_task_indices[task_name].append(i)

        for (task_name, task_tree_list), batch_indices in zip(batch_tree_dict.items(), batch_task_indices.values()):
            batch_association_tree = BatchNeuroTree(task_tree_list)
            if task_name is not None and task_name != 'autoencoder':
                task_output = self.task_networks[task_name](batch_association_tree)  # Leaf Convolution
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



    def working(self, treeset):

        task_indices = self.get_task_indices(treeset)

        task_tree_dict = defaultdict(list)
        result_tree_dict = defaultdict(list)
        result_outputs_dict = defaultdict(list)

        for task_name in task_indices.keys():
            indices = task_indices[task_name]
            for ts_idx, bt_idx in indices:
                # tree = self.batch_tools.pop(treeset[ts_idx], bt_idx, self.task_networks[task_name].only_root)
                tree = treeset[ts_idx].nodes[bt_idx]
                # print(tree.__dict__.keys(), 'working!')
                task_tree_dict[task_name].append(tree)
            # batch_task_tree = self.batch_tools.mergeBatchTree(task_tree_dict[task_name])
            batch_task_tree = BatchNeuroTree(task_tree_dict[task_name])
            result_tree_dict[task_name], result_outputs_dict[task_name] = self.task_networks[task_name](batch_task_tree)


        return result_tree_dict, result_outputs_dict

    def get_task_indices(self, treeset):
        task_indices = defaultdict(list)
        for k, tree in enumerate(treeset):
            for j, batch_task in enumerate(tree.get('t_t')):
                if batch_task is not None:
                    for task in batch_task:
                        task_indices[task].append((k, j))

        return task_indices



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


